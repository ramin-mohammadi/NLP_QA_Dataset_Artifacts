import os
import numpy as np
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput
from typing import Tuple
from tqdm.auto import tqdm
import csv


QA_MAX_ANSWER_LENGTH = 30


# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        'accuracy': (np.argmax(
            eval_preds.predictions,
            axis=1) == eval_preds.label_ids).astype(
            np.float32).mean().item()
    }


# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    # tokenize both questions and the corresponding context
    # if the context length is longer than max_length, we split it to several
    # chunks of max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),  # NOTE CHANGE/get rid of min
        #stride=max_seq_length // 2,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to its corresponding example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position
    # in the original context. This will help us compute the start_positions
    # and end_positions to get the final answer string.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        # We will label features not containing the answer the index of the CLS token.
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        # from the feature idx to sample idx
        sample_index = sample_mapping[i]
        # get the answer for a feature
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and \
                        offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(
                    token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_dataset_qa(examples, tokenizer):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128), # get rid of min
        #stride=max_seq_length // 2,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# This function uses start and end position scores predicted by a question answering model to
# select and extract the predicted answer span from the context.
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py
def postprocess_qa_predictions(examples,
                               features,
                               predictions: Tuple[np.ndarray, np.ndarray],
                               n_best_size: int = 20):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[
            example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits
            # to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                            -1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[
                          -1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or \
                            end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH:
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0],
                                        offset_mapping[end_index][1]),
                            "score": start_logits[start_index] +
                                     end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"],
                             reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0})

        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions


# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples

        self.epoch = 0 # to track epoch number for data cartography CSV

    def evaluate(self,
                 eval_dataset=None,  # denotes the dataset after mapping
                 eval_examples=None,  # denotes the raw dataset
                 ignore_keys=None,  # keys to be ignored in dataset
                 metric_key_prefix: str = "eval"
                 ):
        print("EVALUATE HAS BEEN CALLED")
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # compute the raw predictions (start_logits and end_logits)
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # post process the raw predictions to get the final prediction
            # (from start_logits, end_logits to an answer string)
            eval_preds = postprocess_qa_predictions(eval_examples,
                                                    eval_dataset,
                                                    output.predictions)
            formatted_predictions = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds.items()]
            references = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples]

            # compute the metrics according to the predictions and references
            metrics = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions,
                               label_ids=references)
            )

            # METRICS:  {'exact_match': 77.36991485335857, 'f1': 85.41716947823684}
            print("METRICS: ", metrics)

            #################### Data Cartography F1 scores write to csv
            # first check if csv file exists
            # if not os.path.exists('tools/cartography_out/per_example_f1.csv'):
            #     os.makedirs('tools/cartography_out', exist_ok=True)
            #     with open('tools/cartography_out/per_example_f1.csv', 'w', newline='', encoding='utf-8') as csvfile:
            #         writer = csv.writer(csvfile)
            #         writer.writerow(["example_id", "epoch", "f1"])

            # with open('tools/cartography_out/per_example_f1.csv', 'a', newline='', encoding='utf-8') as csvfile:
            #     writer = csv.writer(csvfile)
            #     epoch_num = self.epoch
            #     for i in range(len(formatted_predictions)):
            #         #ex_id = formatted_predictions[i]['id']
            #         ex_id = eval_examples[i]['id']
            #         # pred = formatted_predictions[i]['prediction_text']
            #         # refs = references[i]['answers']['text']
            #         #f1 = max_f1_against_refs(pred, refs) if refs else 0.0
            #         f1 = self.compute_metrics(
            #             EvalPrediction(predictions=[formatted_predictions[i]],
            #                            label_ids=[references[i]])
            #         )['f1'] / 100.0  # convert to [0,1] range
            #         writer.writerow([ex_id, epoch_num, f1])
            # self.epoch += 1
            #####################

            
            ################ DEBUGGING PRINTS
            # for i in range(len(formatted_predictions)):
            #     print(f"\nExample {i}:")
            #     print("Question:\n", eval_examples[i]['question'])
            #     print("Prediction:\n", formatted_predictions[i]['prediction_text'])
            #     print("References:\n", references[i]['answers']['text'])
                
            # print(len(eval_examples), "examples")
            # print(len(formatted_predictions), "predictions")
            # print(len(references), "references")
            
            # for i in range(len(formatted_predictions)):
            #     print(f"\nExample {i}:")
            #     print("Question:\n", eval_examples[i])
            #     print("Prediction:\n", formatted_predictions[i]['prediction_text'])
            #     print("References:\n", references[i]['answers'])
                
            # print(len(eval_examples), "examples")
            # print(len(formatted_predictions), "predictions")
            # print(len(references), "references")
            
            # # All examples
            # output_dir = getattr(self.args, "output_dir", ".")
            # debug_path = os.path.join(output_dir, "eval_output.txt")
            # with open(debug_path, "w", encoding="utf-8") as f:
            #     for i in range(len(formatted_predictions)):
            #         f.write(f"Example {i}:\n")
            #         f.write("Context:\n" + eval_examples[i]['context'] + "\n")
            #         f.write("Question:\n" + eval_examples[i]['question'] + "\n")
            #         f.write("Prediction:\n" + formatted_predictions[i]['prediction_text'] + "\n")
            #         f.write("References:\n" + str(references[i]['answers']['text']) + "\n\n")
            #     f.write(f"{len(eval_examples)} examples\n")
            #     f.write(f"{len(formatted_predictions)} predictions\n")
            #     f.write(f"{len(references)} references\n\n")
            
            # # incorrect predictions
            # incorrect_path = os.path.join(output_dir, "incorrect_predictions.txt")
            # count_total = 0
            # count_incorrect = 0
            # with open(incorrect_path, "w", encoding="utf-8") as f:
            #     for i in range(len(formatted_predictions)):
            #         pred_text = formatted_predictions[i]['prediction_text']
            #         ref_texts = references[i]['answers']['text']

            #         # # split prediction by whitespace
            #         # pred_tokens = pred_text.split()
            #         # # loop over reference answers and see if one of the pred_tokens matches
            #         # match_found = any(token in ref_texts for token in pred_tokens)

            #         if pred_text not in ref_texts:
            #         #if not match_found:
            #             f.write(f"Example {i}:\n")
            #             f.write("Context:\n" + eval_examples[i]['context'] + "\n")
            #             f.write("Question:\n" + eval_examples[i]['question'] + "\n")
            #             f.write("Prediction:\n" + pred_text + "\n")
            #             f.write("References:\n" + str(ref_texts) + "\n\n")
            #             count_incorrect += 1
            #         count_total += 1
            #     f.write(f"Incorrect predictions: {count_incorrect}\n")
            #     f.write(f"Total examples: {count_total}\n")
            #     f.write(f"Accuracy: {1 - count_incorrect / count_total:.4f}\n")
            
            # print(f"Incorrect predictions: {count_incorrect}")
            # print(f"Total examples: {count_total}")
            # print(f"Accuracy: {1 - count_incorrect / count_total:.4f}")
                
            # print(f"Incorrect predictions written to {incorrect_path}")
            # print(f"Debug information written to {debug_path}")
            # ###############

            # # --- Per-label accuracy Cartography analysis (easy/hard/ambiguous) ---
            # Expect the cartography analysis CSV to be in tools/cartography_analysis/per_example_labels.csv
            # label_file = os.path.join('tools', 'cartography_analysis', 'per_example_labels.csv')
            # label_map = {}
            # if os.path.exists(label_file):
            #     try:
            #         with open(label_file, encoding='utf-8') as lf:
            #             reader = csv.DictReader(lf)
            #             for r in reader:
            #                 exid = r.get('example_id') or r.get('exampleId') or r.get('id')

            #                 lab = r.get('label') or r.get('Label')
            #                 #lab = r.get('kmeans_label') # USE if testing kmeans labels

            #                 if exid is not None and lab is not None:
            #                     label_map[str(exid)] = lab
            #         print(f"Loaded label map from {label_file} ({len(label_map)} entries)")
            #     except Exception as e:
            #         print(f"Could not read label file {label_file}: {e}")
            # else:
            #     print(f"No per_example_labels.csv at {label_file}; skipping per-label analysis")

            # # Count correct predictions per label
            # total_by_label = defaultdict(int)
            # correct_by_label = defaultdict(int)
            # unknown_label = 'unknown'
            # for i in range(len(formatted_predictions)):
            #     pred = formatted_predictions[i]
            #     exid = str(pred.get('id'))
            #     lab = label_map.get(exid, unknown_label)
            #     total_by_label[lab] += 1
            #     pred_text = pred.get('prediction_text', '')
            #     ref_texts = references[i]['answers']['text']
            #     if pred_text in ref_texts:
            #         correct_by_label[lab] += 1

            # # Add per-label metrics
            # # for lab, tot in total_by_label.items():
            # #     corr = correct_by_label.get(lab, 0)
            # #     acc = corr / tot if tot > 0 else 0.0
            # #     metrics[f'label_{lab}_count'] = tot
            # #     metrics[f'label_{lab}_correct'] = corr
            # #     metrics[f'label_{lab}_accuracy'] = acc

            # # Print a short summary
            # print("Per-label accuracy summary:")
            # for lab in sorted(total_by_label.keys()):
            #     corr = correct_by_label.get(lab, 0)
            #     tot = total_by_label[lab]
            #     print(f"  {lab}: {corr}/{tot} = {corr/tot:.4f}")
            ################################

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state,
                                                         self.control, metrics)
        return metrics