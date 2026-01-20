import argparse
import json

def main():
    parser = argparse.ArgumentParser(
        description="Convert Quoref dataset to SQuAD-style format."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input Quoref JSON file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output SQuAD-style JSON file.",
    )
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as infile:
        quoref_data = json.load(infile)

    # squad_style_data = {"data": []}
    squad_style_data = []

    id = 0
    for article in quoref_data["data"]:
        # squad_article = {
        #     "title": article.get("title", ""),
        #     "paragraphs": [],
        # }
        for paragraph in article["paragraphs"]: # a paragraphs contains multiple qas lists & context pairs
            # squad_paragraph = {
            #     "context": paragraph["context"],
            #     "qas": [],
            # }
            for qa in paragraph["qas"]:
                squad_qa = {
                    #"id": qa["id"],
                    "id": str(id),                          # must make sure ids for examples are unique and are strings
                    "title": article['title'],
                    "context": paragraph["context"],
                    "question": qa["question"],
                }
                text_answers = []
                answer_starts = []
                for answer in qa["answers"]:
                    text_answers.append(answer["text"])
                    answer_starts.append(answer["answer_start"])
                    # squad_answer = {
                    #     "text": answer["text"],
                    #     "answer_start": answer["answer_start"],
                    # }
                    # squad_qa["answers"].append(squad_answer)
                squad_qa["answers"] = {
                    "text": text_answers,
                    "answer_start": answer_starts,
                }
                squad_style_data.append(squad_qa)
                id += 1
        #         squad_paragraph["qas"].append(squad_qa)
        #     squad_article["paragraphs"].append(squad_paragraph)
        # squad_style_data["data"].append(squad_article)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        json.dump(squad_style_data, outfile, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()