import json
"""
"""

relevant_fields = [
    "claim_text",
    "surrounding_context",
    "predicted_classification",
    "reasoning",
    "evidences",
]
def transform_file(file_path: str, output_dir: str):
    with open(file_path, 'r',encoding='utf-8') as file:
        data = json.load(file)
    new_data = []
    for item in data["results"]:

        #EVIDENCES
        evidences = []
        for e in item["extractive_chunks"]:
            evidences.append(e["extractive_text"])
        item["evidences"] = evidences

        # Take every relevant fields from item to put in new item llm_output_data
        llm_output_data = {field: item.get(field) for field in relevant_fields}
        new_item = {
            "llm_output_data": llm_output_data
        }
        new_data.append(new_item)
    with open(output_dir, 'w',encoding='utf-8') as file:
        json.dump(new_data, file, indent=2)
    return

if __name__ == "__main__":
    transform_file("Qwen2.5-7B-Instruct-Turbo_20260728_142309.json","Qwen2.5-7B-Instruct-Turbo_20260728_142309_output.json")