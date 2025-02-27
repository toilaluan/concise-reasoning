import json
from datasets import Dataset

with open("compressed_items.json", "r") as f:
    items = json.load(f)

dataset = Dataset.from_list(items)

dataset.map(
    lambda x: {
        "messages": [
            {"role": "user", "content": x["user_message"]},
            {
                "role": "assistant",
                "content": f"<think>{x['thinking']}</think>\n{x['answer']}",
            },
        ]
    },
    num_proc=10,
)

dataset.push_to_hub("toilaluan/compressed-thinking")
