from datasets import load_dataset
import re
import tqdm

DS = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", num_proc=10)

DS = DS.shuffle(seed=42).select(range(1000))
preprocessed_items = []

for item in tqdm.tqdm(DS):
    messages = item["messages"]
    user_message = messages[0]["content"]
    assistant_message_with_thinking = messages[1]["content"]
    thinking_pattern = r"<think>(.*?)</think>"
    thinking = re.search(thinking_pattern, assistant_message_with_thinking, re.DOTALL)
    answer_pattern = r"</think>(.*)"
    answer = re.search(answer_pattern, assistant_message_with_thinking, re.DOTALL)
    try:
        thinking = thinking.group(1)
        answer = answer.group(1)
    except:
        continue

    preprocessed_items.append(
        {
            "user_message": user_message,
            "thinking": thinking,
            "answer": answer,
        }
    )

print(f"Preprocessed {len(preprocessed_items)} items")

import json

with open("preprocessed_items.json", "w") as f:
    json.dump(preprocessed_items, f, indent=4)
