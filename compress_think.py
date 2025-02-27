import json
import tqdm
from llmlingua import PromptCompressor

LLM_LINGUA = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,  # Whether to use llmlingua-2
)


def compress_text(
    text: str, rate: float = 0.33, force_tokens: list[str] = ["\n", "?"]
) -> str:
    compressed_prompt = LLM_LINGUA.compress_prompt(
        text, rate=rate, force_tokens=force_tokens
    )["compressed_prompt"]
    return compressed_prompt


if __name__ == "__main__":
    with open("preprocessed_items.json", "r") as f:
        items = json.load(f)

    for item in tqdm.tqdm(items):
        item["thinking"] = compress_text(item["thinking"])

    with open("compressed_items.json", "w") as f:
        json.dump(items, f, indent=4)
