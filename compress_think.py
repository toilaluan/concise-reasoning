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


def compress_batch(
    texts: list[str],
    rate: float = 0.33,
    force_tokens: list[str] = ["\n", "?"],
    batch_size: int = 16,
) -> list[str]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_results = LLM_LINGUA.compress_prompt(
            batch, rate=rate, force_tokens=force_tokens
        )["compressed_prompt_list"]
        results.extend(batch_results)
    return results


if __name__ == "__main__":
    with open("preprocessed_items.json", "r") as f:
        items = json.load(f)

    # Extract all thinking texts to compress in batches
    thinking_texts = [item["thinking"] for item in items]
    bs = 64
    baches = [thinking_texts[i : i + bs] for i in range(0, len(thinking_texts), bs)]
    # Compress thinking in batches
    compressed_thinking = []
    for batch in tqdm.tqdm(baches):
        compressed_thinking.extend(compress_batch(batch))

    # Update the items with compressed thinking
    for i, item in enumerate(items):
        item["thinking"] = compressed_thinking[i]

    with open("compressed_items.json", "w") as f:
        json.dump(items, f, indent=4)
