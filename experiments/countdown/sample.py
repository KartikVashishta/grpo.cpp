"""Sample one Countdown answer from a downloaded checkpoint."""

import argparse
from pathlib import Path

from task import messages, verify


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--numbers", type=int, nargs=3, default=[26, 79, 99])
    parser.add_argument("--target", type=int, default=46)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True).to(device).eval()
    prompt = messages({"nums": args.numbers, "target": args.target})
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=True,
                                temperature=1.0, top_p=1.0, top_k=0,
                                pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    result = verify(answer, args.numbers, args.target)
    print(answer)
    print(f"{result.status}: {result.value}")


if __name__ == "__main__":
    main()
