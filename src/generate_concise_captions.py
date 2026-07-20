"""
Generate concise captions for CLIP training using an open-source LLM.

This script loads data.csv and generates a second caption field by querying an LLM
to compress the original descriptions while maintaining medical accuracy.
CLIP models typically use a max token length of 77 tokens.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """You are a medical caption compression expert. Your task is to rewrite long medical image descriptions into concise captions suitable for CLIP model training.

Requirements:
1. Keep all medically relevant information (condition, body location, patient demographics)
2. Remove verbose phrasing and unnecessary words
3. Maintain medical accuracy and terminology
4. Target maximum length: 77 tokens (typically 60-70 words)
5. Use clear, direct language

Example:
Input: "This total body photograph tile shows a close up benign skin lesion skin condition on the Right Leg (lower extremity) for a male patient of age 60."
Output: "Benign skin lesion on right leg, male, 60 years old."
"""

USER_PROMPT_TEMPLATE = """Compress this medical image caption while maintaining all medically relevant information:

{original_caption}

Provide only the compressed caption, no explanation."""


def generate_concise_caption(description: str, model, tokenizer, device: str) -> str:
    """
    Generate a concise caption using a local LLM.

    Args:
        description: Original long-form description
        model: HuggingFace model instance
        tokenizer: HuggingFace tokenizer instance
        device: Device to run inference on

    Returns:
        Concise caption string
    """
    try:
        # Format prompt based on model type
        if (
            "Llama" in tokenizer.__class__.__name__
            or "Mistral" in tokenizer.__class__.__name__
        ):
            # Chat template for Llama/Mistral
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        original_caption=description
                    ),
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Generic format
            prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(original_caption=description)}"

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part
        if "assistant" in generated_text.lower():
            # For chat models, extract after assistant marker
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = generated_text
        else:
            # For generic models, extract after the prompt
            response = generated_text[len(prompt) :].strip()

        # Clean up response
        response = response.split("\n")[0].strip()  # Take first line only
        return response if response else description

    except Exception as e:
        print(f"Error generating caption: {e}")
        return description  # Fallback to original


def main():
    parser = argparse.ArgumentParser(
        description="Generate concise captions for CLIP training"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="assets/data.csv",
        help="Path to input CSV file (default: assets/data.csv)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="assets/data_with_concise_captions.csv",
        help="Path to output CSV file (default: assets/data_with_concise_captions.csv)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Save progress every N rows (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if it exists",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: auto-detect)",
    )

    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
        use_cache=True,
    )

    if args.device == "cpu":
        model = model.to(args.device)

    model.eval()
    print(f"Model loaded on {args.device}")

    # Load input data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows")

    # Check if resuming
    start_idx = 0
    if args.resume and Path(args.output_csv).exists():
        print(f"Resuming from {args.output_csv}...")
        df_existing = pd.read_csv(args.output_csv)
        start_idx = len(df_existing)
        print(f"Starting from row {start_idx}")

        # Use existing data for already processed rows
        df.loc[: start_idx - 1, "concise_description"] = df_existing[
            "concise_description"
        ]
    else:
        # Initialize the new column
        df["concise_description"] = None

    # Generate concise captions
    print(f"Generating concise captions using {args.model_name}...")
    for idx in tqdm(range(start_idx, len(df)), desc="Processing"):
        original_desc = df.loc[idx, "description"]
        concise_desc = generate_concise_caption(
            original_desc, model, tokenizer, args.device
        )
        df.loc[idx, "concise_description"] = concise_desc

        # Save progress periodically
        if (idx + 1) % args.batch_size == 0:
            df.to_csv(args.output_csv, index=False)
            print(f"\nProgress saved at row {idx + 1}")

    # Final save
    df.to_csv(args.output_csv, index=False)
    print(f"\nComplete! Saved to {args.output_csv}")

    # Print statistics
    print("\n--- Statistics ---")
    print(f"Total rows: {len(df)}")
    if "concise_description" in df.columns:
        avg_original_len = df["description"].str.split().str.len().mean()
        avg_concise_len = df["concise_description"].str.split().str.len().mean()
        print(f"Average original caption length: {avg_original_len:.1f} words")
        print(f"Average concise caption length: {avg_concise_len:.1f} words")
        print(
            f"Average compression ratio: {(1 - avg_concise_len/avg_original_len)*100:.1f}%"
        )


if __name__ == "__main__":
    main()
