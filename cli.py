import argparse
import os
from pypdf import PdfReader
from src.summarizer import LongDocumentSummarizer


def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="CLI Long-Document Summarizer using DistilBART")
    parser.add_argument("--file", type=str, required=True, help="Path to input .txt, .md, or .pdf file")
    parser.add_argument(
        "--mode",
        type=str,
        default="Balanced",
        choices=["Concise", "Balanced", "Detailed"],
        help="Summary length preset",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional path to save summary text")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Input file not found: {args.file}")

    raw_text = extract_text_from_file(args.file)
    if not raw_text.strip():
        raise ValueError("No readable text extracted from the input file.")

    print(f"Loading model and summarizing '{args.file}' ({len(raw_text.split())} words)...")
    summarizer = LongDocumentSummarizer()
    result = summarizer.summarize(raw_text, length_mode=args.mode)

    print("\n--- FINAL SUMMARY ---")
    print(result["final_summary"])
    print(
        f"\nStats: {result['input_words']} -> {result['summary_words']} words "
        f"({result['compression_ratio']}% reduction) | "
        f"Chunks: {result['num_chunks']} | Time: {result['elapsed_seconds']}s"
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            out.write(result["final_summary"] + "\n")
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()