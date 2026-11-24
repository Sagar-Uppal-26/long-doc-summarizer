import re
import time
from typing import Dict, List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "sshleifer/distilbart-cnn-6-6"

LENGTH_PRESETS = {
    "Concise": {"chunk_min": 30, "chunk_max": 75, "final_min": 60, "final_max": 140},
    "Balanced": {"chunk_min": 45, "chunk_max": 110, "final_min": 100, "final_max": 220},
    "Detailed": {"chunk_min": 65, "chunk_max": 150, "final_min": 150, "final_max": 320},
}


class LongDocumentSummarizer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _split_sentences(self, text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, max_tokens: int = 750, overlap_sentences: int = 2) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))

            # Handle unusually long single sentences safely
            if sent_tokens > max_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                words = sentence.split()
                for i in range(0, len(words), max_tokens // 2):
                    chunks.append(" ".join(words[i : i + max_tokens // 2]))
                continue

            if current_tokens + sent_tokens > max_tokens and current_sentences:
                chunks.append(" ".join(current_sentences))
                overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
                current_sentences = overlap + [sentence]
                current_tokens = sum(
                    len(self.tokenizer.encode(s, add_special_tokens=False)) for s in current_sentences
                )
            else:
                current_sentences.append(sentence)
                current_tokens += sent_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _summarize_block(self, text: str, min_length: int, max_length: int) -> str:
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]
        safe_max = min(max_length, max(32, int(input_len * 0.75)))
        safe_min = min(min_length, max(12, safe_max - 15))

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=2,
                min_length=safe_min,
                max_length=safe_max,
                no_repeat_ngram_size=3,
                length_penalty=1.5,
                early_stopping=True,
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def summarize(
        self,
        text: str,
        length_mode: str = "Balanced",
        max_chunk_tokens: int = 750,
    ) -> Dict:
        start_time = time.perf_counter()
        preset = LENGTH_PRESETS.get(length_mode, LENGTH_PRESETS["Balanced"])
        input_words = len(text.split())

        chunks = self.chunk_text(text, max_tokens=max_chunk_tokens, overlap_sentences=2)
        if not chunks:
            return {
                "final_summary": "",
                "chunk_summaries": [],
                "num_chunks": 0,
                "input_words": 0,
                "summary_words": 0,
                "compression_ratio": 0.0,
                "elapsed_seconds": 0.0,
            }

        # Map phase: summarize each chunk
        chunk_summaries = [
            self._summarize_block(chunk, preset["chunk_min"], preset["chunk_max"])
            for chunk in chunks
        ]

        # Reduce phase: merge and consolidate if multiple chunks exist
        if len(chunk_summaries) == 1:
            final_summary = chunk_summaries[0]
        else:
            combined = " ".join(chunk_summaries)
            combined_tokens = len(self.tokenizer.encode(combined, add_special_tokens=False))

            while combined_tokens > max_chunk_tokens:
                sub_chunks = self.chunk_text(combined, max_tokens=max_chunk_tokens, overlap_sentences=1)
                reduced = [
                    self._summarize_block(sc, preset["chunk_min"], preset["chunk_max"])
                    for sc in sub_chunks
                ]
                combined = " ".join(reduced)
                combined_tokens = len(self.tokenizer.encode(combined, add_special_tokens=False))

            final_summary = self._summarize_block(combined, preset["final_min"], preset["final_max"])

        summary_words = len(final_summary.split())
        compression = round((1 - (summary_words / max(input_words, 1))) * 100, 1)
        elapsed = round(time.perf_counter() - start_time, 2)

        return {
            "final_summary": final_summary,
            "chunk_summaries": chunk_summaries,
            "num_chunks": len(chunks),
            "input_words": input_words,
            "summary_words": summary_words,
            "compression_ratio": max(0.0, compression),
            "elapsed_seconds": elapsed,
        }