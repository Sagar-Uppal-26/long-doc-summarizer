# LLM-Based Long-Document Summarization Tool

A long-document summarization system built with Hugging Face Transformers (`sshleifer/distilbart-cnn-6-6`), PyTorch, and Streamlit. Standard sequence-to-sequence models truncate inputs beyond 1,024 tokens; this tool overcomes that limit using sentence-aware overlapping chunking and a recursive map-reduce aggregation pipeline.

**Live Demo:** [Open Streamlit App](https://your-summarizer-app.streamlit.app)

## How It Works

1. **Sentence-Aware Chunking (`src/summarizer.py`):** Splits raw text at sentence boundaries while tracking exact subword token counts (`max_tokens=750`). Adjacent chunks share a 2-sentence overlap so context across boundaries is preserved.
2. **Map Summarization Pass:** Each chunk is summarized independently using beam search (`num_beams=2`) with n-gram repetition penalties (`no_repeat_ngram_size=3`).
3. **Hierarchical Reduce Pass:** Intermediate chunk summaries are merged and passed through a final consolidation pass (recursively reduced if the combined text still exceeds the model's context window).

## Project Structure

```text
long-doc-summarizer/
├── src/
│   ├── __init__.py
│   └── summarizer.py     # Token-aware chunker & DistilBART map-reduce engine
├── cli.py                # Command-line tool for local file summarization
├── app.py                # Streamlit web UI with PDF/TXT upload and chunk trace
├── requirements.txt
└── README.md
```

## Running Locally

1. **Clone the repository and create a virtual environment:**
   ```bash
   git clone [https://github.com/Sagar-Uppal-26/long-doc-summarizer.git](https://github.com/Sagar-Uppal-26/long-doc-summarizer.git)
   cd long-doc-summarizer
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Run via Command Line (CLI):**
   ```bash
   python cli.py --file path/to/document.pdf --mode Balanced --output summary.txt
   ```

3. **Run the Streamlit Web Interface:**
   ```bash
   streamlit run app.py
   ```