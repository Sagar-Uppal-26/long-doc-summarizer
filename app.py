import streamlit as st
from pypdf import PdfReader
from src.summarizer import LongDocumentSummarizer

st.set_page_config(page_title="Long-Document Summarizer", layout="wide")


@st.cache_resource
def load_summarizer():
    return LongDocumentSummarizer()


summarizer = load_summarizer()

st.title("Long-Document Summarization Tool")
st.caption(
    "Hierarchical Map-Reduce Summarization powered by DistilBART (`sshleifer/distilbart-cnn-6-6`) "
    "with token-aware sentence chunking."
)

with st.sidebar:
    st.header("Pipeline Settings")
    length_mode = st.selectbox("Summary Length", ["Balanced", "Concise", "Detailed"], index=0)
    max_chunk_tokens = st.slider(
        "Max Tokens per Chunk",
        min_value=450,
        max_value=850,
        value=750,
        step=50,
        help="Controls how large each text segment is before running the map summarization pass.",
    )
    st.markdown("---")
    st.markdown(
        "**Architecture:**\n"
        "1. **Chunking:** Splits text at sentence boundaries with 2-sentence overlap.\n"
        "2. **Map Pass:** Summarizes each chunk independently.\n"
        "3. **Reduce Pass:** Consolidates intermediate summaries into a unified brief."
    )

tab_upload, tab_paste = st.tabs(["Upload Document (.pdf / .txt)", "Paste Long Text"])

document_text = ""
source_name = "pasted_text"

with tab_upload:
    uploaded_file = st.file_uploader("Choose a PDF, TXT, or Markdown file", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        source_name = uploaded_file.name.rsplit(".", 1)[0]
        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            document_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            document_text = uploaded_file.read().decode("utf-8", errors="ignore")

        if document_text.strip():
            st.success(f"Loaded **{uploaded_file.name}** ({len(document_text.split()):,} words)")
        else:
            st.error("Could not extract readable text from this PDF (it may be a scanned image).")

with tab_paste:
    pasted = st.text_area(
        "Paste an article, research paper, or report below",
        height=240,
        placeholder="Paste multi-paragraph text here...",
    )
    if pasted.strip():
        document_text = pasted

if st.button("Generate Summary", type="primary", disabled=not bool(document_text.strip())):
    with st.spinner("Running chunking and hierarchical summarization..."):
        result = summarizer.summarize(
            document_text,
            length_mode=length_mode,
            max_chunk_tokens=max_chunk_tokens,
        )

    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Original Words", f"{result['input_words']:,}")
    m2.metric("Summary Words", f"{result['summary_words']:,}")
    m3.metric("Compression", f"{result['compression_ratio']}%")
    m4.metric("Chunks Processed", result["num_chunks"])
    m5.metric("Inference Time", f"{result['elapsed_seconds']}s")

    st.subheader("Consolidated Summary")
    st.write(result["final_summary"])

    st.download_button(
        label="Download Summary (.txt)",
        data=result["final_summary"],
        file_name=f"{source_name}_summary.txt",
        mime="text/plain",
    )

    if result["num_chunks"] > 1:
        with st.expander(f"View Intermediate Chunk Summaries ({result['num_chunks']} chunks)"):
            for idx, chunk_sum in enumerate(result["chunk_summaries"], start=1):
                st.markdown(f"**Chunk {idx}:** {chunk_sum}")