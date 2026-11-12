
import streamlit as st
from main import ask_question, build_document_index, ask_document
from document_loader import extract_text

# Page configuration
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="centered"
)

# Application title
st.title("🧠 DocuMind AI")

st.write(
    "Your intelligent company policy assistant "
    "powered by Retrieval-Augmented Generation."
)

st.divider()


# ==========================================
# DOCUMENT UPLOAD SECTION
# ==========================================

st.subheader("Upload your document")

uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    try:
        # Extract document text
        extracted_text = extract_text(uploaded_file)

        # Build the document index only for a new upload
        if st.session_state.document_name != uploaded_file.name:
            with st.spinner("Generating document embeddings..."):
                st.session_state.document_index = build_document_index(
                    extracted_text
                )

                st.session_state.document_name = uploaded_file.name

            st.success("Document indexed successfully!")

        st.write("Characters extracted:", len(extracted_text))

        # Display a preview of the extracted content
        with st.expander("Preview extracted text"):
            st.text(extracted_text[:2000])

    except Exception as e:
        st.error(f"Unable to process document: {e}")

st.divider()


# User input
question = st.text_input(
    "Ask a question about company policies:",
    placeholder="e.g. How many days can I work remotely?"
)

# Ask button
if st.button("Ask DocuMind", type="primary"):

    if question.strip():

        with st.spinner("Searching documents and generating answer..."):

            if st.session_state.document_index is not None:

                answer = ask_document(
                    question,
                    st.session_state.document_index
                )

            else:
                # Fallback to original company policy
                answer = ask_question(question)

        st.subheader("AI Answer")
        st.write(answer)

    else:
        st.warning("Please enter a question.")

st.divider()

st.caption(
    "Built with Python, Sentence Transformers, "
    "OpenRouter, and Streamlit."
)



# ==========================================
# SESSION STATE
# ==========================================

if "document_index" not in st.session_state:
    st.session_state.document_index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None
