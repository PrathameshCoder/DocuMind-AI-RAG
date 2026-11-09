
import streamlit as st
from main import ask_question

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

# User input
question = st.text_input(
    "Ask a question about company policies:",
    placeholder="e.g. How many days can I work remotely?"
)

# Ask button
if st.button("Ask DocuMind", type="primary"):

    if question.strip():

        with st.spinner("Searching documents and generating answer..."):

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
