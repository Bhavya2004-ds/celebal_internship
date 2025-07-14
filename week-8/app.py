import streamlit as st
from rag_engine import generate_answer

st.set_page_config(page_title="Loan RAG Chatbot")
st.title("Loan Q&A Chatbot")

st.markdown("Ask anything about the loan dataset:")

query = st.text_input("Enter your question")

if query:
    with st.spinner("Searching..."):
        answer = generate_answer(query)
    st.markdown("### Answer:")
    st.markdown(answer)
