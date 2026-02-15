import streamlit as st

st.set_page_config(page_title="AI Multi-Level Learning Assistant")

st.title("AI Multi-Level Learning Assistant")

st.write(
    "Ask questions from your study material and get explanations at different difficulty levels."
)

# Placeholder UI (backend integration later)
question = st.text_input("Enter your question")

level = st.selectbox(
    "Choose explanation level",
    ["ELI5", "College", "Interview"]
)

if st.button("Get Answer"):
    st.info("Backend integration coming soon 🚧")
