import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from DocumentLoader import DocumentLoader
from ChatChain import configure_qa_chain

st.set_page_config(page_title="WiseMind Ai: Chat with documents", page_icon="📖")
st.title("📖 WiseMind Ai: Chat with Documents")

uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    type=list(DocumentLoader.supported_extensions.keys()),
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()

qa_chain = configure_qa_chain(uploaded_files)
assistant = st.chat_message("assistant")
user_query = st.chat_input(placeholder="Ask me Anything!")

if user_query:
    stream_handler = StreamlitCallbackHandler(assistant)
    response = qa_chain.run(user_query, callbacks=[stream_handler])
    st.markdown(response)