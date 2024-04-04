from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever
import os
import tempfile
from DocumentLoader import load_document
from VectorStorage import configure_retriever


def configure_chain(retriever: BaseRetriever) -> Chain:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo', temperature=0, streaming=True
    )
    return ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory,
                                                 verbose=True, max_tokens_limit=4000)


def configure_qa_chain(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))

    retriever = configure_retriever(docs, False)
    return configure_chain(retriever)
