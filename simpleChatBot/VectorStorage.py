from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema import Document, BaseRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever


def configure_retriever(docs: list[Document], use_compression: bool) -> BaseRetriever:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs = {"k": 2, "fetch_k": 4})

    if not use_compression:
        return retriever

    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.76
    )

    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever
    )

