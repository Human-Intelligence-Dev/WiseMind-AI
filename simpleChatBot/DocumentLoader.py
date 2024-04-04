from typing import Any
from langchain.document_loaders import (
    PyPDFLoader, TextLoader,
    UnstructuredWordDocumentLoader, UnstructuredEPubLoader
)
import logging
import pathlib
from langchain.schema import Document

class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], ** kwargs: Any):
        super().__init__(file_path, **kwargs, mode = "elements", strategy="fast")

class DocumentLoaderException(Exception):
    pass

class DocumentLoader(object):
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader
    }

def load_document(temp_filepath: str) -> list[Document]:
    ext = pathlib.Path(temp_filepath).suffix
    loader = DocumentLoader.supported_extensions.get(ext)
    if not loader:
        raise DocumentLoaderException(
            f"Invalid extension type {ext}, cannot load this type of file"
        )

    loader = loader(temp_filepath)
    docs = loader.load()
    logging.info(docs)

    return docs