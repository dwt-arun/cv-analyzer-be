import os
from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf_content(file_path: str) -> str:
    """Loads text content from a PDF file using PyMuPDFLoader."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    try:
        loader = PyMuPDFLoader(file_path)
        data = loader.load()
        content = "\n\n".join(doc.page_content for doc in data)
        return content.strip()
    except Exception as e:
        raise RuntimeError(f"Could not load or parse PDF file: {file_path}. Error: {e}")

