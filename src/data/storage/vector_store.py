import os
import argparse
from pathlib import Path
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


def vector_store(pdf_dir: str, chroma_dir: str):
    """
    Function to fetch pdfs from a directory and store them in a local vector store in other directory.
    """

    pdf_path = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_path:
        raise Exception("No pdfs found in the directory")

    documents = []
    for pdf in pdf_path:
        try:
            loader = PyPDFLoader(str(pdf))
            docs = loader.load()
            print(f"{pdf} Loaded successfully")
        except UnicodeEncodeError:
            print(f"Skipping problematic pdf: {pdf}")
            continue

        for doc in docs:
            doc.metadata.update({"title": pdf.stem, "document_type": "research_paper"})

        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    splitted_docs = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3.1:8b-instruct-q4_K_M")

    vector_store = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    vector_store.add_documents(splitted_docs)
    vector_store.persist()

    # Chroma.from_documents(
    #     documents=splitted_docs, embedding=embeddings, persist_directory=chroma_dir
    # )

    print(
        f"Successfully ingested {len(documents)} documents into Chroma VS at {chroma_dir}"
    )

    # ===================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into Chroma vector store, please provide the path to the directory conataining the PDFs and the path to save it into."
    )

    parser.add_argument(
        "pdf_dir", type=str, help="path to the directory containing the PDFs"
    )
    parser.add_argument("chroma_dir", type=str, help="path to save the PDFs")

    args = parser.parse_args()

    vector_store(args.pdf_dir, args.chroma_dir)
