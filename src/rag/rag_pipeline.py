import os
from pathlib import Path
from typing import Dict, List, Union
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

PLAYBOOKS_DIR = Path(__file__).resolve().parents[2] / "playbooks"
CHROMA_DB_DIR = Path(__file__).resolve().parents[2] / "chroma_db"


def get_embedding_function():
    """Fast, local ONNX embeddings without PyTorch or heavy dependencies."""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def load_and_chunk_playbooks(playbooks_dir: Union[str, Path] = PLAYBOOKS_DIR) -> List[Document]:
    playbooks_path = Path(playbooks_dir)
    if not playbooks_path.exists():
        raise FileNotFoundError(f"Playbooks directory not found at: {playbooks_path}")

    loader = DirectoryLoader(
        str(playbooks_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    all_chunks: List[Document] = []
    for doc in docs:
        header_splits = markdown_splitter.split_text(doc.page_content)
        source_filename = Path(doc.metadata.get("source", "")).name
        for split in header_splits:
            split.metadata["source"] = source_filename
            all_chunks.append(split)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(all_chunks)


def build_or_load_vectorstore(
    playbooks_dir: Union[str, Path] = PLAYBOOKS_DIR,
    persist_dir: Union[str, Path] = CHROMA_DB_DIR,
    force_rebuild: bool = False,
) -> Chroma:
    embeddings = get_embedding_function()
    persist_path = Path(persist_dir)

    if persist_path.exists() and any(persist_path.iterdir()) and not force_rebuild:
        return Chroma(persist_directory=str(persist_path), embedding_function=embeddings)

    chunks = load_and_chunk_playbooks(playbooks_dir)
    return Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=str(persist_path)
    )


def query_retention_playbooks(
    query: str, k: int = 3, persist_dir: Union[str, Path] = CHROMA_DB_DIR
) -> List[Dict[str, Union[str, Dict]]]:
    vectorstore = build_or_load_vectorstore(persist_dir=persist_dir)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    relevant_docs = retriever.invoke(query)

    return [
        {
            "rank": idx,
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown Playbook"),
        }
        for idx, doc in enumerate(relevant_docs, 1)
    ]


def generate_retention_strategy(customer_profile: dict, churn_probability: float) -> str:
    query = (
        f"Customer profile: Contract={customer_profile.get('Contract')}, "
        f"PaymentMethod={customer_profile.get('PaymentMethod')}, "
        f"MonthlyCharges=${customer_profile.get('MonthlyCharges')}, "
        f"Tenure={customer_profile.get('Tenure')} months. "
        f"How to retain high churn risk customer?"
    )

    retrieved_chunks = query_retention_playbooks(query, k=3)
    context = "\n\n".join([f"--- Source: {c['source']} ---\n{c['content']}" for c in retrieved_chunks])

    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert customer success retention strategist for a telecommunications firm."),
        ("user", """A customer has a high churn probability of {churn_risk}%.

Customer Profile:
{customer_data}

Playbook Context:
{playbook_context}

Provide:
1. Concise 2-sentence risk diagnosis.
2. 2 specific retention strategies tailored to their contract/payment method.
3. Suggest a specific talk track or offer script for the retention agent.""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "churn_risk": round(churn_probability * 100, 1),
        "customer_data": str(customer_profile),
        "playbook_context": context,
    })

    return response.content