from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from typing import List, Dict, Any
import os
import traceback


def build_vector_store(pdf_dir: str = "data", persist_dir: str = "./chroma_db") -> Chroma:
    """Build a Chroma vectorstore from PDFs in `pdf_dir` and persist to `persist_dir`."""
    print("Loading PDFs...")
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create Embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Save to ChromaDB (local folder)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir)
    try:
        vectorstore.persist()
    except Exception:
        pass
    return vectorstore


def _call_llm(llm: Ollama, prompt: str) -> str:
    try:
        if hasattr(llm, "invoke"):
            return llm.invoke(prompt)
        if hasattr(llm, "generate"):
            gen = llm.generate([prompt])
            try:
                return gen.generations[0][0].text
            except Exception:
                return str(gen)
        # fallback
        out = llm(prompt)
        return out if isinstance(out, str) else str(out)
    except Exception as e:
        traceback.print_exc()
        return f"[LLM error] {e}"


def get_answer(question: str, persist_dir: str = "./chroma_db", top_k: int = 3) -> Dict[str, Any]:
    """Retrieve top-k documents with Chroma and ask Ollama (gemma2:2b).

    Returns: dict with keys `answer` (str) and `source_documents` (list of docs)
    """
    persist_path = persist_dir
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_path, embedding_function=embeddings)

    # Use similarity_search if available, otherwise use retriever interface
    docs: List[Any] = []
    try:
        if hasattr(vectorstore, "similarity_search"):
            docs = vectorstore.similarity_search(question, k=top_k)
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
            if hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(question)
            elif hasattr(retriever, "retrieve"):
                docs = retriever.retrieve(question)
            else:
                # last resort: call retriever
                docs = retriever(question)
    except Exception:
        traceback.print_exc()
        docs = []

    # Build context
    context_parts = []
    for d in docs:
        try:
            text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        except Exception:
            text = str(d)
        context_parts.append(text)
    context = "\n\n".join(context_parts)

    prompt = f"""You are a professional medical knowledge assistant. Answer the question clearly and comprehensively based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

    # Initialize Ollama (use gemma2:2b per requirement)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = Ollama(model="gemma2:2b", base_url=base_url)
    answer = _call_llm(llm, prompt)

    return {"answer": answer, "source_documents": docs}


if __name__ == "__main__":
    print("Use build_vector_store() to construct the DB, then get_answer(question).")