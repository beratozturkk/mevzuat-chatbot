# chatbot_api.py (TAM YENİ VERSİYON - LLM + Memory)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import os

app = FastAPI(title="Mevzuat Chatbot API")

# ChromaDB
CHROMA_DIR = "./mevzuat_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists(CHROMA_DIR):
    raise Exception("❌ ChromaDB bulunamadı!")

db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# LLM (OpenAI - API key gerekli)
# Alternatif: Anthropic Claude, Google Gemini, Ollama (local)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,  # Deterministik cevaplar
    openai_api_key=os.getenv("OPENAI_API_KEY")  # Ortam değişkeninden al
)

# Session bazlı memory storage
chat_sessions = {}


class Question(BaseModel):
    question: str
    session_id: str = "default"  # Her kullanıcı için farklı session
    top_k: int = 3


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list
    session_id: str


def get_or_create_memory(session_id: str):
    """Session için memory oluştur veya getir"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return chat_sessions[session_id]


@app.get("/")
def read_root():
    return {"message": "Mevzuat Chatbot API Çalışıyor! ✅"}


@app.post("/chat", response_model=ChatResponse)
def chat(q: Question):
    """Konuşmalı soru-cevap"""

    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz!")

    # Memory'yi al
    memory = get_or_create_memory(q.session_id)

    # RAG Chain oluştur
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": q.top_k}),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )

    # Soruyu sor
    result = qa_chain({"question": q.question})

    # Kaynakları formatla
    sources = []
    for doc in result["source_documents"]:
        fikra_text = f", Fıkra {doc.metadata.get('fikra_no')}" if doc.metadata.get('fikra_no') else ""
        sources.append({
            "belge": doc.metadata.get("belge", "Bilinmiyor"),
            "madde_no": doc.metadata.get("madde_no", "?"),
            "fikra_no": doc.metadata.get("fikra_no"),
            "kaynak_metni": f"{doc.metadata.get('belge')} - Madde {doc.metadata.get('madde_no')}{fikra_text}",
            "icerik": doc.page_content[:300] + "..."  # İlk 300 karakter
        })

    return ChatResponse(
        question=q.question,
        answer=result["answer"],
        sources=sources,
        session_id=q.session_id
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Session'ı temizle (yeni konuşma başlat)"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"Session '{session_id}' temizlendi"}
    return {"message": "Session bulunamadı"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
