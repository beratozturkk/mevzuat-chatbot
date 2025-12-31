# chatbot_api.py (GÜNCELLENMİŞ VERSİYON)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

app = FastAPI(title="Mevzuat Chatbot API")

CHROMA_DIR = "./mevzuat_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists(CHROMA_DIR):
    raise Exception("❌ ChromaDB bulunamadı! Önce build_chromadb.py çalıştırın.")

db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


class Question(BaseModel):
    question: str
    top_k: int = 3


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list


@app.get("/")
def read_root():
    return {"message": "Mevzuat Chatbot API Çalışıyor! ✅"}


@app.post("/chat", response_model=ChatResponse)
def chat(q: Question):
    """Kullanıcı sorusuna cevap ver"""

    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz!")

    # İlgili maddeleri getir - similarity_search_with_score ile skor da al
    results_with_scores = db.similarity_search_with_score(q.question, k=q.top_k)

    if not results_with_scores:
        return ChatResponse(
            question=q.question,
            answer="Üzgünüm, bu konuda mevzuatlarda bir bilgi bulamadım.",
            sources=[]
        )

    # Kaynakları hazırla
    sources = []
    context = ""

    for i, (doc, score) in enumerate(results_with_scores, 1):
        # Tam içeriği göster (200 karakter sınırı YOK artık)
        context += f"\n[Kaynak {i}]\n{doc.page_content}\n"

        sources.append({
            "kaynak_no": i,
            "belge": doc.metadata.get("belge", "Bilinmiyor"),
            "madde_no": doc.metadata.get("madde_no", "?"),
            "icerik": doc.page_content,  # TAM İÇERİK
            "relevance_score": round(float(score), 2)  # Alakalılık skoru
        })

    # Cevap üret
    answer = generate_answer(q.question, context, sources)

    return ChatResponse(
        question=q.question,
        answer=answer,
        sources=sources
    )


def generate_answer(question: str, context: str, sources: list) -> str:
    """Daha akıllı cevap üret"""

    answer = f"📚 **'{question}'** sorunuz hakkında {len(sources)} mevzuat maddesi bulundu:\n\n"

    for source in sources:
        answer += f"### 📄 {source['belge']} - Madde {source['madde_no']}\n"
        answer += f"*Alakalılık: %{(1 - source['relevance_score']) * 100:.0f}*\n\n"
        answer += f"{source['icerik']}\n\n"
        answer += "---\n\n"

    return answer


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
