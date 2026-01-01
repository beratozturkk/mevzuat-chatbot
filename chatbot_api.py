# chatbot_api.py - FİX: Gemini Model + Gelişmiş Retrieval

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict
import os
import re
import json
from rank_bm25 import BM25Okapi
from google import genai
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# KONFIGÜRASYON
# ============================================================================

app = FastAPI(title="Mevzuat Chatbot API")

CHROMA_DIR = "./mevzuat_db"
JSON_PATH = "tum_mevzuat_maddeleri.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ============================================================================
# MODEL YÜKLEMELERİ
# ============================================================================

print("🤖 BERTurk embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="dbmdz/bert-base-turkish-cased",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ BERTurk hazır!")

if not os.path.exists(CHROMA_DIR):
    raise Exception(f"❌ ChromaDB bulunamadı: {CHROMA_DIR}")

print("📚 ChromaDB yükleniyor...")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
print("✅ ChromaDB hazır!")

if not os.path.exists(JSON_PATH):
    raise Exception(f"❌ JSON dosyası bulunamadı: {JSON_PATH}")

print("📄 Mevzuat dökümanları yükleniyor...")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    all_documents = json.load(f)
print(f"✅ {len(all_documents)} madde yüklendi")

print("📊 BM25 index oluşturuluyor...")
tokenized_corpus = [doc["icerik"].lower().split() for doc in all_documents]
bm25 = BM25Okapi(tokenized_corpus)
print("✅ BM25 hazır!")

print("🧠 Gemini LLM yapılandırılıyor...")
try:
    # Yeni SDK
    client = genai.Client(api_key=GEMINI_API_KEY)
    MODEL_NAME = 'gemini-2.5-flash'
    print("✅ Gemini LLM hazır!")
    gemini_available = True
except Exception as e:
    print(f"❌ Gemini yapılandırılamadı: {e}")
    gemini_available = False


# ============================================================================
# PYDANTIC MODELLERİ
# ============================================================================

class Question(BaseModel):
    question: str
    session_id: str = "default"
    top_k: int = 5
    temperature: float = 0.3

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list
    session_id: str

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def normalize_text(text: str) -> str:
    """Metni normalize et"""
    text = text.lower()
    text = re.sub(r'[^\wğüşöçıİ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords(query: str) -> List[str]:
    """
    Sorudan önemli anahtar kelimeleri çıkar (stop words'leri filtrele)
    """
    # Türkçe stop words
    stop_words = {
        'bir', 'bu', 've', 'ile', 'için', 'mi', 'mı', 'mu', 'mü',
        'da', 'de', 'ta', 'te', 'ben', 'sen', 'bana', 'sana',
        'ne', 'nedir', 'nasıl', 'gibi', 'olan', 'olarak',
        'verir', 'misin', 'misiniz', 'var', 'yok', 'şey',
        'hakkında', 'konusunda', 'ile', 'alakalı', 'ilgili',
        'bilgi', 'verme', 'vermek', 'söyle', 'anlat',
        'eder', 'olur', 'dır', 'dir', 'tir', 'tır'
    }

    words = normalize_text(query).split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    return keywords

def hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Geliştirilmiş Hybrid Search (BM25 + Semantic + Keyword Filtering)
    """
    query_normalized = normalize_text(query)

    # Anahtar kelimeleri çıkar
    keywords = extract_keywords(query)
    print(f"🔍 Anahtar kelimeler: {keywords}")

    # 1. BM25 Search (sadece keywords ile)
    bm25_scores = bm25.get_scores(keywords)

    # 2. Semantic Search
    try:
        semantic_results = db.similarity_search_with_score(query, k=top_k * 3)
    except Exception as e:
        print(f"⚠️  Semantic search hatası: {e}")
        semantic_results = []

    # 3. Sonuçları birleştir
    candidates = {}

    # BM25 sonuçları
    for idx, score in enumerate(bm25_scores):
        if score > 0:
            doc_id = f"doc_{idx}"
            candidates[doc_id] = {
                'doc': all_documents[idx],
                'bm25_score': float(score),
                'semantic_score': 0.0,
                'content': all_documents[idx]['icerik']
            }

    # Semantic sonuçları
    for doc, distance in semantic_results:
        content = doc.page_content
        for idx, d in enumerate(all_documents):
            if d['icerik'] == content:
                doc_id = f"doc_{idx}"
                semantic_score = 1 - distance

                if doc_id in candidates:
                    candidates[doc_id]['semantic_score'] = semantic_score
                else:
                    candidates[doc_id] = {
                        'doc': d,
                        'bm25_score': 0.0,
                        'semantic_score': semantic_score,
                        'content': content
                    }
                break

    # 4. AKILLI KEYWORD BOOST (İyileştirilmiş)
    domain_keywords = {
        'güz': ['güz', 'bahar', 'dönem', 'yarıyıl', 'akademik takvim'],
        'bahar': ['güz', 'bahar', 'dönem', 'yarıyıl', 'akademik takvim'],
        'başvuru': ['başvuru', 'müracaat', 'kayıt', 'kabul', 'şart', 'koşul'],
        'yatay': ['yatay geçiş', 'dikey geçiş', 'transfer', 'intibak'],
        'çift': ['çift anadal', 'çap', 'yan dal', 'yandal'],
        'yan': ['yan dal', 'yandal', 'çift anadal'],
        'staj': ['staj', 'uygulama', 'iş yeri', 'işletme'],
        'sınav': ['sınav', 'final', 'vize', 'bütünleme', 'mazeret'],
        'dersi': ['ders', 'kurs', 'program', 'müfredat'],
    }

    for doc_id, result in candidates.items():
        content_lower = result['content'].lower()
        belge_lower = result['doc']['belge'].lower()

        keyword_boost = 0
        match_count = 0

        # Her keyword için kontrol
        for kw in keywords:
            # Doğrudan eşleşme
            if kw in content_lower:
                match_count += 1
                keyword_boost += 2.0

            # Belge adında eşleşme (daha değerli)
            if kw in belge_lower:
                keyword_boost += 3.0
                match_count += 1

            # Domain keyword grubu eşleşmesi
            if kw in domain_keywords:
                related_words = domain_keywords[kw]
                for rel_word in related_words:
                    if rel_word in content_lower or rel_word in belge_lower:
                        keyword_boost += 1.5
                        match_count += 0.5
                        break

        # Eşleşme oranı
        if keywords:
            match_ratio = match_count / len(keywords)
            keyword_boost *= match_ratio  # Düşük eşleşmeleri cezalandır

        result['keyword_boost'] = keyword_boost
        result['match_count'] = match_count

    # 5. Final Skor (Hybrid + Match Filtering)
    for doc_id in candidates:
        c = candidates[doc_id]

        # En az 1 keyword eşleşmesi olmalı
        if c['match_count'] < 1:
            c['final_score'] = 0.0
        else:
            c['final_score'] = (
                c['bm25_score'] * 0.35 +
                c['semantic_score'] * 0.35 +
                c['keyword_boost'] * 0.30
            )

    # 6. Sırala ve filtrele (0 skorları çıkar)
    sorted_results = sorted(
        [c for c in candidates.values() if c['final_score'] > 0],
        key=lambda x: x['final_score'],
        reverse=True
    )

    return sorted_results[:top_k]

def create_llm_answer(results: List[Dict], query: str, temperature: float = 0.3) -> str:
    """
    Gemini LLM ile akıllı cevap üret
    """
    if not results:
        return "Üzgünüm, bu konuda mevzuatlarda ilgili bilgi bulamadım. Lütfen sorunuzu farklı kelimelerle ifade etmeyi deneyin."

    if not gemini_available:
        return create_fallback_answer(results, query)

    # Context hazırla
    context = ""
    for i, result in enumerate(results[:3], 1):
        doc = result['doc']
        context += f"\n\n--- KAYNAK {i} ---\n"
        context += f"Belge: {doc['belge']}\n"
        context += f"Madde No: {doc['madde_no']}\n"
        if doc.get('fikra_no'):
            context += f"Fıkra No: {doc['fikra_no']}\n"
        context += f"İçerik:\n{doc['icerik']}\n"

    # Prompt
    prompt = f"""Sen Selçuk Üniversitesi'nin mevzuat konusunda uzman bir asistansın.

GÖREV: Aşağıdaki mevzuat maddelerini kullanarak kullanıcının sorusunu cevapla.

KULLANICI SORUSU:
{query}

İLGİLİ MEVZUAT MADDELERİ:
{context}

ÖNEMLİ KURALLAR:
1. **SADECE** verilen mevzuat maddelerindeki bilgileri kullan
2. Eğer soruyla tam alakalı bilgi yoksa, "Verilen mevzuat maddelerinde bu konuda açık bir bilgi bulamadım" de
3. Cevabını Türkçe, net, anlaşılır ve yapılandırılmış şekilde ver
4. Madde ve fıkra numaralarını belirt
5. Gereksiz tekrar yapma, direkt cevapla
6. Kaynakları göstermeyi unutma

CEVAP:"""

    try:
        # YENİ SDK ile çağrı
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                'temperature': temperature,
                'max_output_tokens': 4096,
            }
        )

        answer = response.text

        # Kaynakları ekle
        answer += "\n\n---\n\n### 📚 Kaynaklar\n\n"
        for i, result in enumerate(results[:3], 1):
            doc = result['doc']
            kaynak = f"**{i}.** {doc['belge']} - Madde {doc['madde_no']}"
            if doc.get('fikra_no'):
                kaynak += f", Fıkra {doc['fikra_no']}"
            answer += kaynak + "\n"

        return answer

    except Exception as e:
        print(f"❌ LLM hatası: {e}")
        return create_fallback_answer(results, query)

def create_fallback_answer(results: List[Dict], query: str) -> str:
    """Fallback: LLM olmadan cevap"""
    if not results:
        return "Üzgünüm, bu konuda bilgi bulamadım."

    answer = f"### İlgili Mevzuat Maddeleri\n\n"

    for i, result in enumerate(results[:3], 1):
        doc = result['doc']
        content = doc['icerik'][:500].strip()
        if len(doc['icerik']) > 500:
            content += "..."

        answer += f"#### {i}. {doc['belge']} - Madde {doc['madde_no']}\n\n"
        answer += f"{content}\n\n"
        answer += "---\n\n"

    answer += "⚠️  **Not:** LLM aktif değil. Gemini API key ekleyin.\n"

    return answer

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "Selçuk Üniversitesi Mevzuat Chatbot API",
        "status": "online",
        "toplam_madde": len(all_documents),
        "llm_aktif": gemini_available,
        "model": "BERTurk + Gemini 2.5 Flash"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz!")

    print(f"\n{'=' * 70}")
    print(f"🔍 Soru: {q.question}")
    print(f"🌡️  Temperature: {q.temperature}")

    # Gelişmiş Hybrid Search
    results = hybrid_search(q.question, q.top_k)

    if not results:
        print("❌ Hiç alakalı sonuç bulunamadı")
        return ChatResponse(
            question=q.question,
            answer="Üzgünüm, bu konuda mevzuatlarda ilgili bilgi bulamadım. Lütfen sorunuzu farklı kelimelerle ifade etmeyi deneyin veya daha spesifik bir soru sorun.",
            sources=[],
            session_id=q.session_id
        )

    print(f"📚 {len(results)} alakalı sonuç bulundu:")
    for i, r in enumerate(results[:5], 1):
        doc = r['doc']
        print(f"   {i}. {doc['belge'][:60]}... - Madde {doc['madde_no']} (skor: {r['final_score']:.2f}, eşleşme: {r['match_count']:.1f})")

    # Kaynakları hazırla
    sources = []
    for r in results:
        doc = r['doc']
        kaynak_metni = f"{doc['belge']} - Madde {doc['madde_no']}"
        if doc.get('fikra_no'):
            kaynak_metni += f", Fıkra {doc['fikra_no']}"

        sources.append({
            "belge": doc['belge'],
            "madde_no": doc['madde_no'],
            "fikra_no": doc.get('fikra_no'),
            "kaynak_metni": kaynak_metni,
            "icerik": doc['icerik'][:400] + "..." if len(doc['icerik']) > 400 else doc['icerik'],
            "score": round(r['final_score'], 2)
        })

    # LLM ile cevap
    answer = create_llm_answer(results, q.question, q.temperature)

    print(f"✅ Cevap hazır ({len(answer)} karakter)")
    print(f"{'=' * 70}\n")

    return ChatResponse(
        question=q.question,
        answer=answer,
        sources=sources,
        session_id=q.session_id
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "documents": len(all_documents),
        "llm_available": gemini_available
    }
@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """
    Session'ı sil (şu an sadece bilgilendirme)
    """
    print(f"🗑️  Session silindi: {session_id}")
    return {"message": "Session silindi", "session_id": session_id}

# ============================================================================
# ÇALIŞTIRMA
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("🚀 Selçuk Üniversitesi Mevzuat Chatbot API v2.0")
    print("=" * 70)
    print(f"📚 Toplam Madde: {len(all_documents)}")
    print(f"🤖 Embedding: BERTurk")
    print(f"🧠 LLM: {'Gemini 2.5 Flash ✅' if gemini_available else 'Yok ❌'}")
    print(f"🔍 Search: Hybrid + Smart Keyword Filtering")
    print("=" * 70)
    print("\n📡 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)