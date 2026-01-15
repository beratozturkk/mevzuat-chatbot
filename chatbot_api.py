from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Optional
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
JSON_PATH = "tum_mevzuat_maddeleri_enriched.json"

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
    fakulte_filter: Optional[str] = None  # 🆕 YENİ PARAMETRE
    session_id: str = "default"
    top_k: int = 5
    temperature: float = 0.3


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list
    session_id: str


# ============================================================================
# 🆕 FAKÜLTE TESPİT FONKSİYONU
# ============================================================================

def detect_fakulte(query: str) -> Optional[str]:
    """Sorgudan fakülte/birim tespit et"""
    query_lower = query.lower()

    # Fakülte anahtar kelimeleri
    fakulte_map = {
        'Teknoloji Fakültesi': ['teknoloji', 'bilgisayar', 'yazılım', 'programlama', 'mühendislik'],
        'Tıp Fakültesi': ['tıp', 'hastane', 'klinik', 'cerrahi', 'anatomi', 'hekimlik'],
        'Diş Hekimliği Fakültesi': ['diş hekimliği', 'diş', 'dental'],
        'Veteriner Fakültesi': ['veteriner', 'hayvan', 'veterinerlik'],
        'Hukuk Fakültesi': ['hukuk', 'kanun', 'mahkeme', 'dava', 'avukat'],
        'Güzel Sanatlar Fakültesi': ['güzel sanatlar', 'resim', 'heykel', 'müzik'],
        'Sağlık Bilimleri Fakültesi': ['sağlık bilimleri', 'fizyoterapi', 'beslenme'],
        'Hemşirelik Fakültesi': ['hemşirelik', 'hemşire'],
        'Mühendislik Fakültesi': ['mühendislik', 'inşaat', 'makine', 'elektrik'],
        'Ziraat Fakültesi': ['ziraat', 'tarım', 'çiftçilik'],
        'Dilek Sabancı Devlet Konservatuarı': ['konservatuvar', 'müzik', 'sahne'],
    }

    for fakulte_name, keywords in fakulte_map.items():
        if any(kw in query_lower for kw in keywords):
            print(f"🎯 Otomatik tespit: {fakulte_name}")
            return fakulte_name

    return None


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
    """Anahtar kelimeleri çıkar"""
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


def hybrid_search(query: str, fakulte_filter: Optional[str] = None, top_k: int = 5) -> List[Dict]:
    """
    🔥 v7.0: GENEL METADATA FİLTRE + FAKÜLTE PARAMETRESİ
    """
    query_normalized = normalize_text(query)

    # "yandal" → "yan dal"
    if 'yandal' in query_normalized:
        query_normalized = query_normalized.replace('yandal', 'yan dal')
        print(f"🔄 Query normalize: 'yandal' → 'yan dal'")

    # "çiftanadal" → "çift anadal"
    if 'çiftanadal' in query_normalized or 'çiftanaadal' in query_normalized:
        query_normalized = query_normalized.replace('çiftanadal', 'çift anadal')
        query_normalized = query_normalized.replace('çiftanaadal', 'çift anadal')
        print(f"🔄 Query normalize: 'çift anadal'")

    # Query expansion
    query_expansions = {
        'cap': 'çift anadal çap',
        'çap': 'çift anadal çap',
        'gano': 'genel ağırlıklı not ortalaması',
    }

    for short, expanded in query_expansions.items():
        if short in query_normalized:
            query_normalized = query_normalized.replace(short, expanded)
            print(f"🔄 Query genişletme: '{short}' → '{expanded}'")

    keywords = extract_keywords(query_normalized)
    print(f"🔍 Anahtar kelimeler: {keywords}")

    # 1. BM25 Search
    bm25_scores = bm25.get_scores(keywords)

    # 2. 🆕 GENEL METADATA FİLTRE
    try:
        where_filter = None

        # 🆕 1. Kullanıcı UI'dan fakülte seçtiyse
        if fakulte_filter:
            where_filter = {
                "$or": [
                    {"fakulte": {"$eq": fakulte_filter}},
                    {"belge_tipi": {"$eq": "university_general"}}
                ]
            }
            print(f"🎯 UI Seçimi: {fakulte_filter}")

        # 🆕 2. Seçmediyse otomatik tespit et
        else:
            detected_fakulte = detect_fakulte(query)

            # Lisansüstü tespiti
            is_lisansustu = any(term in query_normalized for term in [
                "lisansüstü", "yüksek lisans", "doktora", "master", "phd", "tezli", "tezsiz"
            ])

            # Pedagojik formasyon tespiti
            is_pedagojik = "pedagojik" in query_normalized or "formasyon" in query_normalized

            # Fakülte tespit edildiyse ve lisansüstü/pedagojik değilse filtrele
            if detected_fakulte and not is_lisansustu and not is_pedagojik:
                where_filter = {
                    "$or": [
                        {"fakulte": {"$eq": detected_fakulte}},
                        {"belge_tipi": {"$eq": "university_general"}}
                    ]
                }
                print(f"🎯 Otomatik tespit: {detected_fakulte}")

        # Semantic search
        if where_filter:
            semantic_results = db.similarity_search_with_score(
                query_normalized,
                k=top_k * 3,
                filter=where_filter
            )
        else:
            semantic_results = db.similarity_search_with_score(query_normalized, k=top_k * 3)

    except Exception as e:
        print(f"⚠️  Semantic search hatası: {e}")
        semantic_results = []

    # 3. Sonuçları birleştir
    candidates = {}

    for idx, score in enumerate(bm25_scores):
        if score > 0:
            doc_id = f"doc_{idx}"
            candidates[doc_id] = {
                'doc': all_documents[idx],
                'bm25_score': float(score),
                'semantic_score': 0.0,
                'content': all_documents[idx]['icerik']
            }

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

    # 4. KEYWORD BOOST + N-GRAM
    domain_keywords = {
        'teknoloji': ['teknoloji fakültesi', 'bilgisayar mühendisliği', 'yazılım'],
        'bilgisayar': ['bilgisayar mühendisliği', 'teknoloji', 'yazılım'],
        'yazılım': ['bilgisayar', 'programlama', 'kod'],
        'proje': ['mezuniyet projesi', 'bitirme projesi', 'tasarım projesi'],
        'mezuniyet': ['mezuniyet projesi', 'bitirme projesi'],
        'staj': ['staj', 'uygulama', 'iş yeri', 'zorunlu staj'],
        'gano': ['genel ağırlıklı not ortalaması', 'ortalama'],
        'çift': ['çift anadal', 'çap'],
        'yan': ['yan dal', 'yandal'],
        'sss': ['sıkça sorulan', 'soru', 'cevap'],
    }

    # N-gram (çift kelime)
    query_words = query_normalized.split()
    query_bigrams = []
    for i in range(len(query_words) - 1):
        bigram = f"{query_words[i]} {query_words[i + 1]}"
        query_bigrams.append(bigram)

    if query_bigrams:
        print(f"🔍 Bigrams: {query_bigrams}")

    for doc_id, result in candidates.items():
        content_lower = result['content'].lower()
        belge_lower = result['doc']['belge'].lower()

        keyword_boost = 0
        match_count = 0

        # Çift kelime boost
        for bigram in query_bigrams:
            if bigram in content_lower:
                keyword_boost += 5.0
                match_count += 1.5
                print(f"    🎯 Bigram eşleşme: '{bigram}'")

            if bigram in belge_lower:
                keyword_boost += 15.0
                match_count += 3.0
                print(f"    💎 Belge adında bigram: '{bigram}'")

        # Tek kelime boost
        for kw in keywords:
            if kw in content_lower:
                match_count += 1
                keyword_boost += 2.0

            if kw in belge_lower:
                keyword_boost += 10.0
                match_count += 2
                print(f"    🎯 Belge adı eşleşme: '{kw}'")

            # Domain keywords
            if kw in domain_keywords:
                related_words = domain_keywords[kw]
                for rel_word in related_words:
                    if rel_word in content_lower:
                        keyword_boost += 1.5
                        match_count += 0.5
                        break
                    if rel_word in belge_lower:
                        keyword_boost += 3.0
                        match_count += 1.0
                        break

        if keywords:
            match_ratio = match_count / len(keywords)
            keyword_boost *= match_ratio

        result['keyword_boost'] = keyword_boost
        result['match_count'] = match_count

    # 5. 🆕 GENEL METADATA BOOST
    print("\n🏷️  Metadata boost uygulanıyor...")
    query_lower = query.lower()

    for doc_id, result in candidates.items():
        doc = result['doc']
        belge_lower = doc['belge'].lower()

        belge_tipi = doc.get('belge_tipi', 'other')
        oncelik = doc.get('oncelik', 5)
        fakulte = doc.get('fakulte')

        # Fakülte kontrolü
        fakulte_in_query = False
        if fakulte:
            fakulte_lower = fakulte.lower()
            fakulte_in_query = fakulte_lower in query_lower

        # Pedagojik formasyon ceza
        is_pedagojik_belge = "pedagojik" in belge_lower or "formasyon" in belge_lower
        is_pedagojik_query = "pedagojik" in query_lower or "formasyon" in query_lower

        if is_pedagojik_belge and not is_pedagojik_query:
            result['keyword_boost'] -= 8.0
            print(f"  ⛔ Pedagojik ceza: {doc['belge'][:50]}")

        # 🆕 GENEL BOOST SİSTEMİ
        if belge_tipi == 'university_general':
            result['keyword_boost'] += 8.0
            print(f"  ✨ Genel yönetmelik: {doc['belge'][:50]}")

        elif belge_tipi == 'faculty_specific':
            if fakulte and fakulte_in_query:
                result['keyword_boost'] += 12.0  # İstenilen fakülte
                print(f"  🔥 Fakülte eşleşme: {fakulte}")
            elif fakulte:
                # Başka fakülte adı soruda geçiyorsa ceza
                other_fakulte_keywords = ['tıp', 'hukuk', 'mühendislik', 'teknoloji',
                                          'ziraat', 'veteriner', 'güzel sanatlar']
                if any(kw in query_lower for kw in other_fakulte_keywords):
                    result['keyword_boost'] -= 4.0
                    print(f"  ⚠️  Yanlış fakülte ceza: {fakulte}")

        elif belge_tipi == 'program_specific':
            result['keyword_boost'] += (oncelik * 0.5)

        elif belge_tipi == 'low_priority':
            result['keyword_boost'] -= 2.0

        result['priority_raw'] = oncelik

    # 6. Final Skor
    print("\n🎯 Final skor hesaplanıyor...")
    for doc_id in candidates:
        c = candidates[doc_id]

        if c['match_count'] < 0.5 and c['semantic_score'] < 0.7:
            c['final_score'] = 0.0
            continue

        priority_raw = c.get('priority_raw', 5)
        priority_normalized = (priority_raw - 3) / 7.0
        priority_normalized = max(0, min(1, priority_normalized))

        c['final_score'] = (
                c['bm25_score'] * 0.35 +
                c['semantic_score'] * 0.20 +
                c['keyword_boost'] * 0.25 +
                priority_normalized * 0.20
        )

        if priority_raw >= 9:
            print(f"  🔥 Yüksek öncelik: {c['doc']['belge'][:50]}")

        if c['final_score'] < 2.0:
            c['final_score'] = 0.0

    # 7. Sırala
    sorted_results = sorted(
        [c for c in candidates.values() if c['final_score'] > 0],
        key=lambda x: x['final_score'],
        reverse=True
    )

    # DEBUG
    if len(sorted_results) > 0:
        print("\n" + "=" * 80)
        print("🔍 DEBUG: Top 10 Sonuç")
        print("=" * 80)
        for i, r in enumerate(sorted_results[:10], 1):
            doc = r['doc']
            print(f"\n{i}. {doc['belge'][:60]}")
            print(f"   Madde: {doc['madde_no']}")
            print(f"   📊 BM25: {r['bm25_score']:.2f} | Semantic: {r['semantic_score']:.2f}")
            print(f"   🏷️  Boost: {r['keyword_boost']:.2f} | Priority: {r.get('priority_raw', 5)}")
            print(f"   ⭐ FINAL: {r['final_score']:.2f}")
            print(f"   🏛️  Fakülte: {doc.get('fakulte', 'yok')}")
        print("=" * 80 + "\n")

    return sorted_results[:top_k]


def create_llm_answer(results: List[Dict], query: str, temperature: float = 0.3) -> str:
    """Gemini ile cevap üret"""
    if not results:
        return "Üzgünüm, bu konuda mevzuatlarda ilgili bilgi bulamadım. Lütfen sorunuzu farklı kelimelerle ifade etmeyi deneyin."

    if not gemini_available:
        return create_fallback_answer(results, query)

    context = ""
    for i, result in enumerate(results[:3], 1):
        doc = result['doc']
        context += f"\n\n--- KAYNAK {i} ---\n"
        context += f"Belge: {doc['belge']}\n"
        context += f"Madde No: {doc['madde_no']}\n"
        if doc.get('fikra_no'):
            context += f"Fıkra No: {doc['fikra_no']}\n"
        context += f"İçerik:\n{doc['icerik']}\n"

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

CEVAP:"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                'temperature': temperature,
                'max_output_tokens': 4096,
            }
        )

        answer = response.text
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
    """Fallback"""
    if not results:
        return "Üzgünüm, bu konuda bilgi bulamadım."

    answer = f"### İlgili Mevzuat Maddeleri\n\n"

    for i, result in enumerate(results[:3], 1):
        doc = result['doc']
        content = doc['icerik'][:500].strip()
        if len(doc['icerik']) > 500:
            content += "..."

        answer += f"#### {i}. {doc['belge']} - Madde {doc['madde_no']}\n\n"
        answer += f"{content}\n\n---\n\n"

    answer += "⚠️  **Not:** LLM aktif değil.\n"
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
        "model": "BERTurk + Gemini + Fakulte Filtre v7.0"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz!")

    print(f"\n{'=' * 70}")
    print(f"🔍 Soru: {q.question}")
    if q.fakulte_filter:
        print(f"🏛️  Fakülte filtresi: {q.fakulte_filter}")
    print(f"🌡️  Temperature: {q.temperature}")

    # 🆕 Fakülte filtresi ile arama
    results = hybrid_search(q.question, fakulte_filter=q.fakulte_filter, top_k=q.top_k)

    if not results:
        print("❌ Sonuç yok")
        return ChatResponse(
            question=q.question,
            answer="Üzgünüm, bu konuda mevzuatlarda ilgili bilgi bulamadım.",
            sources=[],
            session_id=q.session_id
        )

    print(f"\n📚 {len(results)} sonuç bulundu:")
    for i, r in enumerate(results[:5], 1):
        doc = r['doc']
        print(f"   {i}. {doc['belge'][:60]} - M{doc['madde_no']} (skor: {r['final_score']:.2f})")

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
            "score": round(r['final_score'], 2),
            "priority": r.get('priority_raw', 5)
        })

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
    print(f"🗑️  Session silindi: {session_id}")
    return {"message": "Session silindi", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("🚀 Mevzuat Chatbot API v7.0 (Fakulte Filtre)")
    print("=" * 70)
    print(f"📚 Toplam Madde: {len(all_documents)}")
    print(f"🤖 Embedding: BERTurk")
    print(f"🧠 LLM: {'Gemini ✅' if gemini_available else 'Yok ❌'}")
    print(f"🔍 Search: Hybrid + N-gram + Fakulte Filtre")
    print("=" * 70)
    print("\n📡 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
