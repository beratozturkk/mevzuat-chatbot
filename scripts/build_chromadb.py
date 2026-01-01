# scripts/build_chromadb.py (BERTurk ile)
import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

JSON_PATH = "../tum_mevzuat_maddeleri_old.json"
CHROMA_DIR = "../mevzuat_db"

# BERTurk - Türkçe için optimize edilmiş! 🇹🇷
print("🤖 BERTurk embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="dbmdz/bert-base-turkish-cased",  # TÜRKÇE BERT!
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ BERTurk hazır!")

if not os.path.exists(JSON_PATH):
    print(f"❌ JSON dosyası bulunamadı: {JSON_PATH}")
    exit()

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [d["icerik"] for d in data]
metadata = [
    {
        "belge": d["belge"],
        "madde_no": d["madde_no"],
        "fikra_no": d.get("fikra_no")
    }
    for d in data
]

print(f"📚 Toplam {len(texts)} chunk yüklendi.")
print("🔢 BERTurk embedding'ler oluşturuluyor (biraz uzun sürebilir)...")

db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadata,
    persist_directory=CHROMA_DIR
)

db.persist()
print(f"✅ ChromaDB (BERTurk ile) başarıyla oluşturuldu: {CHROMA_DIR}")
