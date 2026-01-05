# scripts/build_chromadb.py - Metadata ile ChromaDB Oluştur

import json
import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

JSON_PATH = "../tum_mevzuat_maddeleri_enriched.json"  # 👈 YENİ DOSYA
CHROMA_DIR = "../mevzuat_db"

print("🤖 BERTurk embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="dbmdz/bert-base-turkish-cased",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ BERTurk hazır!")

if not os.path.exists(JSON_PATH):
    print(f"❌ JSON dosyası bulunamadı: {JSON_PATH}")
    print("⚠️  Önce enrich_json.py çalıştırmalısınız!")
    exit()

print(f"📄 JSON yükleniyor...")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [d["icerik"] for d in data]
metadata = [
    {
        "belge": d["belge"],
        "madde_no": d["madde_no"],
        "fikra_no": d.get("fikra_no"),
        "belge_tipi": d.get("belge_tipi", "other"),    # 👈 YENİ
        "oncelik": d.get("oncelik", 5),                 # 👈 YENİ
        "fakulte": d.get("fakulte")                     # 👈 YENİ
    }
    for d in data
]

print(f"✅ {len(texts)} madde yüklendi")
print(f"📊 Metadata örneği:")
print(f"   - belge_tipi: {metadata[0]['belge_tipi']}")
print(f"   - oncelik: {metadata[0]['oncelik']}")
print(f"   - fakulte: {metadata[0]['fakulte']}\n")

# Eski DB'yi sil
if os.path.exists(CHROMA_DIR):
    print("🗑️  Eski ChromaDB siliniyor...")
    shutil.rmtree(CHROMA_DIR)
    print("✅ Silindi")

print("\n🔢 BERTurk embeddings oluşturuluyor...")
print("⏳ Bu 2-3 dakika sürebilir...\n")

db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadata,
    persist_directory=CHROMA_DIR
)

db.persist()

print("\n" + "=" * 70)
print("✅ ChromaDB (metadata ile) başarıyla oluşturuldu!")
print("=" * 70)
print(f"📁 Konum: {CHROMA_DIR}")
print(f"📚 Madde sayısı: {len(texts)}")
print(f"🏷️  Metadata alanları: belge_tipi, oncelik, fakulte")
