import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm



JSON_PATH = "/Users/beratozturk/Desktop/mevzuat_chatbot/tum_mevzuat_maddeleri.json"   # batch_extractor çıktısı
CHROMA_DIR = "../mevzuat_db"                  # veritabanı klasörü


# Küçük ama etkili bir model -> Türkçe ve İngilizce için iyi
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


if not os.path.exists(JSON_PATH):
    print(f"❌ JSON dosyası bulunamadı: {JSON_PATH}")
    print("Lütfen önce batch_extractor.py dosyasını çalıştır.")
    exit()

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [d["icerik"] for d in data]
metadata = [{"belge": d["belge"], "madde_no": d["madde_no"]} for d in data]

print(f"📚 Toplam {len(texts)} madde yüklendi.")
print("🔢 Embedding’ler oluşturuluyor...")


db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadata,
    persist_directory=CHROMA_DIR
)

db.persist()
print(f"✅ ChromaDB başarıyla oluşturuldu: {CHROMA_DIR}")
