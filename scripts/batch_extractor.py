# enrich_json.py - v3.0 (GENİŞLETİLMİŞ METADATA)

import json
import re


def normalize_text(text):
    """Metni normalize et"""
    return text.lower().strip()


def detect_document_type(belge_adi: str) -> dict:
    """
    Belge türünü ve önceliğini tespit et
    v3.0: Genişletilmiş tanıma + Dosya adı desteği
    """
    belge_normalized = normalize_text(belge_adi)

    # ÜNİVERSİTE GENEL YÖNETMELİKLER (En yüksek öncelik)
    university_general_keywords = [
        "lisansüstü eğitim ve öğretim yönetmeliği",
        "ön lisans ve lisans eğitim",
        "genel eğitim-öğretim",
        "üniversite senatosu",
        "yükseköğretim kurumları",
        "öğrenci disiplin",
        "yatay geçiş",
        "dikey geçiş",
        "çift anadal",
        "yan dal",
        "erasmus",
        "mevlana",
        "farabi"
    ]

    for keyword in university_general_keywords:
        if keyword in belge_normalized:
            print(f"  ✨ Üniversite genel: {belge_adi[:60]}")
            return {"type": "university_general", "priority": 10}

    # TEKNOLOJİ FAKÜLTESİ (Özel öncelik)
    teknoloji_keywords = [
        "teknoloji fakültesi",
        "bilgisayar mühendisliği",
        "bilgisayar müh",
        "yazılım mühendisliği",
        "ime",  # İnternet ve Mobil Erişim
        "web teknolojileri",
        "mobil programlama"
    ]

    for keyword in teknoloji_keywords:
        if keyword in belge_normalized:
            print(f"  🔥 Teknoloji Fakültesi: {belge_adi[:60]}")
            return {
                "type": "faculty_specific",
                "priority": 9,
                "fakulte": "Teknoloji Fakültesi"
            }

    # STAJ BELGELERİ (Özel durum)
    staj_keywords = ["staj yönergesi", "staj uygulama", "staj rehberi", "staj esasları"]
    for keyword in staj_keywords:
        if keyword in belge_normalized:
            # Eğer teknoloji ile ilgiliyse
            if any(tek in belge_normalized for tek in ["teknoloji", "bilgisayar", "yazılım"]):
                print(f"  🎯 Teknoloji Fak. Staj: {belge_adi[:60]}")
                return {
                    "type": "faculty_specific",
                    "priority": 9,
                    "fakulte": "Teknoloji Fakültesi"
                }
            else:
                print(f"  📋 Genel Staj Belgesi: {belge_adi[:60]}")
                return {"type": "university_general", "priority": 8}

    # PROJEVBİTİRME İŞLERİ BELGELERİ
    proje_keywords = [
        "mezuniyet projesi",
        "bitirme projesi",
        "tasarım projesi",
        "proje şablonu",
        "proje uygulama"
    ]

    for keyword in proje_keywords:
        if keyword in belge_normalized:
            if any(tek in belge_normalized for tek in ["teknoloji", "bilgisayar", "yazılım"]):
                print(f"  🎯 Teknoloji Fak. Proje: {belge_adi[:60]}")
                return {
                    "type": "faculty_specific",
                    "priority": 9,
                    "fakulte": "Teknoloji Fakültesi"
                }
            else:
                print(f"  📋 Genel Proje Belgesi: {belge_adi[:60]}")
                return {"type": "program_specific", "priority": 7}

    # PEDAGOJİK FORMASYON (Düşük öncelik)
    if "pedagojik" in belge_normalized or "formasyon" in belge_normalized:
        print(f"  ⚠️  Pedagojik Formasyon (düşük): {belge_adi[:60]}")
        return {"type": "program_specific", "priority": 4}

    # DİĞER FAKÜLTELER
    fakulte_map = {
        "Tıp Fakültesi": ["tıp fakültesi", "tıp fak"],
        "Diş Hekimliği Fakültesi": ["diş hekimliği"],
        "Veteriner Fakültesi": ["veteriner fakültesi"],
        "Hukuk Fakültesi": ["hukuk fakültesi"],
        "Güzel Sanatlar Fakültesi": ["güzel sanatlar"],
        "Mühendislik Fakültesi": ["mühendislik fakültesi", "makine müh", "inşaat müh"],
        "Ziraat Fakültesi": ["ziraat fakültesi"],
        "Turizm Fakültesi": ["turizm fakültesi"],
        "Fen Fakültesi": ["fen fakültesi"],
        "Sağlık Bilimleri Fakültesi": ["sağlık bilimleri"],
    }

    for fakulte_name, keywords in fakulte_map.items():
        for keyword in keywords:
            if keyword in belge_normalized:
                print(f"  🏛️  Fakülte: {fakulte_name} → {belge_adi[:60]}")
                return {
                    "type": "faculty_specific",
                    "priority": 8,
                    "fakulte": fakulte_name
                }

    # VARSAYILAN
    print(f"  ❓ Sınıflandırılamadı: {belge_adi[:60]}")
    return {"type": "other", "priority": 5}


def enrich_documents(input_json_path: str, output_json_path: str):
    """
    JSON'u zenginleştir: belge_tipi, öncelik, fakülte ekle
    """
    print(f"📂 Dosya okunuyor: {input_json_path}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"📊 Toplam {len(documents)} madde bulundu\n")
    print("🔍 Metadata zenginleştirme başlıyor...\n")

    for doc in documents:
        belge_adi = doc.get("belge", "")

        # Metadata tespit et
        metadata = detect_document_type(belge_adi)

        # JSON'a ekle
        doc["belge_tipi"] = metadata["type"]
        doc["oncelik"] = metadata["priority"]

        if "fakulte" in metadata:
            doc["fakulte"] = metadata["fakulte"]
        else:
            doc["fakulte"] = None

    print(f"\n✅ Metadata zenginleştirme tamamlandı!")
    print(f"💾 Kaydediliyor: {output_json_path}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"✅ Tamamlandı! {len(documents)} madde kaydedildi.\n")

    # İstatistikler
    stats = {}
    for doc in documents:
        belge_tipi = doc.get("belge_tipi", "unknown")
        stats[belge_tipi] = stats.get(belge_tipi, 0) + 1

    print("📊 İSTATİSTİKLER:")
    for tip, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {tip}: {count} madde")

    # Teknoloji Fakültesi kontrolü
    tek_count = sum(1 for doc in documents if doc.get("fakulte") == "Teknoloji Fakültesi")
    print(f"\n🔥 Teknoloji Fakültesi: {tek_count} madde")


if __name__ == "__main__":
    import os

    # 🆕 Script'in bulunduğu dizini bul
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Bir üst dizin

    # 🆕 Dosya yollarını proje köküne göre ayarla
    INPUT_JSON = os.path.join(PROJECT_ROOT, "tum_mevzuat_maddeleri.json")
    OUTPUT_JSON = os.path.join(PROJECT_ROOT, "tum_mevzuat_maddeleri_enriched.json")

    print(f"📂 JSON Yolu: {INPUT_JSON}")
    print(f"💾 Çıktı Yolu: {OUTPUT_JSON}\n")

    enrich_documents(INPUT_JSON, OUTPUT_JSON)

