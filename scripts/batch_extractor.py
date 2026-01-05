# batch_extractor.py (TAM İÇERİK OKUMA - FIXED)
import os
import re
import json
from tqdm import tqdm

# PDF okuyucu seç
USE_PDFPLUMBER = False
pdfplumber = None
PdfReader = None

try:
    import pdfplumber

    USE_PDFPLUMBER = True
    print("✅ pdfplumber kullanılıyor")
except ImportError:
    try:
        from PyPDF2 import PdfReader

        print("⚠️ PyPDF2 kullanılıyor (pdfplumber önerilir)")
    except ImportError:
        print("❌ Ne pdfplumber ne de PyPDF2 yüklü!")
        print("Lütfen çalıştırın: pip install pdfplumber")
        exit(1)


def pdf_to_text(pdf_path):
    """PDF'ten text çıkar"""
    try:
        if USE_PDFPLUMBER and pdfplumber:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        elif PdfReader:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        else:
            return ""
    except Exception as e:
        print(f"❌ Hata ({os.path.basename(pdf_path)}): {e}")
        return ""


def split_into_maddeler(text, belge_adi):
    """Metni maddelere ayır - Çoklu format desteği"""

    # 1️⃣ ÖNCE STANDART "MADDE" FORMATI DENE
    madde_pattern = r'(?:^|\n)\s*MADDE\s*[-–:.]?\s*(\d+)\s*[-–:.]?'
    matches = list(re.finditer(madde_pattern, text, re.IGNORECASE | re.MULTILINE))

    if matches:
        print(f"  ✅ MADDE formatı bulundu ({len(matches)} madde)")
        return _extract_chunks(text, matches, belge_adi, "madde")

    # 2️⃣ SAYILI PARAGRAF FORMATI (1), 2), 3), ...)
    numbered_pattern = r'(?:^|\n)\s*(\d+)\)\s+'
    matches = list(re.finditer(numbered_pattern, text, re.MULTILINE))

    # En az 5 tane olmalı (yoksa yanlış eşleşme olabilir)
    if len(matches) >= 5:
        print(f"  ✅ Numaralı format bulundu ({len(matches)} madde)")
        return _extract_chunks(text, matches, belge_adi, "madde")

    # 3️⃣ SORU-CEVAP FORMATI
    soru_pattern = r'Soru\s+(\d+|G):\s*\n(.*?)\nCevap:\s*\n(.*?)(?=\nSoru|\Z)'
    matches = list(re.finditer(soru_pattern, text, re.DOTALL))

    if matches:
        print(f"  ✅ SSS formatı bulundu ({len(matches)} soru)")
        chunks = []
        for match in matches:
            soru_no = match.group(1)
            soru = match.group(2).strip()
            cevap = match.group(3).strip()

            chunks.append({
                "belge": belge_adi,
                "madde_no": f"S{soru_no}",
                "fikra_no": None,
                "icerik": f"SORU: {soru}\n\nCEVAP: {cevap}"
            })
        return chunks

    # 4️⃣ HİÇBİRİ BULUNAMADI - TÜM BELGE TEK PARÇA
    print(f"  ⚠️ Madde yok, tüm içerik tek parça")
    return [{
        "belge": belge_adi,
        "madde_no": "0",
        "fikra_no": None,
        "icerik": text.strip()
    }]


def _extract_chunks(text, matches, belge_adi, chunk_type):
    """Eşleşmelerden chunk'ları çıkar (ortak fonksiyon)"""
    chunks = []

    for i, match in enumerate(matches):
        madde_no = match.group(1)
        start_pos = match.start()

        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        madde_full = text[start_pos:end_pos].strip()

        if madde_full:
            chunks.append({
                "belge": belge_adi,
                "madde_no": madde_no,
                "fikra_no": None,
                "icerik": madde_full
            })

    return chunks

    for i, match in enumerate(matches):
        madde_no = match.group(1)
        start_pos = match.start()

        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        madde_full = text[start_pos:end_pos].strip()

        if madde_full:
            chunks.append({
                "belge": belge_adi,
                "madde_no": madde_no,
                "fikra_no": None,
                "icerik": madde_full
            })

    return chunks


def clean_belge_name(filename):
    """Dosya adını temizle"""
    name = filename.replace(".pdf", "")
    name = name.replace("_", " ")
    name = re.sub(r'\d{15,}', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def process_all_pdfs(pdf_folder):
    """Tüm PDF'leri işle"""
    all_data = []

    if not os.path.exists(pdf_folder):
        print(f"❌ Klasör bulunamadı: {pdf_folder}")
        return []

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ PDF bulunamadı")
        return []

    print(f"📚 {len(pdf_files)} PDF bulundu\n")

    success = 0
    failed = 0

    for filename in tqdm(pdf_files, desc="PDF'ler işleniyor"):
        path = os.path.join(pdf_folder, filename)
        text = pdf_to_text(path)

        if not text or len(text.strip()) < 50:
            print(f"❌ Boş: {filename}")
            failed += 1
            continue

        belge_adi = clean_belge_name(filename)
        chunks = split_into_maddeler(text, belge_adi)

        if chunks:
            all_data.extend(chunks)
            success += 1
            first_len = len(chunks[0]['icerik'])
            print(f"✅ {belge_adi[:50]}... → {len(chunks)} madde (ilk: {first_len} kar)")
        else:
            failed += 1

    print(f"\n✅ Başarılı: {success}, ❌ Başarısız: {failed}")
    print(f"📊 Toplam madde: {len(all_data)}")

    return all_data


if __name__ == "__main__":
    PDF_FOLDER = "../Mevzuat"
    OUTPUT_FILE = "../tum_mevzuat_maddeleri.json"

    print("🚀 PDF İşleme Başlıyor...\n")

    all_chunks = process_all_pdfs(PDF_FOLDER)

    if all_chunks:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Kaydedildi: {OUTPUT_FILE}")

        unique = len(set([c['belge'] for c in all_chunks]))
        print(f"📋 Benzersiz belge: {unique}")

        lengths = [len(c['icerik']) for c in all_chunks]
        print(f"📏 Ort. madde: {sum(lengths) // len(lengths)} kar")
        print(f"📏 En uzun: {max(lengths)} kar")
    else:
        print("❌ Hiç veri yok!")
