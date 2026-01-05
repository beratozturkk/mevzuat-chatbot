# scripts/enrich_json.py - JSON'a Akıllı Metadata Ekle

import json
import re
import os
import unicodedata

JSON_PATH = "../tum_mevzuat_maddeleri.json"
OUTPUT_PATH = "../tum_mevzuat_maddeleri_enriched.json"


def detect_document_type(belge_adi):
    """
    Belge tipini akıllıca tespit et ve öncelik ver
    """
    belge_lower = belge_adi.lower()

    # Türkçe karakter normalize (ö→o, ü→u, ş→s vb.)
    belge_normalized = ''.join(
        c for c in unicodedata.normalize('NFD', belge_lower)
        if unicodedata.category(c) != 'Mn'
    )
    belge_normalized = belge_normalized.replace('ı', 'i').replace('ş', 's').replace('ğ', 'g').replace('ü', 'u').replace(
        'ö', 'o').replace('ç', 'c')

    # 🆕 0. SSS (ÇOK YÜKSEK ÖNCELİK - SADECE BU ÖZEL!)
    if 'sss' in belge_lower or 'sıkça sorulan' in belge_lower or 'sikca sorulan' in belge_normalized:
        print(f"  🔥 SSS belgesi (ÇOK ÖNEMLİ): {belge_adi[:60]}...")
        return {'type': 'program_specific', 'priority': 10}

    # 1. ÜNİVERSİTE GENEL YÖNETMELİKLER (En yüksek öncelik)
    if (('on lisans' in belge_normalized or 'ön lisans' in belge_lower) and
            ('egitim' in belge_normalized or 'öğretim' in belge_lower) and
            ('sinav' in belge_normalized or 'sınav' in belge_lower)):
        print(f"  ✅ Genel yönetmelik: {belge_adi[:60]}...")
        return {'type': 'university_general', 'priority': 10}

    if 'lisansustu' in belge_normalized or 'lisansüstü' in belge_lower:
        print(f"  ✅ Lisansüstü yönetmelik: {belge_adi[:60]}...")
        return {'type': 'university_general', 'priority': 10}

    if 'ogrenci disiplin' in belge_normalized or 'öğrenci disiplin' in belge_lower:
        print(f"  ✅ Disiplin yönetmeliği: {belge_adi[:60]}...")
        return {'type': 'university_general', 'priority': 10}

    if 'uzaktan ogretim' in belge_normalized or 'uzaktan öğretim' in belge_lower:
        print(f"  ✅ Uzaktan öğretim: {belge_adi[:60]}...")
        return {'type': 'university_general', 'priority': 10}

    # 2. PROGRAM SPESİFİK (Yüksek öncelik - yan dal, çap, staj, uygulama esasları vb.)
    program_keywords = [
        'yan dal', 'yandal',
        'cift anadal', 'cift ana dal', 'çift anadal', 'çift ana dal', 'cap', 'çap',
        'erasmus', 'mevlana', 'farabi',
        'staj',
        'yatay gecis', 'yatay geçiş',
        'dikey gecis', 'dikey geçiş',
        'intibak',
        'azami sure', 'azami süre',
        'pedagojik formasyon', 'pedagojik',
        'uygulama esaslari', 'uygulama esasları'  # 🆕 Eklendi
    ]

    for keyword in program_keywords:
        keyword_normalized = keyword.replace('ç', 'c').replace('ı', 'i').replace('ş', 's').replace('ğ', 'g').replace(
            'ü', 'u').replace('ö', 'o')
        if keyword in belge_lower or keyword_normalized in belge_normalized:
            print(f"  📋 Program spesifik: {belge_adi[:60]}...")
            return {'type': 'program_specific', 'priority': 8}

    # 3. FAKÜLTE/YÜKSEKOKUL SPESİFİK (Orta öncelik)
    faculty_keywords = [
        'fakültesi', 'fakultesi',
        'yüksekokulu', 'yuksekokulu',
        'enstitüsü', 'enstitusu',
        'konservatuvar', 'konservatuar'
    ]

    if any(word in belge_lower or word.replace('ü', 'u').replace('ö', 'o') in belge_normalized for word in
           faculty_keywords):
        print(f"  🏛️  Fakülte spesifik: {belge_adi[:60]}...")
        return {'type': 'faculty_specific', 'priority': 5}

    # 4. ÖĞRENCİ TOPLULUKLARI, KOMİSYONLAR (Düşük öncelik)
    low_priority_keywords = [
        'topluluk', 'komisyon', 'kurul', 'konsey',
        'bilgi edinme', 'işyeri hekimliği', 'isyeri hekimligi'
    ]

    if any(word in belge_lower or word.replace('ş', 's').replace('ı', 'i') in belge_normalized for word in
           low_priority_keywords):
        print(f"  ℹ️  Düşük öncelik: {belge_adi[:60]}...")
        return {'type': 'low_priority', 'priority': 3}

    # 5. GENEL YÖNETMELİK (Orta-Yüksek)
    if 'yönetmeliği' in belge_lower or 'yönergesi' in belge_lower or 'yonetmeligi' in belge_normalized or 'yonergesi' in belge_normalized:
        print(f"  📄 Genel düzenleme: {belge_adi[:60]}...")
        return {'type': 'general_regulation', 'priority': 7}

    # 6. DİĞER
    print(f"  ❓ Kategorize edilemedi: {belge_adi[:60]}...")
    return {'type': 'other', 'priority': 4}



def extract_faculty_name(belge_adi):
    """
    Belge adından fakülte/yüksekokul ismini çıkar
    """
    patterns = [
        r'([\wğüşöçıİ\s]+?)\s*Fakültesi',
        r'([\wğüşöçıİ\s]+?)\s*Yüksekokulu',
        r'([\wğüşöçıİ\s]+?)\s*Enstitüsü',
        r'([\wğüşöçıİ\s]+?)\s*Konservatuvar'
    ]

    for pattern in patterns:
        match = re.search(pattern, belge_adi, re.IGNORECASE)
        if match:
            faculty = match.group(1).strip()
            # Temizle (sayıları çıkar)
            faculty = re.sub(r'\d+', '', faculty).strip()
            return faculty

    return None


def enrich_json():
    """
    JSON'u zenginleştir - metadata ekle
    """
    print("\n" + "=" * 70)
    print("📚 JSON Zenginleştirme Başlıyor")
    print("=" * 70 + "\n")

    # JSON'u yükle
    if not os.path.exists(JSON_PATH):
        print(f"❌ JSON dosyası bulunamadı: {JSON_PATH}")
        return

    print(f"📄 JSON yükleniyor: {JSON_PATH}")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ {len(data)} madde yüklendi\n")
    print("🔧 Metadata ekleniyor...\n")

    # HER BELGE İÇİN METADATA HAZIRLA (ÖNCE)
    belge_metadata = {}
    unique_belgeler = set()

    for item in data:
        belge_adi = item['belge']
        if belge_adi not in unique_belgeler:
            unique_belgeler.add(belge_adi)
            doc_info = detect_document_type(belge_adi)
            fakulte = extract_faculty_name(belge_adi)
            belge_metadata[belge_adi] = {
                'type': doc_info['type'],
                'priority': doc_info['priority'],
                'fakulte': fakulte
            }

    # ŞIMDI TÜM MADDELERE UYGULA
    stats = {}
    for item in data:
        belge_adi = item['belge']
        metadata = belge_metadata[belge_adi]

        item['belge_tipi'] = metadata['type']
        item['oncelik'] = metadata['priority']
        item['fakulte'] = metadata['fakulte']

        stats[metadata['type']] = stats.get(metadata['type'], 0) + 1

    # Yeni JSON'u kaydet
    print(f"\n💾 Zenginleştirilmiş JSON kaydediliyor: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Başarıyla kaydedildi!")

    # İstatistikleri göster
    print("\n" + "=" * 70)
    print("📊 Zenginleştirme İstatistikleri")
    print("=" * 70 + "\n")

    print(f"📚 Toplam Madde: {len(data)}")
    print(f"📄 Benzersiz Belge: {len(unique_belgeler)}")
    print(f"\n📋 Belge Tipi Dağılımı:")

    for doc_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(data)) * 100
        print(f"   - {doc_type.ljust(25)}: {count:4d} madde ({percentage:.1f}%)")

    print("\n✨ Zenginleştirme tamamlandı!\n")
    print(f"🔄 Sonraki adım:")
    print(f"   python build_chromadb.py\n")


if __name__ == "__main__":
    enrich_json()
