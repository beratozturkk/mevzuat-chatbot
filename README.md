# Selçuk Üniversitesi Mevzuat ChatBot

Bu proje, Selçuk Üniversitesi mevzuat belgeleri (yönetmelikler, yönergeler, esaslar)
üzerinden doğal dilde soru-cevap yapılmasını sağlayan bir ChatBot uygulamasıdır.

## Proje Amacı
- Üniversite mevzuatına hızlı erişim sağlamak
- PDF tabanlı mevzuat belgelerini anlamlandırmak
- RAG (Retrieval-Augmented Generation) mimarisini uygulamak

## Kullanılan Teknolojiler
- Python
- LangChain
- ChromaDB
- LLM tabanlı modeller
- Streamlit tabanlı arayüz

## Proje Yapısı
- `app/` → Uygulama mantığı
- `scripts/` → Veri işleme ve embedding scriptleri
- `chatbot_api.py` → API katmanı
- `chatbot_ui.py` → Kullanıcı arayüzü

## Not
Bu projede kullanılan mevzuat belgeleri Selçuk Üniversitesi resmî web
sayfasından alınmış olup yalnızca akademik amaçla kullanılmıştır.
