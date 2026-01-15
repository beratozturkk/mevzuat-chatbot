import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Mevzuat Chatbot", page_icon="📚", layout="wide")

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("📚 Teknoloji Fakültesi Mevzuat Chatbot")
st.markdown("BERTurk + Hybrid Search ile güçlendirilmiş mevzuat asistanı")

API_URL = "http://localhost:8000/chat"

# ============================================================================
# SIDEBAR - AYARLAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Ayarlar")

    # 🆕 FAKÜLTE SEÇİCİ
    st.subheader("🏛️ Fakülte/Birim Seçimi")
    fakulte_secimi = st.selectbox(
        "Aramayı daraltmak için fakülte seçin (opsiyonel)",
        options=[
            "Tümü (otomatik tespit)",
            "Teknoloji Fakültesi",
            "Tıp Fakültesi",
            "Diş Hekimliği Fakültesi",
            "Veteriner Fakültesi",
            "Hukuk Fakültesi",
            "Güzel Sanatlar Fakültesi",
            "Sağlık Bilimleri Fakültesi",
            "Hemşirelik Fakültesi",
            "İletişim Fakültesi",
            "Mimarlık Fakültesi",
            "Mühendislik Fakültesi",
            "Ziraat Fakültesi",
            "Dilek Sabancı Devlet Konservatuarı",
            "Yabancı Diller Yüksekokulu",
            "Sağlık Hizmetleri Meslek Yüksekokulu",
            "Beyşehir Ali Akkanat Uygulamalı Bilimler Yüksekokulu"
        ],
        help="Sorunuz belirli bir fakülte/birim ile ilgiliyse seçin. Sistem otomatik tespit de yapar."
    )

    # Fakülte seçildiyse bilgi göster
    if fakulte_secimi != "Tümü (otomatik tespit)":
        st.info(f"🎯 Arama **{fakulte_secimi}** ile sınırlandırıldı")

    st.markdown("---")

    # Temperature kontrolü
    st.subheader("🌡️ Cevap Ayarları")
    temperature = st.slider(
        "Temperature (Yaratıcılık)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Düşük: Daha tutarlı, Yüksek: Daha yaratıcı"
    )

    topk = st.slider(
        "Kaynak Sayısı",
        1, 10, 3,
        help="LLM'e gönderilecek maksimum madde sayısı"
    )

    st.markdown("---")

    # Konuşmayı sıfırla
    if st.button("🗑️ Konuşmayı Sıfırla", use_container_width=True):
        st.session_state.messages = []
        requests.delete(f"http://localhost:8000/session/{st.session_state.session_id}")
        st.success("Konuşma sıfırlandı!")
        st.rerun()

    st.markdown("---")

    # Yeni Özellikler
    st.markdown("### ✨ Yeni Özellikler")
    st.markdown("""
    - ✅ **BERTurk** - Türkçe için optimize
    - ✅ **Hybrid Search** - Semantic + Keyword
    - ✅ **Reranking** - Daha alakalı sonuçlar
    - ✅ **N-gram Boost** - Çift kelime eşleşmesi
    - ✅ **Fakülte Filtresi** - Daraltılmış arama
    - ✅ **Temperature kontrol** - UI'dan ayarlama
    - ✅ **Odaklı cevaplar** - Sadece mevzuattan
    """)

    st.markdown("---")
    st.info(f"🌡️ Temperature: {temperature}")

# ============================================================================
# CHAT GEÇMİŞİ
# ============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "sources" in message:
            with st.expander("📚 Kaynaklar"):
                for source in message["sources"]:
                    st.markdown(f"**{source['kaynak_metni']}** (Skor: {source['score']})")
                    st.text(source["icerik"])

# ============================================================================
# KULLANICI İNPUT
# ============================================================================

if prompt := st.chat_input("Sorunuzu yazın..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan cevabı
    with st.chat_message("assistant"):
        with st.spinner("Mevzuatları tarıyorum..."):
            try:
                # 🆕 Fakülte filtresini API'ye gönder
                fakulte_filter = None
                if fakulte_secimi != "Tümü (otomatik tespit)":
                    fakulte_filter = fakulte_secimi

                response = requests.post(
                    API_URL,
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                        "top_k": topk,
                        "temperature": temperature,
                        "fakulte_filter": fakulte_filter  # 🆕 YENİ PARAMETRE
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    st.markdown(data["answer"])

                    if data["sources"]:
                        with st.expander("📚 Kaynaklar"):
                            for source in data["sources"]:
                                st.markdown(
                                    f"**{source['kaynak_metni']}** (Skor: {source['score']}, Öncelik: {source['priority']})")
                                st.text(source["icerik"])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data["sources"]
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["answer"]
                        })
                else:
                    st.error(f"❌ Hata: {response.json()}")

            except Exception as e:
                st.error(f"❌ Bağlantı hatası: {e}")
