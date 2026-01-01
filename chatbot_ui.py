# chatbot_ui.py (Temperature Kontrolü Eklendi)
import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Mevzuat Chatbot", page_icon="📚", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎓 Teknoloji Fakültesi Mevzuat Chatbot")
st.markdown("**BERTurk + Hybrid Search** ile güçlendirilmiş mevzuat asistanı")

API_URL = "http://localhost:8000/chat"

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")

    # Temperature slider
    temperature = st.slider(
        "🌡️ Temperature (Yaratıcılık)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Düşük: Daha tutarlı, Yüksek: Daha yaratıcı"
    )

    top_k = st.slider("📚 Kaynak Sayısı", 1, 5, 3)

    if st.button("🗑️ Konuşmayı Sıfırla"):
        st.session_state.messages = []
        requests.delete(f"http://localhost:8000/session/{st.session_state.session_id}")
        st.success("Konuşma sıfırlandı!")
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ Yeni Özellikler")
    st.markdown("""
    - ✅ **BERTurk** - Türkçe için optimize
    - ✅ **Hybrid Search** - Semantic + Keyword
    - ✅ **Reranking** - Daha alakalı sonuçlar
    - ✅ **Temperature kontrolü** - UI'dan ayarlama
    - ✅ **Odaklı cevaplar** - Sadece mevzuattan
    """)

    st.markdown("---")
    st.info(f"🌡️ Temperature: {temperature}")

# Chat geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Kaynaklar"):
                for source in message["sources"]:
                    st.markdown(f"**{source['kaynak_metni']}**")
                    st.text(source["icerik"])

# Kullanıcı inputu
if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Mevzuatları tarıyorum..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                        "top_k": top_k,
                        "temperature": temperature
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    st.markdown(data["answer"])

                    if data["sources"]:
                        with st.expander("📚 Kaynaklar"):
                            for source in data["sources"]:
                                st.markdown(f"**{source['kaynak_metni']}**")
                                st.text(source["icerik"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"]
                    })
                else:
                    st.error(f"❌ Hata: {response.json()}")

            except Exception as e:
                st.error(f"❌ Bağlantı hatası: {e}")
