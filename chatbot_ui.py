# chatbot_ui.py (GÜNCELLENMİŞ VERSİYON)
import streamlit as st
import requests

st.set_page_config(page_title="Mevzuat Chatbot", page_icon="📚", layout="wide")

st.title("🎓 Teknoloji Fakültesi Mevzuat Chatbot")
st.markdown("Okulun mevzuat belgelerine göre sorularınızı yanıtlıyorum!")

API_URL = "http://localhost:8000/chat"

question = st.text_input("❓ Sorunuzu yazın:", placeholder="Örn: Sınav programı nasıl hazırlanır?")

col1, col2 = st.columns([3, 1])
with col2:
    top_k = st.slider("Kaynak Sayısı", 1, 5, 3)

if st.button("🔍 Sorgula", type="primary"):
    if question.strip():
        with st.spinner("Mevzuatlar taranıyor..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question, "top_k": top_k}
                )

                if response.status_code == 200:
                    data = response.json()

                    st.success("✅ Cevap bulundu!")

                    # Cevabı göster (Markdown formatında)
                    st.markdown("### 💬 Cevap")
                    st.markdown(data["answer"])

                    # Kaynakları detaylı göster
                    if data["sources"]:
                        st.markdown("### 📚 Detaylı Kaynaklar")
                        for source in data["sources"]:
                            relevance = (1 - source['relevance_score']) * 100

                            with st.expander(
                                    f"📄 {source['belge']} - Madde {source['madde_no']} "
                                    f"(Alakalılık: %{relevance:.0f})"
                            ):
                                st.write(source["icerik"])

                else:
                    st.error(f"❌ Hata: {response.json()}")

            except Exception as e:
                st.error(f"❌ Bağlantı hatası: {e}")
                st.info("API çalışıyor mu? `python chatbot_api.py` çalıştırın.")
    else:
        st.warning("⚠️ Lütfen bir soru yazın!")

with st.sidebar:
    st.header("ℹ️ Bilgi")
    st.markdown("""
    Bu chatbot, Teknoloji Fakültesi mevzuat belgelerini 
    RAG teknolojisi ile tarar.

    **Güncellemeler:**
    - ✅ Tam madde içerikleri gösteriliyor
    - ✅ Alakalılık skorları eklendi
    - ✅ Türkçe karakter sorunları giderildi
    """)
