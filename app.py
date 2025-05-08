import streamlit as st
import os
from pdf_handler import extract_pages_from_pdf, chunk_pages, get_pdf_page_image_bytes
import fitz
from embedder import embed_and_store, load_vectorstore
from chatbot import get_qa_chain, generate_suggested_questions, summarize_documents, extract_keywords_from_documents, generate_concept_map_data, extract_timeline_from_documents
import json
import pandas as pd # Grafik için Pandas ekleyelim
from collections import defaultdict # Chunk sayısını saymak için

st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Session state'i başlat
if 'pdf_previews' not in st.session_state:
    st.session_state.pdf_previews = {}
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = ""
if 'refined_answer' not in st.session_state:
    st.session_state.refined_answer = ""
if 'source_documents' not in st.session_state:
    st.session_state.source_documents = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []
if 'current_question_input' not in st.session_state:
    st.session_state.current_question_input = ""
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = ""
if 'extracted_keywords' not in st.session_state:
    st.session_state.extracted_keywords = []
if 'concept_map_data' not in st.session_state:
    st.session_state.concept_map_data = ""
if 'timeline_data' not in st.session_state:
    st.session_state.timeline_data = ""
if 'page_chunk_counts' not in st.session_state: # Bilgi yoğunluğu için session state
    st.session_state.page_chunk_counts = {}


st.title("📄 PDF Destekli Rol-Tabanlı Chatbot")

st.info(
    "🔒 **Veri Gizliliği ve Güvenlik:** Bu uygulama tamamen yerel makinenizde çalışır. "
    "Yüklediğiniz PDF'ler ve sorduğunuz sorular harici bir sunucuya gönderilmez. "
    "Tüm işlemler (metin çıkarma, embedding, cevap üretme) Ollama ve yerel modeliniz (Qwen2.5) aracılığıyla bilgisayarınızda gerçekleştirilir."
)
st.markdown("---")

# Rol seçimi
roles_data = json.load(open("roles.json", "r", encoding="utf-8"))
available_roles = roles_data
selected_role_from_list = st.selectbox("🧑 Hazır Rollerden Seç", [""] + available_roles, index=0, help="Bir rol seçin veya aşağıya kendi rolünüzü yazın.")

custom_role_input = st.text_area("📝 Veya Kendi Rolünü Yaz (isteğe bağlı)", placeholder="Örn: 'Belgelerdeki finansal riskleri analiz eden bir finans uzmanı.'")

# Nihai rolü belirle
final_selected_role = custom_role_input.strip() if custom_role_input.strip() else selected_role_from_list

if not final_selected_role:
    st.warning("⚠️ Lütfen bir rol seçin veya kendi rolünüzü yazın.")
    st.stop()

st.info(f"🤖 Aktif Rol: {final_selected_role}")

# Cevap dili seçimi
available_languages = {"Türkçe": "tr", "English": "en"}
selected_language_label = st.selectbox("🌐 Cevap Dili Seç", list(available_languages.keys()))
selected_language_code = available_languages[selected_language_label]


# PDF yükleme
uploaded_files = st.file_uploader("📤 PDF Yükle (Birden fazla dosya seçebilirsiniz)", type=["pdf"], accept_multiple_files=True)
processed_pdf_paths = []
all_chunks_for_session = [] # Bu session'daki tüm chunk'ları saklamak için

if uploaded_files:
    # Yeni yükleme olduğunda bazı session state'leri sıfırla
    st.session_state.pdf_previews = {}
    st.session_state.suggested_questions = []
    st.session_state.document_summary = ""
    st.session_state.extracted_keywords = []
    st.session_state.concept_map_data = ""
    st.session_state.timeline_data = ""
    st.session_state.page_chunk_counts = {} # Bilgi yoğunluğunu da sıfırla
    st.session_state.last_answer = ""
    st.session_state.refined_answer = ""
    st.session_state.source_documents = []

    current_file_chunks = []
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        processed_pdf_paths.append(file_path)

        try:
            doc = fitz.open(file_path)
            st.session_state.pdf_previews[uploaded_file.name] = {
                "total_pages": doc.page_count,
                "current_page_display": 1,
                "path": file_path
            }
            doc.close()
        except Exception as e:
            st.error(f"{uploaded_file.name} için sayfa sayısı alınamadı: {e}")
            continue

        pages_data = extract_pages_from_pdf(file_path)
        if pages_data:
            chunks_from_file = chunk_pages(pages_data)
            current_file_chunks.extend(chunks_from_file)
            st.write(f"📄 {uploaded_file.name} ({st.session_state.pdf_previews.get(uploaded_file.name, {}).get('total_pages', 'N/A')} sayfa) işlendi ({len(chunks_from_file)} chunk).")
        else:
            st.write(f"⚠️ {uploaded_file.name} dosyasından metin çıkarılamadı veya dosya boş.")

    all_chunks_for_session = current_file_chunks

    if all_chunks_for_session:
        # Bilgi yoğunluğu hesaplaması
        page_counts = defaultdict(lambda: defaultdict(int))
        for chunk in all_chunks_for_session:
            source = chunk.metadata.get("source", "Bilinmeyen Kaynak")
            page = chunk.metadata.get("page", 0) # Sayfa no yoksa 0 varsayalım
            if page > 0: # Geçerli sayfa numarası varsa say
                page_counts[source][page] += 1
        st.session_state.page_chunk_counts = {k: dict(v) for k, v in page_counts.items()} # defaultdict'u dict'e çevir

        # Embedding ve diğer işlemler
        if not os.path.exists("vectordb"):
            os.makedirs("vectordb")
        embed_and_store(all_chunks_for_session)
        st.success("✅ Tüm PDF'ler işlendi ve veritabanı oluşturuldu/güncellendi!")

        with st.spinner("🤔 Örnek sorular ve anahtar kelimeler hazırlanıyor..."):
            st.session_state.suggested_questions = generate_suggested_questions(all_chunks_for_session, final_selected_role, selected_language_code, num_questions=3)
            st.session_state.extracted_keywords = extract_keywords_from_documents(all_chunks_for_session, final_selected_role, selected_language_code, num_keywords=10)
    else:
        st.warning("⚠️ Yüklenen PDF'lerden metin çıkarılamadı veya PDF'ler boş.")
        st.session_state.suggested_questions = []
        st.session_state.extracted_keywords = []
        st.session_state.concept_map_data = ""
        st.session_state.timeline_data = ""
        st.session_state.page_chunk_counts = {}


# PDF Önizleme Alanı
if st.session_state.pdf_previews:
    st.markdown("---")
    st.subheader("📂 Yüklenen PDF'ler ve Önizleme")
    for pdf_name, preview_data in st.session_state.pdf_previews.items():
        with st.expander(f"{pdf_name} ({preview_data['total_pages']} sayfa)"):
            if preview_data['total_pages'] > 0:
                page_to_show_user = st.number_input(
                    f"Sayfa Numarası (1-{preview_data['total_pages']})",
                    min_value=1,
                    max_value=preview_data['total_pages'],
                    value=preview_data['current_page_display'],
                    key=f"preview_page_num_{pdf_name}"
                )
                st.session_state.pdf_previews[pdf_name]['current_page_display'] = page_to_show_user
                page_num_fitz = page_to_show_user - 1

                texts_to_highlight_on_page = []
                if st.session_state.source_documents:
                    for src_doc in st.session_state.source_documents:
                        if src_doc.metadata.get('source') == pdf_name and \
                           src_doc.metadata.get('page') == page_to_show_user:
                            texts_to_highlight_on_page.append(src_doc.page_content)

                img_bytes = get_pdf_page_image_bytes(preview_data['path'], page_num_fitz, texts_to_highlight_on_page)
                if img_bytes:
                    st.image(img_bytes, caption=f"{pdf_name} - Sayfa {page_to_show_user}{' (vurgulandı)' if texts_to_highlight_on_page else ''}", use_column_width=True)
                else:
                    st.warning(f"{pdf_name} - Sayfa {page_to_show_user} için önizleme oluşturulamadı.")
            else:
                st.info(f"{pdf_name} içeriği boş veya okunamadı.")

# Bilgi Yoğunluğu Analizi Alanı
if st.session_state.page_chunk_counts:
    st.markdown("---")
    st.subheader("📊 Bilgi Yoğunluğu (Sayfa Başına Chunk Sayısı)")
    for pdf_name, counts in st.session_state.page_chunk_counts.items():
        with st.expander(f"{pdf_name} - Yoğunluk Grafiği"):
            if counts:
                # Grafiği çizmek için veriyi hazırla (sayfa numarasına göre sıralı)
                sorted_counts = dict(sorted(counts.items()))
                # Pandas DataFrame oluşturmak daha sağlam olabilir ama dict ile deneyelim
                # st.bar_chart(pd.DataFrame.from_dict(sorted_counts, orient='index', columns=['Chunk Sayısı']))
                st.bar_chart(sorted_counts)
            else:
                st.info(f"{pdf_name} için yoğunluk verisi hesaplanamadı.")


# Soru alanı ve diğer işlemler (Sadece PDF'ler yüklendi ve işlendiyse göster)
if uploaded_files and all_chunks_for_session:
    # Anahtar Kelimeler (Kenar Çubuğunda)
    if st.session_state.extracted_keywords:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔑 Anahtar Kelimeler")
        for kw in st.session_state.extracted_keywords:
            st.sidebar.caption(kw)

    # Örnek Sorular
    if st.session_state.suggested_questions:
        st.markdown("---")
        st.subheader("💡 Örnek Sorular")
        num_suggestion_cols = min(len(st.session_state.suggested_questions), 3)
        if num_suggestion_cols > 0:
            cols = st.columns(num_suggestion_cols)
            for i, sq in enumerate(st.session_state.suggested_questions):
                with cols[i % num_suggestion_cols]:
                    if st.button(sq, key=f"suggested_q_{i}", use_container_width=True):
                        st.session_state.current_question_input = sq
                        st.experimental_rerun()

    # Soru Giriş Alanı
    question = st.text_input("❓ Soru Sor", value=st.session_state.current_question_input, key="main_question_input_field")
    if st.session_state.main_question_input_field != st.session_state.current_question_input:
        st.session_state.current_question_input = st.session_state.main_question_input_field
        st.experimental_rerun()

    # Aksiyon Butonları
    action_cols = st.columns(4)
    with action_cols[0]: # Yanıtla Butonu
        if st.button("💬 Yanıtla", use_container_width=True) and st.session_state.current_question_input:
            st.session_state.last_question = st.session_state.current_question_input
            st.session_state.refined_answer = ""
            st.session_state.document_summary = ""
            st.session_state.concept_map_data = ""
            st.session_state.timeline_data = ""
            vectorstore = load_vectorstore()
            if vectorstore:
                with st.spinner("Yanıt hazırlanıyor..."):
                    qa_chain = get_qa_chain(vectorstore, final_selected_role, selected_language_code)
                    input_data = {"query": st.session_state.current_question_input}
                    response_data = qa_chain.invoke(input_data)
                    current_answer = response_data["result"]
                    current_sources = response_data.get("source_documents", [])
                    st.session_state.last_answer = current_answer
                    st.session_state.source_documents = current_sources
                    st.session_state.conversation_history.append({
                        "question": st.session_state.current_question_input, "answer": current_answer, "sources": current_sources,
                        "role": final_selected_role, "language": selected_language_label, "refined_answer": ""
                    })
            else:
                st.error("❌ Vektör veritabanı yüklenemedi. Lütfen PDF yükleyip işleyin.")
                st.session_state.last_answer = ""
                st.session_state.source_documents = []

    with action_cols[1]: # Özetle Butonu
        if st.button("🧮 Özetle", use_container_width=True):
            st.session_state.last_answer = ""
            st.session_state.refined_answer = ""
            st.session_state.source_documents = []
            st.session_state.concept_map_data = ""
            st.session_state.timeline_data = ""
            if all_chunks_for_session:
                with st.spinner("📚 Belgeler özetleniyor... Bu işlem biraz zaman alabilir."):
                    summary = summarize_documents(all_chunks_for_session, final_selected_role, selected_language_code)
                    st.session_state.document_summary = summary
            else:
                st.warning("⚠️ Özetlenecek belge bulunamadı. Lütfen önce PDF yükleyin.")
                st.session_state.document_summary = ""

    with action_cols[2]: # Konsept Haritası Butonu
        if st.button("🧠 Konsept Haritası", use_container_width=True):
            st.session_state.last_answer = ""
            st.session_state.refined_answer = ""
            st.session_state.source_documents = []
            st.session_state.document_summary = ""
            st.session_state.timeline_data = ""
            if all_chunks_for_session:
                with st.spinner("🗺️ Konsept haritası oluşturuluyor..."):
                    map_data = generate_concept_map_data(all_chunks_for_session, final_selected_role, selected_language_code)
                    st.session_state.concept_map_data = map_data
            else:
                st.warning("⚠️ Konsept haritası için belge bulunamadı. Lütfen önce PDF yükleyin.")
                st.session_state.concept_map_data = ""

    with action_cols[3]: # Zaman Çizelgesi Butonu
        if st.button("⏳ Zaman Çizelgesi", use_container_width=True):
            st.session_state.last_answer = ""
            st.session_state.refined_answer = ""
            st.session_state.source_documents = []
            st.session_state.document_summary = ""
            st.session_state.concept_map_data = ""
            if all_chunks_for_session:
                with st.spinner("📅 Zaman çizelgesi çıkarılıyor..."):
                    timeline = extract_timeline_from_documents(all_chunks_for_session, final_selected_role, selected_language_code)
                    st.session_state.timeline_data = timeline
            else:
                st.warning("⚠️ Zaman çizelgesi için belge bulunamadı. Lütfen önce PDF yükleyin.")
                st.session_state.timeline_data = ""


    # Çıktı Alanları
    st.markdown("---") # Ayırıcı

    # Belge Özeti Gösterimi
    if st.session_state.document_summary:
        st.markdown("### 📜 Belge Özeti")
        st.write(st.session_state.document_summary)

    # Konsept Haritası Gösterimi
    if st.session_state.concept_map_data:
        st.markdown("### 🗺️ Konsept Haritası")
        if "```mermaid" in st.session_state.concept_map_data:
            st.markdown(st.session_state.concept_map_data)
        else:
            st.warning("Konsept haritası görselleştirilemedi. Ham veri:")
            st.code(st.session_state.concept_map_data)

    # Zaman Çizelgesi Gösterimi
    if st.session_state.timeline_data:
        st.markdown("### 📅 Zaman Çizelgesi")
        st.markdown(st.session_state.timeline_data)


    # Cevap ve İlgili İşlemler Gösterimi
    if st.session_state.last_answer:
        st.markdown("### 💡 Güncel Cevap")
        st.write(st.session_state.last_answer)

        col_refine1, col_refine2 = st.columns(2)
        with col_refine1:
            if st.button("🔁 Cevabı Detaylandır"):
                from chatbot import refine_answer
                if st.session_state.last_question and st.session_state.last_answer:
                    with st.spinner("Cevap detaylandırılıyor..."):
                        refined_text = refine_answer(st.session_state.last_question, st.session_state.last_answer, "detaylandır", final_selected_role, selected_language_code)
                        st.session_state.refined_answer = refined_text
                        if st.session_state.conversation_history:
                            st.session_state.conversation_history[-1]["refined_answer"] = refined_text
                else:
                    st.warning("Detaylandırmak için önce bir cevap alınmalı.")
        with col_refine2:
            if st.button("🔀 Cevabı Sadeleştir"):
                from chatbot import refine_answer
                if st.session_state.last_question and st.session_state.last_answer:
                     with st.spinner("Cevap sadeleştiriliyor..."):
                        refined_text = refine_answer(st.session_state.last_question, st.session_state.last_answer, "sadeleştir", final_selected_role, selected_language_code)
                        st.session_state.refined_answer = refined_text
                        if st.session_state.conversation_history:
                            st.session_state.conversation_history[-1]["refined_answer"] = refined_text
                else:
                    st.warning("Sadeleştirmek için önce bir cevap alınmalı.")

        if st.session_state.refined_answer:
            st.markdown("#### ✨ Güncel Düzenlenmiş Cevap:")
            st.write(st.session_state.refined_answer)

        if st.session_state.source_documents:
            st.markdown("📚 **Referanslar:**")
            references = set()
            for doc in st.session_state.source_documents:
                source_name = doc.metadata.get("source", "Bilinmeyen Kaynak")
                page_number = doc.metadata.get("page", "Bilinmeyen Sayfa")
                references.add(f"- {source_name} (Sayfa: {page_number})")
            for ref in sorted(list(references)):
                st.markdown(ref)

# Konuşma Geçmişi (Kenar Çubuğunda)
st.sidebar.title("📜 Konuşma Geçmişi")
if not st.session_state.conversation_history:
    st.sidebar.info("Henüz bir konuşma geçmişi yok.")
else:
    for i, entry in enumerate(reversed(st.session_state.conversation_history)):
        with st.sidebar.expander(f"Soru {len(st.session_state.conversation_history) - i}: {entry['question'][:30]}..."):
            st.markdown(f"**Soru:** {entry['question']}")
            st.markdown(f"**Rol:** {entry['role']}")
            st.markdown(f"**Dil:** {entry['language']}")
            st.markdown("**Cevap:**")
            st.write(entry['answer'])
            if entry.get('refined_answer'):
                st.markdown("**Düzenlenmiş Cevap:**")
                st.write(entry['refined_answer'])
            if entry['sources']:
                st.markdown("**Referanslar:**")
                current_references = set()
                for doc_ref in entry['sources']:
                    source_name_ref = doc_ref.metadata.get("source", "Bilinmeyen Kaynak")
                    page_number_ref = doc_ref.metadata.get("page", "Bilinmeyen Sayfa")
                    current_references.add(f"- {source_name_ref} (Sayfa: {page_number_ref})")
                for r_ref in sorted(list(current_references)):
                    st.markdown(r_ref)
