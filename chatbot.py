from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from prompts import get_prompt_template

def get_qa_chain(vectorstore, role, language_code="tr"): # language_code parametresi eklendi
    llm = OllamaLLM(model="qwen2.5:latest")
    prompt = get_prompt_template(role, language_code) # language_code prompt'a iletildi
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # Kaynak belgeleri döndürmeyi etkinleştir
    )
    return qa_chain

def refine_answer(original_question, original_answer, refinement_type, role, language_code="tr"): # language_code parametresi eklendi
    """
    Verilen bir cevabı detaylandırır veya sadeleştirir.
    original_question: Kullanıcının ilk sorduğu soru.
    original_answer: LLM'in verdiği ilk cevap.
    refinement_type: "detaylandır" veya "sadeleştir".
    role: Mevcut aktif rol.
    """
    llm = OllamaLLM(model="qwen2.5:latest")
    
    # İyileştirme türüne göre prompt oluştur
    if refinement_type == "detaylandır":
        refinement_instruction = "Yukarıdaki cevabı daha detaylı bir şekilde, teknik terimler ve açıklamalar ekleyerek genişlet."
    elif refinement_type == "sadeleştir":
        refinement_instruction = "Yukarıdaki cevabı daha basit bir dille, herkesin anlayabileceği şekilde yeniden ifade et."
    else:
        return "Geçersiz iyileştirme türü."

    # Rol bilgisini de prompt'a dahil edebiliriz, böylece iyileştirme rolün tonuna uygun olur.
    
    # Dil talimatını ekle
    language_instruction = ""
    if language_code == "en":
        language_instruction = "Provide the refined answer in English."
    elif language_code == "tr":
        language_instruction = "Düzenlenmiş cevabı Türkçe olarak verin."
    # Diğer diller için benzer talimatlar eklenebilir.

    prompt_text = f"""Bir '{role}' rolündesiniz.
Kullanıcının sorusu: "{original_question}"
Verilen ilk cevap: "{original_answer}"

GÖREV: {refinement_instruction}
{language_instruction}
Sadece düzenlenmiş cevabı verin.
"""

    # LangChain'in LLMChain'ini veya doğrudan llm.invoke'u kullanabiliriz.
    # Burada basitlik için doğrudan invoke kullanalım.
    try:
        refined_response = llm.invoke(prompt_text)
        # OllamaLLM doğrudan string döndürür, eğer bir sözlük dönerse response['result'] gibi erişmek gerekebilir.
        # Modelin çıktısına göre bu kısmı ayarlamak gerekebilir. Genellikle string döner.
        return refined_response 
    except Exception as e:
        print(f"Cevap iyileştirilirken hata oluştu: {e}")
        return "Cevap iyileştirilirken bir sorun oluştu."

def generate_suggested_questions(document_chunks, role, language_code="tr", num_questions=3):
    """
    Yüklenen belgelere dayanarak örnek sorular üretir.
    document_chunks: LangChain Document nesnelerinin listesi.
    role: Mevcut aktif rol (soruların role uygun olması için).
    language_code: Soruların üretileceği dil.
    num_questions: Üretilecek soru sayısı.
    """
    llm = OllamaLLM(model="qwen2.5:latest")

    # LLM'e verilecek bağlamı oluştur (örneğin ilk birkaç chunk'ın birleşimi)
    # Çok fazla chunk vermek token limitini aşabilir, bu yüzden dikkatli olmalıyız.
    # Ya da chunk'lardan rastgele örnekler alabiliriz.
    # Şimdilik ilk N karakteri alalım (örneğin ilk 5 chunk'tan).
    context_text = ""
    char_limit = 2000 # LLM'e gönderilecek maksimum karakter sayısı (yaklaşık)
    for chunk_doc in document_chunks:
        if len(context_text) + len(chunk_doc.page_content) < char_limit:
            context_text += chunk_doc.page_content + "\n\n"
        else:
            break
    
    if not context_text: # Eğer hiç chunk yoksa veya hepsi çok uzunsa
        context_text = "Belge içeriği hakkında genel sorular."


    # Dil talimatını oluştur
    question_language_instruction = ""
    if language_code == "en":
        question_language_instruction = f"Generate {num_questions} insightful questions about the following content, in English. The user will be interacting as a '{role}'."
        output_format_instruction = "Provide only the questions, each on a new line. Do not number them or add any other text."
    elif language_code == "tr":
        question_language_instruction = f"Aşağıdaki içerik hakkında, bir '{role}' rolündeki kullanıcının sorabileceği {num_questions} adet düşündürücü soru üret, Türkçe olarak."
        output_format_instruction = "Sadece soruları, her biri yeni bir satırda olacak şekilde ver. Numaralandırma veya başka bir metin ekleme."
    
    prompt_text = f"""{question_language_instruction}

İçerik Özeti:
---
{context_text.strip()}
---

{output_format_instruction}
"""

    try:
        response = llm.invoke(prompt_text)
        # Yanıtın doğrudan soruları satır satır içerdiğini varsayıyoruz.
        suggested_questions = [q.strip() for q in response.split('\n') if q.strip()]
        return suggested_questions[:num_questions] # Fazla soru üretilirse kırp
    except Exception as e:
        print(f"Soru önerileri üretilirken hata: {e}")
        return []

def summarize_documents(document_chunks, role, language_code="tr"):
    """
    Yüklenen belgelerin tamamından bir özet üretir.
    document_chunks: LangChain Document nesnelerinin listesi.
    role: Mevcut aktif rol (özetin role uygun olması için).
    language_code: Özetin üretileceği dil.
    """
    llm = OllamaLLM(model="qwen2.5:latest")

    # Tüm chunk'ların metinlerini birleştir. Token limitine dikkat et!
    # Eğer metin çok uzunsa, ya sadece belirli bir kısmını özetle ya da
    # haritalama-azaltma (map-reduce) gibi daha karmaşık bir özetleme stratejisi kullan.
    # Şimdilik basit bir birleştirme ve karakter limiti uygulayalım.
    full_text = ""
    char_limit_for_summary = 10000 # Özetleme için LLM'e gönderilecek maks karakter (ayarlanabilir)
    
    for chunk_doc in document_chunks:
        if len(full_text) + len(chunk_doc.page_content) < char_limit_for_summary:
            full_text += chunk_doc.page_content + "\n\n"
        else:
            # Eğer limiti aşıyorsak, metnin sonuna bir not ekleyebiliriz.
            full_text += "\n\n[İçeriğin bir kısmı token limiti nedeniyle kesilmiştir.]"
            break
    
    if not full_text.strip():
        return "Özetlenecek içerik bulunamadı."

    # Dil talimatını oluştur
    summary_language_instruction = ""
    role_instruction = f"Bir '{role}' olarak davranıyorsun."
    task_instruction = "Aşağıdaki metnin kapsamlı bir özetini çıkar."

    if language_code == "en":
        summary_language_instruction = "Provide the summary in English."
        role_instruction = f"You are acting as a '{role}'."
        task_instruction = "Generate a comprehensive summary of the following text."
    elif language_code == "tr":
        summary_language_instruction = "Özeti Türkçe olarak verin."
        # role_instruction ve task_instruction zaten Türkçe.
    
    prompt_text = f"""{role_instruction}
{task_instruction}
{summary_language_instruction}

Metin:
---
{full_text.strip()}
---

Lütfen yukarıdaki metnin ana noktalarını içeren, iyi yapılandırılmış bir özet sunun.
"""

    try:
        response = llm.invoke(prompt_text)
        return response # Yanıtın doğrudan özeti içerdiğini varsayıyoruz.
    except Exception as e:
        print(f"Belge özeti üretilirken hata: {e}")
        return "Belge özeti üretilirken bir sorun oluştu."

def extract_keywords_from_documents(document_chunks, role, language_code="tr", num_keywords=10):
    """
    Yüklenen belgelerden anahtar kelimeleri çıkarır.
    document_chunks: LangChain Document nesnelerinin listesi.
    role: Mevcut aktif rol.
    language_code: Anahtar kelimelerin çıkarılacağı ve listeleneceği dil.
    num_keywords: Çıkarılacak maksimum anahtar kelime sayısı.
    """
    llm = OllamaLLM(model="qwen2.5:latest")

    full_text = ""
    # Özetleme için kullandığımız karakter limitini burada da kullanabiliriz.
    # Daha kısa bir metin anahtar kelime çıkarımı için yeterli olabilir.
    char_limit_for_keywords = 5000 
    for chunk_doc in document_chunks:
        if len(full_text) + len(chunk_doc.page_content) < char_limit_for_keywords:
            full_text += chunk_doc.page_content + "\n\n"
        else:
            break
    
    if not full_text.strip():
        return []

    # Dil ve rol talimatları
    role_instruction = f"Bir '{role}' olarak davranıyorsun."
    task_instruction = f"Aşağıdaki metinden en önemli {num_keywords} anahtar kelimeyi veya kavramı çıkar."
    output_format_instruction = "Anahtar kelimeleri virgülle ayırarak tek bir satırda listele. Başka hiçbir şey ekleme."
    language_preference = "Türkçe"

    if language_code == "en":
        role_instruction = f"You are acting as a '{role}'."
        task_instruction = f"Extract the top {num_keywords} keywords or concepts from the following text."
        output_format_instruction = "List the keywords as a single comma-separated string. Add nothing else."
        language_preference = "English"
    
    prompt_text = f"""{role_instruction}
{task_instruction}
Dil Tercihi: {language_preference}.

Metin:
---
{full_text.strip()}
---

{output_format_instruction}
"""

    try:
        response = llm.invoke(prompt_text)
        # Yanıtın "keyword1, keyword2, keyword3" formatında olduğunu varsayıyoruz.
        keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
        return keywords[:num_keywords]
    except Exception as e:
        print(f"Anahtar kelime çıkarılırken hata: {e}")
        return []

def generate_concept_map_data(document_chunks, role, language_code="tr"):
    """
    Yüklenen belgelerden metin tabanlı bir konsept haritası (Mermaid.js formatında) üretir.
    """
    llm = OllamaLLM(model="qwen2.5:latest")

    # Anahtar kelime çıkarmada olduğu gibi metnin bir kısmını alalım
    full_text = ""
    char_limit_for_map = 7000 # Konsept haritası için biraz daha fazla metin gerekebilir
    for chunk_doc in document_chunks:
        if len(full_text) + len(chunk_doc.page_content) < char_limit_for_map:
            full_text += chunk_doc.page_content + "\n\n"
        else:
            break
    
    if not full_text.strip():
        return "Konsept haritası için içerik bulunamadı."

    # Dil ve rol talimatları
    role_instruction = f"Bir '{role}' olarak, aşağıdaki metindeki ana kavramları ve aralarındaki ilişkileri analiz et."
    task_instruction = "Bu analize dayanarak, metin tabanlı bir konsept haritası oluştur. Çıktıyı Mermaid.js 'graph TD' veya 'graph LR' formatında ver. Harita, ana fikirleri ve bunların alt başlıklarını hiyerarşik bir şekilde göstermelidir."
    output_example = """
Örnek Çıktı Formatı (Mermaid.js graph TD):
```mermaid
graph TD
    A[Ana Kavram] --> B(Alt Kavram 1)
    A --> C(Alt Kavram 2)
    B --> D{Detay 1.1}
    B --> E{Detay 1.2}
    C --> F{Detay 2.1}
```
"""
    language_preference = "Türkçe"

    if language_code == "en":
        role_instruction = f"As a '{role}', analyze the main concepts and their relationships in the following text."
        task_instruction = "Based on this analysis, create a text-based concept map. Provide the output in Mermaid.js 'graph TD' or 'graph LR' format. The map should hierarchically show the main ideas and their sub-topics."
        language_preference = "English (for node labels if possible, but structure is key)"
    
    prompt_text = f"""{role_instruction}
{task_instruction}
Dil Tercihi (kavram etiketleri için mümkünse): {language_preference}.

Metin:
---
{full_text.strip()}
---

{output_example}
Lütfen SADECE Mermaid.js kod bloğunu (` ```mermaid ... ``` `) yanıt olarak ver. Başka hiçbir açıklama veya metin ekleme.
"""

    try:
        response = llm.invoke(prompt_text)
        # Modelin doğrudan ```mermaid ... ``` bloğunu döndürdüğünü varsayıyoruz.
        # Eğer değilse, bu bloğu ayıklamak için ek işlem gerekebilir.
        if "```mermaid" in response and "```" in response.split("```mermaid")[1]:
            mermaid_code = "```mermaid" + response.split("```mermaid")[1].split("```")[0] + "```"
            return mermaid_code.strip()
        else: # Basit bir fallback veya hata
            print(f"Model beklenen Mermaid formatında yanıt vermedi: {response}")
            # Belki de sadece metin tabanlı bir hiyerarşi istemek daha güvenli olabilir.
            # Şimdilik bu şekilde bırakalım.
            return "Konsept haritası üretilemedi (beklenen formatta değil)." 
            
    except Exception as e:
        print(f"Konsept haritası üretilirken hata: {e}")
        return "Konsept haritası üretilirken bir sorun oluştu."

def extract_timeline_from_documents(document_chunks, role, language_code="tr"):
    """
    Yüklenen belgelerden tarihsel olayları çıkarıp bir zaman çizelgesi oluşturur.
    """
    llm = OllamaLLM(model="qwen2.5:latest")

    # Metnin tamamını veya önemli bir kısmını alalım
    full_text = ""
    char_limit_for_timeline = 8000 # Zaman çizelgesi için daha fazla bağlam gerekebilir
    for chunk_doc in document_chunks:
        if len(full_text) + len(chunk_doc.page_content) < char_limit_for_timeline:
            full_text += chunk_doc.page_content + "\n\n"
        else:
            break
    
    if not full_text.strip():
        return "Zaman çizelgesi için içerik bulunamadı."

    # Dil ve rol talimatları
    role_instruction = f"Bir '{role}' olarak, aşağıdaki metindeki tarihleri ve bu tarihlerle ilişkili önemli olayları veya bilgileri analiz et."
    task_instruction = "Bu analize dayanarak, kronolojik olarak sıralanmış bir zaman çizelgesi oluştur. Her bir maddeyi 'Tarih: Açıklama' formatında listele."
    language_preference = "Türkçe"

    if language_code == "en":
        role_instruction = f"As a '{role}', analyze the following text to identify dates and the significant events or information associated with them."
        task_instruction = "Based on this analysis, create a chronologically ordered timeline. List each item in the format 'Date: Description'."
        language_preference = "English"
    
    prompt_text = f"""{role_instruction}
{task_instruction}
Dil Tercihi: {language_preference}.

Metin:
---
{full_text.strip()}
---

Lütfen bulunan olayları kronolojik sıraya göre, her biri yeni bir satırda olacak şekilde listele. Eğer metinde belirgin tarihler yoksa, "Belgede belirgin bir zaman çizelgesi bulunamadı." yanıtını ver.
"""

    try:
        response = llm.invoke(prompt_text)
        # Yanıtın doğrudan markdown listesi veya ilgili mesaj olduğunu varsayıyoruz.
        return response 
    except Exception as e:
        print(f"Zaman çizelgesi çıkarılırken hata: {e}")
        return "Zaman çizelgesi çıkarılırken bir sorun oluştu."
