from langchain.prompts import PromptTemplate

def get_prompt_template(role_param: str, language_code: str = "tr"):
    # Dil talimatını oluştur
    language_instruction = ""
    if language_code == "en":
        language_instruction = "Please provide the answer in English."
    elif language_code == "tr":
        language_instruction = "Lütfen cevabı Türkçe olarak verin."
    # Diğer diller için benzer talimatlar eklenebilir.

    template_string = f"""
Sen bir {role_param} gibi davranan bir yapay zekasın.
Kullanıcı aşağıdaki PDF belgesini yükledi:

{{context}}

Kullanıcının sorusu:
{{question}}

Lütfen bu soruyu bir {role_param} gibi, detaylı ve anlaşılır şekilde yanıtla.
{language_instruction}
Cevap verirken belge içeriğine referans vermeyi unutma.
"""
    return PromptTemplate(input_variables=["context", "question"], template=template_string)
