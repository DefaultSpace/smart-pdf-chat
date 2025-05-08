import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from io import BytesIO # BytesIO'yu ekliyoruz

def extract_pages_from_pdf(pdf_path):
    """PDF'ten sayfaları metin ve sayfa numarası olarak çıkarır."""
    doc = fitz.open(pdf_path)
    pages_data = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip(): # Sadece metin içeren sayfaları ekle
            pages_data.append({"page_content": text, "metadata": {"source": os.path.basename(pdf_path), "page": page_num + 1}})
    return pages_data

def chunk_pages(pages_data_list):
    """
    Sayfa verilerini alır, metinleri chunk'lar ve LangChain Document nesneleri oluşturur.
    pages_data_list: extract_pages_from_pdf'ten dönen [{page_content: str, metadata: dict}, ...] listesi.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    all_chunks = []
    for page_data in pages_data_list:
        # Sayfa metnini chunk'la
        chunks_from_page = text_splitter.split_text(page_data["page_content"])
        
        # Her chunk için Document nesnesi oluştur ve metadata'yı koru/güncelle
        for chunk_content in chunks_from_page:
            # Orijinal metadata'yı kopyala ve chunk'ın içeriğini ekle
            # Eğer chunk'a özel bir metadata eklemek isterseniz burada yapabilirsiniz.
            # Şimdilik sayfa bazlı metadata yeterli.
            doc = Document(page_content=chunk_content, metadata=page_data["metadata"].copy())
            all_chunks.append(doc)
            
    return all_chunks

def get_pdf_page_image_bytes(pdf_path, page_number, highlight_texts=None):
    """
    Verilen PDF dosyasının belirtilen sayfasının PNG görüntüsünü byte olarak döndürür.
    Sayfa numaraları 0'dan başlar (fitz için).
    highlight_texts: Vurgulanacak metinlerin listesi. Örn: ["metin1", "metin2"]
    """
    doc = None # doc değişkenini try bloğundan önce tanımla
    try:
        doc = fitz.open(pdf_path)
        if not (0 <= page_number < len(doc)):
            if doc: doc.close()
            return None

        page = doc.load_page(page_number)
        
        if highlight_texts:
            for text_to_highlight in highlight_texts:
                # search_for metodu, metnin geçtiği yerlerin koordinatlarını (Rect listesi) döndürür.
                # Her bir Rect için bir highlight annotasyonu ekleyebiliriz.
                # Not: Çok uzun veya çok sık geçen metinler için performans etkilenebilir.
                # Daha hassas eşleştirme için text_splitter'ın chunk'larını doğrudan kullanmak ve
                # bu chunk'ların orijinal PDF'teki yerlerini daha kesin belirlemek gerekebilir.
                # Şimdilik basit metin araması yapıyoruz.
                try:
                    # quads=True daha kesin sonuçlar verebilir ama daha yavaş olabilir.
                    # text_instances = page.search_for(text_to_highlight, hit_max=10) # Çok fazla eşleşmeyi önlemek için hit_max
                    text_instances = page.search_for(text_to_highlight)
                    if text_instances:
                        for inst in text_instances:
                            highlight = page.add_highlight_annot(inst)
                            # Vurgu rengini de ayarlayabiliriz:
                            # highlight.set_colors(stroke=fitz.utils.getColor("yellow"), fill=fitz.utils.getColor("yellow"))
                            # highlight.update() # Değişiklikleri uygula
                except Exception as search_error:
                    print(f"Metin aranırken hata '{text_to_highlight}': {search_error}")


        pix = page.get_pixmap()
        img_bytes = BytesIO()
        pix.save(img_bytes, "png")
        img_bytes.seek(0)
        doc.close()
        return img_bytes.getvalue()
        
    except Exception as e:
        print(f"Sayfa görüntüsü ({pdf_path}, sayfa {page_number}) alınırken hata: {e}")
        if doc:
            doc.close()
        return None
    except Exception as e:
        print(f"Sayfa görüntüsü alınırken hata: {e}")
        if 'doc' in locals() and doc:
            doc.close()
        return None
    # finally bloğu kaldırıldı çünkü doc.close() artık try/except içinde yönetiliyor.
