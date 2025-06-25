import pytesseract
import cv2
import numpy as np
import re
import os
from datetime import datetime

class SimpleOCRReader:
    def __init__(self, tesseract_path=None):
        """
        Basit ve etkili OCR okuyucu
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Tesseract yolunu otomatik bul
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"/opt/homebrew/bin/tesseract",
                r"/usr/bin/tesseract"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"✅ Tesseract: {path}")
                    break

    def create_optimized_versions(self, image_path):
        """
        Sadece en etkili 6-8 versiyon oluştur
        """
        print(f"📷 Görüntü yükleniyor: {image_path}")
        
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Görüntü bulunamadı: {image_path}")
        
        versions = {}
        
        # 1. Orijinal
        versions['original'] = original
        
        # 2. 2x büyütülmüş (en etkili)
        scaled_2x = cv2.resize(original, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        versions['scaled_2x'] = scaled_2x
        
        # 3. Gri tonlama
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        versions['grayscale'] = gray
        
        # 4. Gürültü giderme + 2x büyütme
        scaled_gray = cv2.cvtColor(scaled_2x, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(scaled_gray, h=50)
        versions['scaled_denoised'] = denoised
        
        # 5. OTSU eşikleme (gri üzerinde)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions['otsu'] = otsu
        
        # 6. OTSU eşikleme (gürültü giderilmiş üzerinde) - EN ETKİLİ
        _, otsu_denoised = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions['otsu_denoised'] = otsu_denoised
        
        # 7. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        versions['adaptive'] = adaptive
        
        # 8. CLAHE + OTSU
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        _, clahe_otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions['clahe_otsu'] = clahe_otsu
        
        print(f"✅ {len(versions)} versiyon hazırlandı")
        return versions

    def extract_text_simple(self, versions, show_images=False):
        """
        Basit metin çıkarma - sadece en etkili yöntemler
        """
        print("🔍 OCR çalışıyor...")
        
        # En etkili konfigürasyonlar
        configs = ['--psm 6', '--psm 4', '--psm 3']
        languages = ['tur', 'eng']
        
        all_texts = []
        
        for name, image in versions.items():
            if show_images:
                cv2.imshow(name, image)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            
            for lang in languages:
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(image, config=f'--oem 3 {config}', lang=lang)
                        if text.strip():
                            all_texts.append({
                                'version': name,
                                'language': lang,
                                'text': text.strip()
                            })
                    except:
                        continue
        
        print(f"✅ {len(all_texts)} metin çıkarıldı")
        return all_texts

    def find_amounts(self, text_results):
        """
        Fatura tutarlarını bul - basit ve etkili
        """
        print("💰 Tutarlar aranıyor...")
        
        # Basit ama etkili desenler
        patterns = [
            r'TUTAR[:\s]*(\d+[.,]\d+)',
            r'TOPLAM[:\s]*(\d+[.,]\d+)', 
            r'ÖDENECEK[:\s]*(\d+[.,]\d+)',
            r'(\d+[.,]\d+)\s*TL',
            r'TL[:\s]*(\d+[.,]\d+)',
            r'(\d{1,6}[.,]\d{2})'  # Genel para formatı
        ]
        
        amounts = []
        
        for result in text_results:
            text = result['text'].upper()
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    try:
                        amount = float(match.replace(',', '.'))
                        if 0.01 <= amount <= 100000:  # Makul aralık
                            amounts.append({
                                'amount': amount,
                                'version': result['version'],
                                'language': result['language'],
                                'text_preview': text[:100]
                            })
                    except:
                        continue
        
        print(f"💰 {len(amounts)} tutar bulundu")
        return amounts

    def get_best_amount(self, amounts):
        """
        En güvenilir tutarı seç
        """
        if not amounts:
            return None
        
        # Tutarları grupla
        from collections import Counter
        amount_counts = Counter([round(item['amount'], 2) for item in amounts])
        
        # En çok tespit edilen tutar
        best_amount = amount_counts.most_common(1)[0][0]
        count = amount_counts.most_common(1)[0][1]
        
        return {
            'amount': best_amount,
            'count': count,
            'total': len(amounts),
            'confidence': count / len(amounts)
        }

    def process_invoice(self, image_path, show_images=False):
        """
        Ana işlem - basit ve hızlı
        """
        print(f"🚀 İşlem başlıyor: {image_path}")
        print("=" * 50)
        
        try:
            # 1. Görüntü versiyonları (sadece 8 tane)
            versions = self.create_optimized_versions(image_path)
            
            # 2. Metin çıkar
            texts = self.extract_text_simple(versions, show_images)
            
            # 3. Tutarları bul
            amounts = self.find_amounts(texts)
            
            # 4. En iyi tutarı seç
            result = self.get_best_amount(amounts)
            
            print("=" * 50)
            if result:
                print(f"🎉 BAŞARILI!")
                print(f"💰 Tutar: {result['amount']:.2f} TL")
                print(f"🎯 Güven: {result['confidence']:.1%} ({result['count']}/{result['total']})")
                return {
                    'success': True,
                    'amount': result['amount'],
                    'confidence': result['confidence']
                }
            else:
                print("❌ Tutar bulunamadı!")
                return {'success': False, 'amount': None}
                
        except Exception as e:
            print(f"❌ Hata: {e}")
            return {'success': False, 'error': str(e)}


# BASİT KULLANIM - Orijinal kodunuz gibi
def simple_test():
    """
    Orijinal kodunuza benzer basit test
    """
    print("🔧 Basit OCR Testi")
    print("=" * 40)
    
    # Manuel tesseract path (gerekirse)
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    # Test görüntüsü
    try:
        image = cv2.imread('testocr.png')
        if image is not None:
            text = pytesseract.image_to_string(image, lang='eng')
            print("Test metin:")
            print(text if text.strip() else "Metin yok")
            print("****")
        else:
            print("testocr.png bulunamadı")
    except Exception as e:
        print(f"Test hatası: {e}")
    
    # Fatura işleme
    print("\n📄 Fatura işleniyor...")
    reader = SimpleOCRReader()
    
    # Fatura işle (görüntüleri göstermek için show_images=True)
    result = reader.process_invoice('fatura.png', show_images=False)
    
    if result['success']:
        print(f"\n✨ SONUÇ: {result['amount']:.2f} TL")
    else:
        print(f"\n❌ Başarısız")


# ÇOK FATURA TESTİ
def test_10_invoices():
    """
    10 farklı fatura testi
    """
    reader = SimpleOCRReader()
    
    print("🧪 10 Fatura Testi")
    print("=" * 40)
    
    success_count = 0
    
    for i in range(1, 11):
        filename = f'fatura{i}.png'
        if os.path.exists(filename):
            print(f"\n{i}. {filename}")
            result = reader.process_invoice(filename, show_images=False)
            if result['success']:
                success_count += 1
                print(f"   ✅ {result['amount']:.2f} TL")
            else:
                print(f"   ❌ Başarısız")
        else:
            print(f"\n{i}. {filename} - dosya yok")
    
    print(f"\n📊 Başarı: {success_count}/10")


# MANUEL İŞLEM (Orijinal kodunuz tarzı)
def manual_processing():
    """
    Orijinal kodunuzdaki gibi adım adım
    """
    print("🔧 Manuel İşlem")
    
    # Tesseract path
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    # Fatura yükle
    fatura_img = cv2.imread('fatura.png')
    if fatura_img is None:
        print("❌ fatura.png bulunamadı")
        return
    
    # 2x büyüt
    fatura_img = cv2.resize(fatura_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Gri tonlama  
    gray = cv2.cvtColor(fatura_img, cv2.COLOR_BGR2GRAY)
    
    # Gürültü giderme
    denoised = cv2.fastNlMeansDenoising(gray, h=50)
    
    # Görüntü göster
    cv2.imshow('denoised', denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Adaptive threshold
    threshold = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Görüntü göster
    cv2.imshow('threshold', threshold) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # OCR
    fatura_text = pytesseract.image_to_string(threshold, config=r'--psm 6', lang='tur')
    print("Fatura metni:")
    print(fatura_text)
    
    # Tutar ara
    import re
    amounts = re.findall(r'(\d+[.,]\d+)', fatura_text)
    print(f"\nBulunan sayılar: {amounts}")


if __name__ == "__main__":
    # Basit test
    simple_test()
    
    # Manuel işlem testi (isterseniz)
    # manual_processing()
    
    # 10 fatura testi (isterseniz)
    choice = input("\n10 fatura testi yapmak ister misiniz? (y/n): ")
    if choice.lower() in ['y', 'yes', 'e', 'evet']:
        test_10_invoices()