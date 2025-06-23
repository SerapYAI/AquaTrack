import cv2
import numpy as np

# 1) Orijinal görüntüyü yükle
img = cv2.imread('mersinyavrusu.png')
h, w = img.shape[:2]

# 2) GrabCut ile kaba ön plan maskesini elde et
mask = np.zeros((h, w), np.uint8)
bgd = np.zeros((1,65), np.float64)
fgd = np.zeros((1,65), np.float64)
rect = (int(w*0.02), int(h*0.02), int(w*0.96), int(h*0.96))
cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')

# 3) Morfolojik kapanışla deliği kapat
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)

# 4) Bağlı bileşenleri bul, sadece geniş yatay bileşenleri (balıkları) seç
num, labels, stats, _ = cv2.connectedComponentsWithStats(mask2, connectivity=8)
clean = np.zeros_like(mask2)
for i in range(1, num):
    x, y, ww, hh, area = stats[i]
    if area < 2000:               # çok küçük gürültüyü atla
        continue
    if ww/hh < 1.2:               # yeterince geniş (yatay) değilse atla
        continue
    clean[labels == i] = 1

# 5) Kenarları yumuşatıp alfa kanalı oluştur
clean = cv2.GaussianBlur(clean.astype('float32'), (7,7), 0)
alpha = (clean > 0.5).astype('uint8') * 255

# 6) RGBA’ye dönüştür ve kaydet
b, g, r = cv2.split(img)
rgba = cv2.merge([b, g, r, alpha])
cv2.imwrite('mersinyavrusu_clean.png', rgba)

