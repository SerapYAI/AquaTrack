import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

model = tf.keras.models.load_model("model.keras")


# Görseli yükleme
data_dir = "data"  # resimlerin olduğu klasör
for fname in sorted(os.listdir(data_dir)):
    if not fname.lower().endswith(".png"):
        continue

    img_path = os.path.join(data_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"{fname} okunamadı.")
        continue


# İşleme: resize, invert, normalize
img = cv2.resize(img, (28,28))
img = 255 - img
img_processed = img.astype("float32") / 255.0
img_processed = img_processed.reshape(1,28,28,1)

# Tahmin yap
prediction = model.predict(img_processed)
predicted_digit = np.argmax(prediction)

# ============
# Grafik çizimi
# ============
plt.figure(figsize=(12,4))

# Orijinal görsel
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Orijinal Görsel")
plt.axis('off')

# İşlenmiş 28x28 giriş
plt.subplot(1,3,2)
plt.imshow(img_processed.reshape(28,28), cmap='gray')
plt.title("28x28 Gri Görsel")
plt.axis('off')

# Tahmin olasılıkları bar plot
plt.subplot(1,3,3)
plt.bar(range(10), prediction[0], color='red')
plt.title("Tahmin Olasılıkları")
plt.xlabel("Rakamlar")
plt.ylabel("Güven Skoru")
plt.xticks(range(10))

plt.tight_layout()
plt.show()

print("Tahmin edilen rakam:", predicted_digit)