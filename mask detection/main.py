import os
import shutil
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def split_data(input_dir, split_percentage=0.8, seed=42):
    random.seed(seed)
    categories = [
        d
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d not in ["train", "test"]
    ]

    for category in categories:
        category_path = os.path.join(input_dir, category)
        images = [
            f for f in os.listdir(category_path) if f.lower().endswith((".jpg", ".png"))
        ]
        random.shuffle(images)

        split_index = int(len(images) * split_percentage)
        train_images = images[:split_index]
        test_images = images[split_index:]

        for split_name, split_images in [
            ("train", train_images),
            ("test", test_images),
        ]:
            split_dir = os.path.join(input_dir, split_name, category)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                src = os.path.join(category_path, img)
                dst = os.path.join(split_dir, img)
                shutil.move(src, dst)

        if not os.listdir(category_path):
            os.rmdir(category_path)


def create_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(
        data_dir, "test"
    )  # 'test' klasörü doğrulama için kullanılıyor

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    classes = list(train_generator.class_indices.keys())
    return train_generator, val_generator, classes


def build_model(base_model_name="MobileNetV2"):
    if base_model_name == "MobileNetV2":
        base_model = MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
    elif base_model_name == "ResNet50":
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
    elif base_model_name == "EfficientNetB0":
        base_model = EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
    else:
        raise ValueError("Invalid base model name")

    base_model.trainable = False

    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    return model


def train_and_evaluate(data_dir):
    split_data(data_dir)
    img_size = (224, 224)
    batch_size = 32
    train_generator, val_generator, classes = create_data_generators(
        data_dir, img_size, batch_size
    )

    model = build_model("MobileNetV2")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=30,
        callbacks=[early_stop],
    )

    # Eğitim grafiklerini çiz
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Eğitim Kaybı")
    plt.plot(history.history["val_loss"], label="Doğrulama Kaybı")
    plt.title("Epoch Boyunca Kayıp Değişimi")
    plt.xlabel("Epoch")
    plt.ylabel("Kayıp")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Eğitim Doğruluğu")
    plt.plot(history.history["val_accuracy"], label="Doğrulama Doğruluğu")
    plt.title("Epoch Boyunca Doğruluk Değişimi")
    plt.xlabel("Epoch")
    plt.ylabel("Doğruluk")
    plt.legend()
    plt.tight_layout()
    plt.savefig("egitim_grafikleri.png")
    plt.show()

    # Model değerlendirme
    val_generator.reset()
    y_true = val_generator.classes
    y_pred_prob = model.predict(val_generator)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Gerçek Etiket")
    plt.xlabel("Tahmin Edilen Etiket")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=classes))

    # Metrikleri txt dosyası olarak kaydet
    report_text = classification_report(y_true, y_pred, target_names=classes)
    with open("classification_report.txt", "w") as f:
        f.write(report_text)

    print(f"Dogruluk (Accuracy): {report['accuracy']:.4f}")
    print(f"With Mask Precision: {report['with_mask']['precision']:.4f}")
    print(f"Without Mask Recall: {report['without_mask']['recall']:.4f}")
    print(f"Genel F1-Score: {report['macro avg']['f1-score']:.4f}")

    # 5 örnek görsel üzerinde tahminleri göster ve kaydet
    show_sample_predictions(model, val_generator, classes, img_size)

    model.save("mask_detection_model.h5")
    print("Model kaydedildi: mask_detection_model.h5")


def show_sample_predictions(model, val_generator, classes, target_size, sample_count=5):
    import matplotlib.patches as patches

    # Val generator'daki tüm dosya yolları
    filepaths = val_generator.filepaths
    # Rastgele 5 dosya seç
    sample_files = random.sample(filepaths, sample_count)

    plt.figure(figsize=(15, 3 * sample_count))
    for i, img_path in enumerate(sample_files):
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        pred_prob = model.predict(img_input)[0][0]
        pred_label = classes[1] if pred_prob > 0.5 else classes[0]
        true_label = os.path.basename(os.path.dirname(img_path))

        plt.subplot(sample_count, 1, i + 1)
        plt.imshow(img_array)
        plt.axis("off")
        plt.title(
            f"Gerçek: {true_label} | Tahmin: {pred_label} ({pred_prob:.2f})",
            fontsize=14,
            color="green" if pred_label == true_label else "red",
        )

    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.show()


if __name__ == "__main__":
    train_and_evaluate("./data")
