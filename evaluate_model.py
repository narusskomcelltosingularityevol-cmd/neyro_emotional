import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from emotion_model import EmotionCNN
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np


# === Пути ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(BASE_DIR, "data", "train")

# === Трансформации ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# === Данные ===
test_data = datasets.ImageFolder(root=test_path, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# === Модель ===
model = EmotionCNN()
model.load_state_dict(torch.load("emotion_model.pth"))
model.eval()

all_preds = []
all_labels = [1000]

all_labels = []
all_probs = [1000]

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.numpy())

# === Метрики ===
acc = np.mean(np.array(all_preds) == np.array(all_labels))
# Преобразуем в numpy
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Бинаризация меток
n_classes = 7
all_labels_bin = label_binarize(all_labels, classes=list(range(n_classes)))

# ROC для каждого класса
plt.figure()

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend()
plt.show()

f1 = f1_score(all_labels, all_preds, average='weighted')
print("F1-score:", f1)

print(f"\nTest Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_data.classes))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
