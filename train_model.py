import torch
import torch.nn as nn
import os
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from emotion_model import EmotionCNN
from collections import Counter
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "data", "train")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = datasets.ImageFolder(
    root=train_path,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

class_counts = Counter(train_data.targets)
total_samples = sum(class_counts.values())

weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
class_weights = torch.tensor(weights, dtype=torch.float)

criterion = nn.CrossEntropyLoss(weight=class_weights)

model = EmotionCNN()

for param in model.model.parameters():
    param.requires_grad = False

for param in model.model.layer4.parameters():
    param.requires_grad = True

for param in model.model.fc.parameters():
    param.requires_grad = True


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

best_acc = 0

for epoch in range(20):
    correct = 0
    total = 0
    loss_sum = 0
    acc = 100 * correct / (total + 1)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "emotion_model.pth")
        print("Лучшая модель сохранена")

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} | Loss: {loss_sum:.3f} | Acc: {acc:.2f}%")
