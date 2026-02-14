import torch
from emotion_model import EmotionCNN
import sys
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class EmotionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Emotion Detector - Конкурс 'Инженеры будущего'")
        self.setGeometry(100, 100, 800, 600)
        self.device = torch.device("cpu")

        self.model = EmotionCNN().to(self.device)
        self.model.load_state_dict(
        torch.load("emotion_model.pth", map_location=self.device)
            )
        self.model.eval()

        # Интерфейс
        self.label = QLabel(self)
        self.info_label = QLabel("FPS: 0 | Эмоция: Ожидание", self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.info_label)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Видеозахват
        self.cap = cv2.VideoCapture(0)
        
        # Загрузка детектора лиц (стандартный Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Таймер для обновления кадров
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) # ~30 FPS

        # Список эмоций по заданию (FER-2013)
        self.emotion_labels = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"
        ]

    def predict_emotion(self, face_roi):

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        face = transform(face_roi)
        face = face.unsqueeze(0)  # batch dimension

        with torch.no_grad():
            outputs = self.model(face)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        emotion = self.emotion_labels[predicted.item()]
        confidence = confidence.item()

        return emotion, confidence

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        # Перевод в ч/б для детектора лиц
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Вырезаем лицо для нейросети
            face_roi = gray[y:y+h, x:x+w]
            
            # Получаем предсказание
            emotion, confidence = self.predict_emotion(face_roi)

            # Рисуем рамку и текст
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Отображение в PyQt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())