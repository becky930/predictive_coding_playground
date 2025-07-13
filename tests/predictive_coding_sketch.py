# predictive_coding_sketch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Encoder: Simple CNN
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # MNIST 

    def forward(self, x):
        return F.relu(self.conv(x))

# Feedback module (Decoder): 상위 feature로 입력을 재구성
class FeedbackDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1)

    def forward(self, features):
        return torch.sigmoid(self.deconv(features))

# 전체 모델 구조
class PredictiveCodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = FeedbackDecoder()

    def forward(self, x):
        features = self.encoder(x)
        prediction = self.decoder(features)
        error = x - prediction  # 예측 오차
        return prediction, error

input_image = torch.randn(1, 1, 28, 28)
model = PredictiveCodingNet()

# 예측 및 에러 계산
prediction, error = model(input_image)

def show_image(tensor, title=""):
    image = tensor.detach().numpy().squeeze()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

show_image(input_image[0], "Input")
show_image(prediction[0], "Prediction")
show_image(error[0], "Error (Input - Prediction)")