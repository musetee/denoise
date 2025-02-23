import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# ==============================
# 1. 读取 CT 图像并添加噪声
# ==============================
def load_ct_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度 CT
    img = cv2.resize(img, (256, 256))  # 调整大小
    img = img / 255.0  # 归一化
    return img

# 生成模拟的 CT 噪声（高斯噪声）
def add_noise(image, noise_level=0.1):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)  # 保持在 0-1 范围内

# 读取 CT 图像
image_path = "your_ct_image.jpg"  # 替换为你的 CT 图片
image = load_ct_image(image_path)
noisy_image = add_noise(image)

# ==============================
# 2. 小波变换去噪
# ==============================
def wavelet_denoising(image, wavelet='db1', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, np.std(c)*0.5, mode='soft') for c in coeffs[1:]]
    return pywt.waverec2(coeffs_thresh, wavelet)

denoised_wavelet = wavelet_denoising(noisy_image)

# ==============================
# 3. PCA 进行去噪
# ==============================
def pca_denoising(image, n_components=50):
    h, w = image.shape
    flattened = image.reshape(h, w)  # 转换为 2D 矩阵
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(flattened)
    reconstructed = pca.inverse_transform(transformed)
    return reconstructed.reshape(h, w)

denoised_pca = pca_denoising(denoised_wavelet)

# ==============================
# 4. 深度学习 DnCNN 进一步去噪
# ==============================
class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_filters=64):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(1, num_filters, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_filters, 1, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)  # 残差学习：输出去噪后的结果

# 转换数据格式
transform = transforms.Compose([
    transforms.ToTensor()
])

# 构造数据集
class CTDataset(Dataset):
    def __init__(self, clean, noisy):
        self.clean = clean
        self.noisy = noisy

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return transform(self.clean[idx]).unsqueeze(0), transform(self.noisy[idx]).unsqueeze(0)

# 训练 DnCNN
def train_dncnn(denoised_pca, noisy_image, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DnCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    clean_images = [denoised_pca]
    noisy_images = [noisy_image]

    dataset = CTDataset(clean_images, noisy_images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(num_epochs):
        for clean, noisy in dataloader:
            clean, noisy = clean.to(device), noisy.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

    return model

model = train_dncnn(denoised_pca, noisy_image)

# 运行去噪
def deep_learning_denoising(model, image):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        input_tensor = transform(image).unsqueeze(0).unsqueeze(0).to(device)
        output = model(input_tensor)
        return output.squeeze().cpu().numpy()

denoised_final = deep_learning_denoising(model, denoised_pca)

# ==============================
# 5. 显示去噪结果
# ==============================
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.subplot(1, 4, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title("Noisy")

plt.subplot(1, 4, 3)
plt.imshow(denoised_pca, cmap='gray')
plt.title("Wavelet + PCA")

plt.subplot(1, 4, 4)
plt.imshow(denoised_final, cmap='gray')
plt.title("DnCNN Final")

plt.show()
