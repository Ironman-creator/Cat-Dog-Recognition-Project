# 一、整体功能
# 这段代码主要围绕猫狗分类模型展开，涉及模型定义、数据加载与预处理、模型评估、预测结果展示以及相关训练指标和分类报告的可视化等操作，旨在全面评估模型在猫狗图像分类任务上的性能。
# 二、具体步骤
# 环境设置与模型定义：
# 导入一系列必要的库，包括用于文件操作、图像处理、深度学习框架相关、数据加载与处理、可视化以及模型评估指标计算等方面的库。
# 定义了设备（GPU 或 CPU），根据是否有可用的 GPU 来确定模型运行的设备。
# 定义了 AnimalCNN 类作为猫狗分类模型，该模型包含多个卷积层、池化层、批量归一化层以及全连接层，通过 forward 方法定义了数据的前向传播路径。
# 数据准备：
# 指定训练集和测试集的路径，获取训练集和测试集的标签，并确保两者标签一致后存储为 animal_labels。
# 定义了数据增强的转换操作，分别针对训练集和测试集。训练集的转换包括随机旋转、水平翻转、随机裁剪等操作后再进行归一化；测试集则主要进行尺寸调整和归一化。
# 使用 ImageFolder 结合相应的转换操作加载训练集和测试集数据，并创建 DataLoader 对象，以便按批次加载数据，设置了批次大小为 64，训练集数据加载时打乱顺序，测试集则不打乱。
# 模型加载与评估：
# 加载预训练的模型权重文件（best_model.pth），并将模型移动到指定设备上，同时展示模型的结构摘要信息。
# 定义了交叉熵损失函数（CrossEntropyLoss）作为模型的损失函数。
# 实现了 evaluate 函数用于评估模型在给定数据加载器上的性能，在评估过程中，模型设置为评估模式，计算损失、准确率以及收集真实标签和预测标签，最终返回平均损失、准确率以及相关标签列表。
# 分别使用训练集和测试集的数据加载器对模型进行评估，输出训练集和测试集的损失、准确率等信息，并生成详细的分类报告，包括精确率、召回率、F1 值等指标，同时将分类报告保存到文件中。
# 预测结果展示与可视化：
# 定义了 display_predictions 函数用于展示模型对训练集和测试集数据的预测结果，随机抽取指定数量（默认为 10 张）的图片，展示其真实标签和预测标签，并通过不同颜色区分预测正确与否。
# 定义了 plot_confusion_matrix 函数用于计算并绘制训练集和测试集的混淆矩阵，通过热力图直观展示模型在不同类别上的预测情况，将混淆矩阵图保存到指定路径下。
# 调用上述函数分别展示训练集和测试集的预测结果以及绘制相应的混淆矩阵。
# 训练指标可视化：
# 实现了 plot_training_results 函数用于读取训练过程中的损失和准确率记录文件，将其绘制成曲线图，展示训练集和测试集的损失、准确率随训练轮次的变化情况，可将曲线图保存到指定路径下。
# 调用该函数绘制并展示训练过程中的损失和准确率曲线图。

import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from sklearn.metrics import classification_report

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型
class AnimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.pool(self.conv1(x))))
        x = self.relu(self.bn2(self.pool(self.conv2(x))))
        x = self.relu(self.bn3(self.pool(self.conv3(x))))
        x = self.relu(self.bn4(self.pool(self.conv4(x))))
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 训练集和测试集路径
train_path = './dataset/train'
test_path  = './dataset/test'


# 获取训练集和测试集集的标签
train_labels = sorted(os.listdir(train_path))
test_labels = sorted(os.listdir(test_path))
if train_labels == test_labels:
    animal_labels = train_labels = test_labels
print("animal_labels: ", animal_labels)

def plot_training_results(filename, save_path=None):
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
        data = [line.strip().split(',') for line in lines]
        epochs = np.array([int(row[0]) for row in data])
        train_losses = np.array([float(row[1]) for row in data])
        train_accuracies = np.array([float(row[2]) for row in data])
        test_losses = np.array([float(row[3]) for row in data])
        test_accuracies = np.array([float(row[4]) for row in data])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, test_losses, label='Testing Loss', color='orange')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies * 100, label='Training Accuracy', color='blue')
    plt.plot(epochs, test_accuracies * 100, label='Testing Accuracy', color='orange')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

# 绘制训练过程中的损失和准确率曲线图
training_results = './training_results/training_results.txt'
save_path = './Image/LossAndAccuracy.png'
plot_training_results(training_results, save_path)

# 评估模型
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        with tqdm(total=len(loader), desc='Evaluating', unit='batch') as pbar:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                pbar.update(1)
    accuracy = correct / total
    return loss / len(loader), accuracy, all_labels, all_preds

# 加载模型
model_path = './weights/best_model.pth'
model = AnimalCNN(num_classes=len(animal_labels)).to(device)

summary(model, (3, 148, 148))

# model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only=False))
criterion = nn.CrossEntropyLoss()

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(148, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
])

test_transform = transforms.Compose([
    transforms.Resize((148, 148)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
])

train_dataset = ImageFolder(train_path, transform=train_transform)
test_dataset = ImageFolder(test_path, transform=test_transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 评估模型
train_loss, train_acc, train_labels, train_preds = evaluate(train_loader)
test_loss, test_acc, test_labels, test_preds = evaluate(test_loader)
print('Train loss: ', train_loss)
print('Train accuracy: ', train_acc)
print('Test loss: ', test_loss)
print('Test accuracy: ', test_acc)

# 输出分类报告表
train_report = classification_report(train_labels, train_preds, target_names=animal_labels)
test_report = classification_report(test_labels, test_preds, target_names=animal_labels)
print('Train Classification Report:\n', train_report)
print('Test Classification Report:\n', test_report)

# 保存分类报告表
with open('./training_results/classification_report.txt', 'w') as f:
    f.write('Train Classification Report:\n')
    f.write(train_report)
    f.write('\nTest Classification Report:\n')
    f.write(test_report)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def display_predictions(model, data_loader, animal_labels, num_images=10, images_per_row=5):
    model.eval()
    images, labels, predictions = [], [], []
    with torch.no_grad():
        for i, (img_batch, lbl_batch) in enumerate(data_loader):
            if len(images) >= num_images:
                break
            img_batch, lbl_batch = img_batch.to(device), lbl_batch.to(device)
            outputs = model(img_batch)
            _, predicted = torch.max(outputs, 1)
            images.extend(img_batch.cpu())
            labels.extend(lbl_batch.cpu())
            predictions.extend(predicted.cpu())

    fig, axes = plt.subplots(nrows=num_images // images_per_row, ncols=images_per_row, figsize=(10, 4))
    axes = axes.flatten()
    
    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy()  # 转置图像数据
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        
        true_animal = animal_labels[true_label]
        pred_animal = animal_labels[pred_label]
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_animal}\nPred: {pred_animal}', color=color, fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.1)
    plt.show()

# 显示训练集的预测结果(随机抽取10张图片)
display_predictions(model, train_loader, animal_labels)

# 显示测试集的预测结果(随机抽取10张图片)
display_predictions(model, test_loader, animal_labels)


from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(true_labels, pred_labels, animal_labels, save_path=None):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=animal_labels, yticklabels=animal_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

# 计算并绘制训练集的混淆矩阵
plot_confusion_matrix(train_labels, train_preds, animal_labels, save_path='./Image/train_confusion_matrix.png')

# 计算并绘制测试集的混淆矩阵
plot_confusion_matrix(test_labels, test_preds, animal_labels, save_path='./Image/test_confusion_matrix.png')


