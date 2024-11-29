# 一、整体功能
# 这段代码主要实现了一个基于卷积神经网络（CNN）的猫狗分类模型的训练过程，包括数据准备、模型定义、训练循环、模型评估以及训练结果的保存等操作，旨在通过训练得到一个能够有效区分猫狗图像的模型，并记录相关训练指标以便后续分析。
# 二、具体步骤
# 数据准备：
# 首先指定了训练集和测试集的路径，然后通过读取对应路径下的文件列表获取训练集和测试集的标签，并确保两者标签一致后将其存储为 animal_labels。
# 定义了数据增强的转换操作，对于训练集，包含随机旋转、水平翻转、随机裁剪等操作后再转换为张量并进行归一化；对于测试集，主要进行尺寸调整后转换为张量并归一化。
# 使用 ImageFolder 类结合相应的转换操作加载训练集和测试集数据，并通过 DataLoader 创建数据加载器，设置批次大小为 64，训练集数据加载时打乱顺序，测试集则不打乱。
# 模型定义：
# 定义了 AnimalCNN 类作为猫狗分类的卷积神经网络模型，该模型包含多个卷积层、池化层、批量归一化层以及全连接层。通过 forward 方法详细描述了数据在模型中的前向传播过程，即数据依次经过各层的处理最终得到分类结果。
# 将定义好的模型移动到指定设备（GPU 或 CPU，根据是否有可用的 GPU 来确定）上运行。
# 训练设置：
# 定义了损失函数为交叉熵损失（CrossEntropyLoss），优化器为 Adam 优化器，并设置了学习率为 0.001。
# 创建了几个空列表，用于存储训练过程中每个 epoch 的训练损失、训练准确率、测试损失和测试准确率等指标。
# 训练循环：
# 通过循环进行多个 epoch 的训练，在每个 epoch 中：
# 首先将模型设置为训练模式，在训练数据加载器上进行迭代。对于每一批次的数据，将图像和标签数据移动到指定设备上，先清空优化器的梯度，然后将数据传入模型得到输出，计算损失并进行反向传播更新模型参数，同时累计该批次的损失和计算预测正确的样本数量，通过 tqdm 进度条实时显示当前 epoch 的训练损失和准确率信息。
# 完成一轮训练数据的迭代后，计算该 epoch 的训练损失和训练准确率，并将其分别添加到对应的列表中。
# 接着将模型设置为评估模式，在测试数据加载器上进行类似的迭代操作，但不进行梯度更新，用于计算该 epoch 的测试损失和测试准确率，并添加到相应列表中，同时通过 tqdm 进度条实时显示测试损失和准确率信息。
# 在每个 epoch 结束后，打印出当前 epoch 的训练损失、训练准确率、测试损失和测试准确率等详细信息。
# 根据测试准确率判断是否保存当前模型为最佳模型，如果是第一个 epoch 或者当前 epoch 的测试准确率高于之前所有 epoch 的测试准确率（除当前 epoch 外），则将模型的状态字典保存到指定的权重文件（这里保存为 new_best_model.pth）中。
# 训练结果保存：
# 创建 training_results 文件夹（如果不存在），指定一个新的训练结果记录文件路径（new_training_results.txt）。
# 将每个 epoch 的训练损失、训练准确率、测试损失和测试准确率等信息按照规定格式写入到该文件中，以便后续查看和分析训练过程的变化情况。
# 最后，将训练结束后的最终模型状态字典保存到指定的权重文件（new_final_model.pth）中。
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 训练集和测试集路径
train_path = './Dataset/train'
test_path = './Dataset/test'

# 获取训练集和测试集集的标签
train_labels = sorted(os.listdir(train_path))
test_labels = sorted(os.listdir(test_path))
if train_labels == test_labels:
    animal_labels = train_labels = test_labels
print("animal_labels: ", animal_labels)

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

# 加载数据集
train_dataset = ImageFolder(train_path, transform=train_transform)
test_dataset = ImageFolder(test_path, transform=test_transform)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AnimalCNN(num_classes=len(animal_labels)).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 25 
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Train Loss': running_loss / (pbar.n + 1),
                'Train Acc': correct / total
            })
            pbar.update(1)
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                pbar.set_postfix({
                    'Test Loss': test_loss / (pbar.n + 1),
                    'Test Acc': correct / total
                })
                pbar.update(1)
    
    test_loss = test_loss / len(test_loader)
    test_acc = correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # 保存最佳模型
    if epoch == 0 or test_acc > max(test_accuracies[:-1]):
        # torch.save(model.state_dict(), './weights/best_model.pth')
        torch.save(model.state_dict(), './weights/new_best_model.pth')

# 训练结束后，将结果写入文件
os.makedirs('./training_results', exist_ok=True)
# training_results = './training_results/training_results.txt'
training_results = './training_results/new_training_results.txt'
with open(training_results, 'w') as f:
    f.write('epoch,train_losses,train_accuracies,test_losses,test_accuracies\n')
    for i in range(len(train_losses)):
        f.write(f'{i+1},{train_losses[i]},{train_accuracies[i]},{test_losses[i]},{test_accuracies[i]}\n')

# 保存最终的模型
# torch.save(model.state_dict(), './weights/final_model.pth')
torch.save(model.state_dict(), './weights/new_final_model.pth')

