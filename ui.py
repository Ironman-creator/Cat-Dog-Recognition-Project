# 一、整体功能
# 这段代码创建了一个图形用户界面（GUI）应用程序，用于实现猫狗图像的识别功能。它基于之前定义的 AnimalCNN 模型，允许用户通过界面上传猫狗图像，然后利用加载的预训练模型对图像进行分类识别，并在界面上展示识别结果（包括类别和置信度）以及上传的图像本身。
# 具体内容
# 二、模型定义：
# 重新定义了 AnimalCNN 类，其结构与前面代码部分相同，是一个用于猫狗分类的卷积神经网络模型，包含多个卷积层、池化层、批量归一化层以及全连接层，通过 forward 方法描述了数据的前向传播过程以得到分类结果。
# 应用程序类定义（AnimalClassifierApp）：
# 初始化与界面布局设置：
# 类继承自 QWidget，在 __init__ 方法中调用 initUI 方法来初始化界面。
# initUI 方法中，首先设置了窗口标题为 “猫狗识别系统” 并调整窗口尺寸。然后通过 QGridLayout 等布局管理器创建了复杂的界面布局，包括用于显示标题的 QLabel（设置了字体、对齐方式等样式）、用于显示图像的固定尺寸 QLabel、用于显示识别结果的只读 QTextEdit（设置了字体颜色、大小、对齐方式等样式）以及包含 “图像载入”“猫狗识别”“退出系统” 等按钮的 QVBoxLayout，最后将这些组件添加到主布局 QGridLayout 中并设置为窗口的布局。
# 模型加载与数据转换定义：
# 确定运行设备（GPU 或 CPU），实例化 AnimalCNN 模型，并加载预训练的模型权重文件（./weights/best_model.pth），将模型移动到指定设备上并设置为评估模式。
# 定义了数据转换操作，用于将用户上传的图像调整尺寸、转换为张量并进行归一化处理，以便能作为模型的输入。同时初始化了一个变量 image_path 用于存储用户上传图像的路径。
# 按钮功能实现：
# load_image 方法（上传图像按钮功能）：
# 当用户点击 “图像载入” 按钮时，通过 QFileDialog 弹出文件选择对话框，让用户选择一张图片文件（支持常见的图像格式如 .jpg、.jpeg、.png）。如果用户选择了文件，就将文件路径存储到 image_path 变量中，然后将所选图片显示在界面的 image_label 上，并清空 result_label 中的识别结果文本。
# classify_image 方法（识别按钮功能）：
# 当用户点击 “猫狗识别” 按钮且 image_path 有值（即已上传图像）时，首先使用 PIL 库的 Image 模块打开上传的图像并转换为 RGB 格式，然后通过之前定义的数据转换操作将其转换为适合模型输入的张量格式并移动到指定设备上。
# 在不进行梯度计算的情况下（with torch.no_grad()），将转换后的图像张量传入模型进行预测，得到输出结果后通过 softmax 函数计算各类别的概率，进而确定预测的类别（猫或狗）和对应的置信度。
# 最后，再次将上传的图像以合适的尺寸显示在 image_label 上，并在 result_label 中更新显示识别结果的文本信息，包括识别结果（猫或狗）和置信度，并设置好文本的对齐方式和光标位置等。
# 应用程序启动：
# 在 if __name__ == '__main__': 语句块中，创建了 QApplication 实例，实例化了 AnimalClassifierApp 类并显示窗口，然后通过 app.exec_() 进入应用程序的事件循环，等待用户操作，直到用户关闭应用程序窗口，此时程序退出。
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit
from PyQt5.QtGui import QPixmap, QImage, QFont, QTextCursor
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

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

class AnimalClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('猫狗识别系统')
        self.resize(900, 600)  # 设置窗口尺寸
        
        # 创建布局
        grid = QGridLayout()

        # 创建并设置标题
        title_label = QLabel('猫狗识别系统', self)
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        grid.addWidget(title_label, 0, 0, 1, 4)

        # 显示图像的标签
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(400, 400)  # 调整图像显示尺寸
        grid.addWidget(self.image_label, 1, 0, 2, 2)

        # 识别结果的标签
        self.result_label = QTextEdit(self)
        self.result_label.setFixedSize(200, 100)
        self.result_label.setReadOnly(True)
        self.result_label.setStyleSheet("color: red; font-size: 16px;")
        self.result_label.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.result_label, 1, 2, 1, 2)

        # 按钮布局
        button_layout = QVBoxLayout()

        # 上传图像按钮
        upload_btn = QPushButton('图像载入', self)
        upload_btn.clicked.connect(self.load_image)
        button_layout.addWidget(upload_btn)

        # 识别按钮
        recognize_btn = QPushButton('猫狗识别', self)
        recognize_btn.clicked.connect(self.classify_image)
        button_layout.addWidget(recognize_btn)

        # 退出按钮
        exit_btn = QPushButton('退出系统', self)
        exit_btn.clicked.connect(self.close)
        button_layout.addWidget(exit_btn)

        grid.addLayout(button_layout, 2, 2, 1, 2)
        self.setLayout(grid)

        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AnimalCNN(num_classes=2)  
        self.model.load_state_dict(torch.load('./weights/best_model.pth', map_location=self.device, weights_only=False))
        self.model.to(self.device)
        self.model.eval()

        # 定义数据转换
        self.transform = transforms.Compose([
            transforms.Resize((148, 148)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
        ])

        self.image_path = ''

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.result_label.setText('识别结果: ')

    def classify_image(self):
        if self.image_path:
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                label = 'cat' if predicted.item() == 0 else 'dog'
                confidence = confidence.item()

            # 将图像转换为QPixmap
            pixmap = QPixmap(self.image_path)
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            
            # 设置识别结果字体颜色和对齐方式
            self.result_label.setText(f'识别结果如下:\n\n识别结果: {label} \n置信度: {confidence:.2f}')
            self.result_label.setAlignment(Qt.AlignCenter)
            cursor = self.result_label.textCursor()
            cursor.select(QTextCursor.Document)
            self.result_label.setTextCursor(cursor)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AnimalClassifierApp()
    ex.show()
    sys.exit(app.exec_())

       
