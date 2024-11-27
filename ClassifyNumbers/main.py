import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 入力サイズ、出力サイズ（分類問題）
D_i, D_k, D_o = 28*28, 240, 10  # MNIST画像は28x28ピクセル、出力は10クラス（数字0〜9）

# トランスフォーム（前処理） - MNIST画像をテンソルに変換し、正規化
transform = transforms.Compose([
    transforms.ToTensor(),  # 画像をテンソルに変換
    transforms.Normalize((0.5,), (0.5,))  # ピクセル値を[-1, 1]に正規化
])

# データセットのダウンロード
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# データローダーの作成（バッチサイズを指定）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# モデルの定義（ニューラルネットワーク）
model = nn.Sequential(
    nn.Flatten(),  # 28x28の画像を1次元ベクトルに変換
    nn.Linear(D_i, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_o)
)

# He初期化
def weights_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        layer.bias.data.fill_(0.0)

model.apply(weights_init)

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()  # 分類問題なのでCrossEntropyLoss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# エポック数
epochs = 25

# トレーニングループ
for epoch in range(epochs):
    model.train()  # モデルを訓練モードに設定
    epoch_loss = 0.0
    correct = 0
    total = 0
    for data in train_loader:
        x_batch, y_batch = data
        optimizer.zero_grad()  # 勾配をゼロにリセット
        pred = model(x_batch)  # モデルの予測
        loss = criterion(pred, y_batch)  # 損失を計算
        loss.backward()  # 誤差逆伝播
        optimizer.step()  # オプティマイザによるパラメータ更新
        epoch_loss += loss.item()

        # 精度を計算
        _, predicted = torch.max(pred, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# テストデータでの評価
model.eval()  # モデルを評価モードに設定
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        x_batch, y_batch = data
        pred = model(x_batch)
        _, predicted = torch.max(pred, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# モデルを保存
torch.save(model.state_dict(), 'mnist_model.pth')

