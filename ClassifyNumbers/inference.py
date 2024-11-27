import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image

# 入力サイズ、出力サイズ（分類問題）
D_i, D_k, D_o = 28*28, 400, 10  # MNIST画像は28x28ピクセル、出力は10クラス（数字0〜9）

# トランスフォーム（前処理） - 画像をグレースケールに変換し、テンソル化、正規化
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # グレースケール化
    transforms.Resize((28, 28)),                 # 画像を28x28ピクセルにリサイズ
    transforms.ToTensor(),                       # 画像をテンソルに変換
    transforms.Normalize((0.5,), (0.5,))         # 正規化
])

# モデルの定義（ニューラルネットワーク）
model = nn.Sequential(
    nn.Flatten(),  # 28x28の画像を1次元ベクトルに変換
    nn.Linear(D_i, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_o)
)

# 学習済みモデルのパラメータを読み込む
model.load_state_dict(torch.load('mnist_model.pth',weights_only=True))
model.eval()  # モデルを評価モードに設定

# 画像のパスを指定（例: 'sample.png'）
image_path = 'sample.png'

# 画像の前処理
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)  # バッチサイズの次元を追加

# 推論
with torch.no_grad():
    output = model(image_tensor)  # モデルの予測
    _, predicted = torch.max(output, 1)  # 最も高いスコアのクラスを取得

print(f"Predicted class: {predicted.item()}")
