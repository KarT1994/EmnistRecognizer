# EmnistRecognizer
フレームワークを用いずnumpyだけで実装したCNNで、アルファベット26種類の画像認識。
（活性化関数、畳み込み層だけでなくバッチ正規化やドロップアウトなどもすべてnumpyで実装）

## 学習データ
Emnist letters
大文字小文字混合の26文字手書きアルファベット
ラベルは大文字小文字分類無しの26種類（0:a,1:b,...25:z)
https://www.tensorflow.org/datasets/catalog/emnist

## モデルアーキテクチャ
「Very Deep Convolutional Networks for Large-Scale Image Recognition」を参考に構築（かなり古い論文だが実績あり）
https://arxiv.org/abs/1409.1556
BN：バッチ正規化
DP：ドロップアウト
