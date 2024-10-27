import numpy as np
import tensorflow as tf

from tokenizer import CategoricalFeatureTokenizer

# テスト用のデータ
cardinalities = [3, 10]  # カテゴリカル特徴の種類数
d_token = 3  # 各トークンの次元数
bias = True  # バイアスを使用
initialization = 'uniform'  # パラメータの初期化方法

# トークナイザーのインスタンスを作成
tokenizer = CategoricalFeatureTokenizer(cardinalities, d_token, bias, initialization)

# テスト用の入力データを作成
# 各特徴量が取る値はその特徴のカテゴリ数未満でなければなりません
x_test = tf.constant([
    [0, 5],
    [1, 7],
    [0, 2],
    [2, 4]
])

# トークナイザーを適用して出力トークンを取得
tokens = tokenizer(x_test)

# トークンの形状を確認
n_objects, n_features = x_test.shape
expected_shape = (n_objects, n_features, d_token)

print("出力トークンの形状:", tokens.shape)
print("期待される形状:", expected_shape)

# 形状が一致するか確認
assert tokens.shape == expected_shape, "トークナイザーの出力形状が期待と異なります"

# 出力トークンの一部を表示
print("出力トークンの一部:")
print(tokens.numpy()[:2])  # 最初の2つのトークンを表示

# 埋め込み層の重みの形状を確認
embedding_weights = tokenizer.embeddings.get_weights()[0]
print("埋め込み層の重みの形状:", embedding_weights.shape)

# バイアスベクトルがある場合は形状を確認
if bias:
    print("バイアスベクトルの形状:", tokenizer.bias.shape)
