import numpy as np
import tensorflow as tf
from tokenizer import CategoricalFeatureTokenizer

# テスト用のデータ
cardinalities = [2, 3, 14, 8, 2, 2, 2, 3]  # カテゴリカル特徴の種類数
d_token = 1  # 各トークンの次元数
bias = True  # バイアスを使用する
initialization = 'uniform'  # パラメータの初期化方法

# トークナイザーのインスタンスを作成
tokenizer = CategoricalFeatureTokenizer(cardinalities=cardinalities, d_token=d_token, bias=bias, initialization=initialization)

# テスト用の入力データを作成
# 各特徴量が取る値はその特徴のカテゴリ数未満でなければなりません
x_test = tf.constant([
[0,0,0,0,0,0,0,0],
 [1,  1 , 1 , 1,  1,  1 , 1 , 1],
 [1,  2,  2,  2,  1,  1,  1,  2],
 [1,  2,  3,  3,  1,  1,  1,  2],
 [1,  2,  4,  4,  1,  1,  1,  2],
 [1,  2,  5,  5,  1,  1,  1,  2],
 [1,  2,  6,  6,  1,  1,  1,  2],
 [1,  2,  7,  7,  1,  1,  1,  2],
 [1,  2,  8,  7,  1,  1,  1,  2],
 [1,  2,  9,  7,  1,  1,  1,  2],
 [1,  2, 10,  7,  1,  1,  1,  2],
 [1,  2, 11,  7,  1,  1,  1,  2],
 [1,  2, 12,  7,  1,  1,  1,  2],
 [1,  2, 13,  7,  1,  1,  1,  2]])

print(x_test + tokenizer.category_offsets[None])
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
print(tokens.numpy())  # 最初の2つのトークンを表示

# 埋め込み層の重みの形状を確認
embedding_weights = tokenizer.embeddings.get_weights()[0]
print("埋め込み層の重みの形状:", embedding_weights.shape)
print(embedding_weights)

# バイアスベクトルがある場合は形状を確認
if bias:
    print("バイアスベクトルの形状:", tokenizer.bias.shape)
    print(tokenizer.bias.numpy())

print("テストが正常に完了しました！")
