import numpy as np
import tensorflow as tf
from tokenizer import CategoricalFeatureTokenizer,FeatureTokenizer

# 数値特徴が2つ、カテゴリカル特徴が3つの例
n_num_features = 2
cat_cardinalities = [3, 4, 5]  # それぞれのカテゴリカル特徴の異なる値の数
d_token = 1
initialization = 'uniform'

# トークナイザーの初期化
feature_tokenizer = FeatureTokenizer(n_num_features, cat_cardinalities, d_token,initialization)

# 入力データの例 (4つのサンプル、5つの特徴)
inputs = tf.constant([
    [1.0, 2.0, 0, 1, 3],
    [2.0, 3.0, 1, 0, 4],
    [3.0, 1.0, 2, 1, 0],
    [4.0, 5.0, 0, 3, 2]
], dtype=tf.float32)

# トークン化
tokens = feature_tokenizer(inputs)
print(tokens.shape)  # トークンの形状を確認

print(tokens)