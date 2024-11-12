"""@private"""
import enum
import math
import time
import warnings
from typing import (
    List,
    cast,
)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
from tensorflow.keras.initializers import RandomUniform, RandomNormal


class BaseTokenizer(Layer):
    def __init__(
        self,
        d_token: int,
        bias: bool,
        initialization: str,
        **kwargs
    ) -> None:
        super().__init__()

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.weight.shape[1]
    
    def call(self, inputs):
       NotImplementedError()

class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) -> '_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, shape, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            return RandomUniform(minval=-d_sqrt_inv, maxval=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            return RandomNormal(mean=0.0, stddev=d_sqrt_inv)

class CategoricalFeatureTokenizer(BaseTokenizer):
    """Transforms categorical features to tokens (embeddings).

    See `FeatureTokenizer` for the illustration.

    The module efficiently implements a collection of `torch.nn.Embedding` (with
    optional biases).

    Examples:
        .. testcode::

            # the input must contain integers. For example, if the first feature can
            # take 3 distinct values, then its cardinality is 3 and the first column
            # must contain values from the range `[0, 1, 2]`.
            cardinalities = [3, 10]
            x = torch.tensor([
                [0, 5],
                [1, 7],
                [0, 2],
                [2, 4]
            ])
            n_objects, n_features = x.shape
            d_token = 3
            tokenizer = CategoricalFeatureTokenizer(cardinalities, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """

    def __init__(
        self,
        cardinalities: List[int],
        d_token: int,
        bias: bool = True,
        initialization: str = 'uniform'
    ) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature. For example,
                :code:`cardinalities=[3, 4]` describes two features: the first one can
                take values in the range :code:`[0, 1, 2]` and the second one can take
                values in the range :code:`[0, 1, 2, 3]`.
            d_token: the size of one token.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of feature value. The bias vectors are not shared
                between features.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__(d_token,bias,initialization)
        assert cardinalities, 'cardinalities must be non-empty'
        assert d_token > 0, 'd_token must be positive'
        initialization_ = _TokenInitialization.from_str(initialization)

        category_offsets = tf.constant(np.cumsum([0] + cardinalities[:-1]),dtype=tf.int32)

        self.category_offsets = category_offsets        
        self.embeddings = Embedding(input_dim=sum(cardinalities),output_dim=d_token,embeddings_initializer=initialization_.apply(shape=None, d=d_token))
        
        if bias:
            self.bias = self.add_weight(
                    shape=(len(cardinalities), d_token),
                    initializer=initialization_.apply(shape=None, d=d_token),
                    trainable=True,
                    name='bias_vector'
            ) 
        else:
            self.bias = None
     
    def call(self, inputs):
        x = inputs
        # x = inputs + tf.expand_dims(self.category_offsets, axis=0)
        x = self.embeddings(x)
        if self.bias is not None:
            x += self.bias
            # x += tf.expand_dims(self.bias, axis=0)
        return x


class FeatureTokenizer(BaseTokenizer):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
        initialization: str = 'uniform'
    ) -> None:
    
        super().__init__(n_num_features, cat_cardinalities, d_token)
        assert n_num_features >= 0, 'n_num_features must be non-negative'
        assert (
            n_num_features or cat_cardinalities
        ), 'at least one of n_num_features or cat_cardinalities must be positive/non-empty'
        self.initialization = 'uniform'
        
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(
                cat_cardinalities, d_token, True, self.initialization
            )
            if cat_cardinalities
            else None
        )
        self.n_num_features = n_num_features


    def call(self,inputs):
        num_features = inputs[:, :self.n_num_features]  # 数値データを抽出
        cat_features = inputs[:, self.n_num_features:]  # カテゴリカルデータを抽出

        # カテゴリカルデータをトークナイズ
        if self.cat_tokenizer is not None:
            cat_tokens = self.cat_tokenizer(cat_features)
        else:
            cat_tokens = tf.zeros(tf.shape(cat_features) + (self.d_token,),dtype=tf.float32)

        # 数値データをそのまま使用
        # 数値データの次元をトークンと同じ次元に拡張
        num_tokens = tf.expand_dims(num_features, axis=-1)
        # num_tokens = tf.cast(num_features, dtype=tf.float32)
    

        tokens = tf.concat([num_tokens,cat_tokens],axis=1)
        return tokens