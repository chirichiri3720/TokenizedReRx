import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import dataset
from dataset import Australian
from rerx import MLP, TokenizedMLP, J48graft, ReRx



def get_X_y(data, feature_cols, label_col):

    X, y = data[feature_cols], data[label_col].cat.codes.values.squeeze()
    return X, y

def main():
    os.makedirs("outputs", exist_ok=True)

    dataframe = Australian()
    feature_cols, label_col = dataframe.feature_columns, dataframe.target_column
    data = dataframe.data

    train_data, test_data = train_test_split(data, test_size=0.2)
    train_data, val_data = train_test_split(train_data, test_size=0.1)

    mlp = TokenizedMLP(h_dim=4)
    tree = J48graft(out_dir="outputs/")
    model = ReRx(base_model=mlp, tree=tree, output_dim=2, is_eval=True)

    X_train, y_train = get_X_y(train_data, feature_cols, label_col)
    X_val, y_val = get_X_y(val_data, feature_cols, label_col)
    X_test, y_test = get_X_y(test_data, feature_cols, label_col)

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    scores = model.evaluate(X_test, y_test)
    for metrics, score in scores.items():
        print(f"{metrics:<15}: {score}")


if __name__ == "__main__":
    main()
