import logging

import openml


logger = logging.getLogger(__name__)


def get_task_and_dim_out(data_id, df, columns, cate_indicator, target_col):
    target_idx = columns.index(target_col)

    if data_id in exceptions_binary:
        task = "binary"
        dim_out = 1
    elif data_id in exceptions_multiclass:
        task = "multiclass"
        dim_out = int(df[target_col].nunique())
    elif cont_checker(df, target_col, cate_indicator[target_idx]):
        task = "regression"
        dim_out = 1
    elif int(df[target_col].nunique()) == 2:
        task = "binary"
        dim_out = 1
    else:
        task = "multiclass"
        dim_out = int(df[target_col].nunique())
    return task, dim_out


def cont_checker(df, col, is_cate):
    return not is_cate and df[col].dtype != bool and df[col].dtype != object


def cate_checker(df, col, is_cate):
    return is_cate or df[col].dtype == bool or df[col].dtype == object


def get_columns_list(df, columns, cate_indicator, target_col, checker):
    return [col for col, is_cate in zip(columns, cate_indicator) if col != target_col and checker(df, col, is_cate)]


def print_dataset_details(dataset: openml.datasets.OpenMLDataset):
    df, _, cate_indicator, columns = dataset.get_data(dataset_format="dataframe")
    print(dataset.name)
    print(dataset.openml_url)
    print(df)

    target_col = dataset.default_target_attribute
    print("Nan count", df.isna().sum().sum())
    print("cont", get_columns_list(df, columns, cate_indicator, target_col, cont_checker))
    print("cate", get_columns_list(df, columns, cate_indicator, target_col, cate_checker))
    print("target", target_col)

    task, dim_out = get_task_and_dim_out(dataset.id, df, columns, cate_indicator, target_col)
    print(f"task: {task}")
    print(f"dim_out: {dim_out}")
    print(df[target_col].value_counts())
    exit()



class OpenMLDataFrame(object):
    def __init__(self, show_details: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.show_details = show_details

    
    def download(self, id) -> None:

        dataset = openml.datasets.get_dataset(id)

        if self.show_details:
            print_dataset_details(dataset)

        self.data, _, cate_indicator, columns = dataset.get_data(dataset_format="dataframe")

        self.target_column = dataset.default_target_attribute
        self.continuous_columns = [x for i, x in enumerate(columns) if cate_indicator[i] == False and x != self.target_column]
        self.categorical_columns = [x for x in columns if x not in self.continuous_columns and x != self.target_column]
        self.feature_columns = self.continuous_columns + self.categorical_columns
        self.data = self.data.dropna(axis=0)


class Australian(OpenMLDataFrame):
    def __init__(self, **kwargs) -> None:
        super().__init__( **kwargs)
        id = 40981
        self.download(id)
