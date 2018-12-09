import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def get_data():
    df = pd.read_csv('file', header=None)
#################################################################
    # for Embedded System
    # normalize
    df_x = df.iloc[:, :9]
    df_x = df_x.div(df_x.sum(axis=1), axis=0)  # normalize
    print(df_x.head(10))
    df_x_scale_temp = StandardScaler().fit_transform(df_x)  # numpy.array
    #df_x_scale = pd.DataFrame(data=df_x_scale_temp)
    X_scaling = df_x_scale_temp  # input pin
    y = df.iloc[:, -1]  # location
    y_new = y-1
#################################################################
    # df_new = df.replace({'location 1': 0, 'location 2': 1, 'location 3': 2, 'location 4': 3,
    #                      'location 5': 4, 'location 6': 5, 'location 7': 6, 'location 8': 7,
    #                      'location 9': 8, 'location 10': 9, 'location 11': 10, 'location 12': 11,
    #                      'location 13': 12, 'location 14': 13, 'location 15': 14, 'location 16': 15,
    #                      'location 17': 16, 'location 18': 17, 'location 19': 18, 'location 20': 19,
    #                      'location 21': 20, 'location 22': 21, 'location 23': 22, 'location 24': 23,
    #                      'location 25': 24})
    # X = df_new.iloc[:, :9]
    # # print(X.head(10))
    # X_scaling = StandardScaler().fit_transform(X)  # numpy.array
    # y_new = df_new.iloc[:, -1]
#################################################################
    X_train, y_train, X_test, y_test = train_test_split(
        X_scaling, y_new.values, test_size=0.3, random_state=2018)  # return np.array #if X_scaling if not pandas,use .values

    # avoid using Variable and still get grad
    train_feature = torch.tensor(X_train, requires_grad=True)
    train_target = torch.tensor(X_test).type(torch.LongTensor)

    test_feature = torch.torch.tensor(y_train, requires_grad=True)
    test_target = torch.torch.tensor(y_test).type(
        torch.LongTensor)  # data type must be long type

    train_datasets = torch.utils.data.TensorDataset(
        train_feature, train_target)
    test_datasets = torch.utils.data.TensorDataset(
        test_feature, test_target)

    return train_datasets, test_datasets
