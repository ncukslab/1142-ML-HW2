# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # TODO 1.1: 讀取 CSV
    df = pd.read_csv(file_path)

    # TODO 1.2: 統一欄位首字母大寫，並計算缺失值數量
    df.columns = [c.capitalize() for c in df.columns]

    # 計算全表缺失值的總數
    missing_count = df.isnull().sum().sum()

    return df, int(missing_count)


def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    # 先計算 Age 的中位數，然後直接填補回該欄位
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)

    # TODO 2.2: 以 Embarked 眾數填補
    # .mode() 會回傳一個 Series，我們取第 0 個位置的最常見值
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)

    return df


def remove_outliers(df):
    # TODO 3.1: 計算 Fare 平均與標準差
    fare_mean = df['Fare'].mean()
    fare_std = df['Fare'].std()

    # TODO 3.2: 移除 Fare > mean + 3*std 的資料
    upper_bound = fare_mean + 3 * fare_std
    df = df[df['Fare'] <= upper_bound]

    return df


def encode_features(df):
    # TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼
    df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int)

    return df_encoded


def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()
    df_scaled = df.copy()
    cols_to_scale = ['Age', 'Fare']
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df_scaled


def split_data(df):
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')



if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_CSV = os.path.join(BASE_DIR, "..", "titanic.csv")
    OUTPUT_CSV = os.path.join(BASE_DIR, "..", "titanic_processed.csv")

    df, missing_count = load_data(INPUT_CSV)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = encode_features(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_data(df, OUTPUT_CSV)

    print("Titanic 資料前處理完成")
