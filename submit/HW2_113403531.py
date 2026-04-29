# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # TODO 1.1: 讀取 CSV
    # 修正點：必須將讀取的結果賦值給 df，不可維持 None
    df = pd.read_csv(file_path)

    # TODO 1.2: 統一欄位首字母大寫，並計算缺失值數量
    df.columns = [c.capitalize() for c in df.columns]

    # 使用 .isnull().sum().sum() 計算全表缺失值總和
    missing_count = df.isnull().sum().sum()

    return df, int(missing_count)


def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    # 計算 Age 的中位數
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)

    # TODO 2.2: 以 Embarked 眾數填補
    # .mode() 回傳的是 Series，取 index 0 拿到出現次數最多的值
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)

    return df


def remove_outliers(df):
    # TODO 3.1: 計算 Fare 平均與標準差
    fare_mean = df['Fare'].mean()
    fare_std = df['Fare'].std()

    # TODO 3.2: 移除 Fare > mean + 3*std 的資料
    # 定義上限門檻
    upper_limit = fare_mean + 3 * fare_std

    # 過濾掉超過門檻的列，保留小於等於門檻的資料
    df = df[df['Fare'] <= upper_limit]

    return df


def encode_features(df):
    # 先列出你想要轉換的目標
    # 這裡假設 load_data 處理後的結果是首字母大寫
    target_cols = ['Sex', 'Embarked']

    # 檢查這些欄位是否真的存在於目前的 df 中
    # 這樣可以避免重複執行 Cell 導致的 KeyError
    existing_cols = [c for c in target_cols if c in df.columns]

    if not existing_cols:
        print("警告：找不到目標欄位，可能已經編碼過了！")
        return df

    # 動態產生 prefix 字典
    prefix_dict = {col: col for col in existing_cols}

    df_encoded = pd.get_dummies(
        df, 
        columns=existing_cols, 
        prefix=prefix_dict,
        dtype=int
    )

    return df_encoded


def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()

    # 建立 df 的副本，避免直接修改原始資料
    df_scaled = df.copy()

    # 指定要縮放的欄位
    cols_to_scale = ['Age', 'Fare']

    # fit_transform 會計算平均值與標準差並進行轉換
    # 將轉換後的 NumPy 陣列填回對應的欄位
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

    return df_scaled


def split_data(df):
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    # drop(axis=1) 代表移除指定欄位來取得特徵矩陣
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    # test_size=0.2 代表 20% 資料用於測試，80% 用於訓練
    # random_state=42 確保每次執行的切割結果都一樣（方便實驗重現）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    # index=False 是為了避免把 Pandas 自動生成的索引（0, 1, 2...）也存入 CSV 中
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
