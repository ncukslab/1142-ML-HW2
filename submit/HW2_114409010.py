# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # TODO 1.1: 讀取 CSV
    # TODO 1.2: 統一欄位首字母大寫，並計算缺失值數量
    df = pd.read_csv(file_path)
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isnull().sum().sum()
    return df, int(missing_count)


def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    # TODO 2.2: 以 Embarked 眾數填補
    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df


def remove_outliers(df):
    # TODO 3.1: 計算 Fare 平均與標準差
    # TODO 3.2: 移除 Fare > mean + 3*std 的資料
    F_mean=df['Fare'].mean()
    F_std=df['Fare'].std()
    out_std=F_mean+3*F_std
    df=df.drop(df[df['Fare']>out_std].index)
    return df


def encode_features(df):
    # TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼
    df_encoded = pd.get_dummies(df,columns=['Sex','Embarked'],prefix={'Sex':'sex','Embarked':'emb'})
    return df_encoded


def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age','Fare']])
    return df


def split_data(df):
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    X=df.drop('Survived',axis=1)
    Y=df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    df.to_csv(output_path,encoding='utf-8-sig',index=False)



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
