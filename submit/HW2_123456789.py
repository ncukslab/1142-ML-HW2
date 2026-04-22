# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isnull().sum().sum()
    return df, int(missing_count)


def handle_missing(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df


def remove_outliers(df):
    mean = df['Fare'].mean()
    std = df['Fare'].std()
    df = df[df['Fare'] <= mean + 3 * std].reset_index(drop=True)
    return df


def encode_features(df):
    df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    return df_encoded


def scale_features(df):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[['Age', 'Fare']] = scaler.fit_transform(df_scaled[['Age', 'Fare']])
    return df_scaled


def split_data(df):
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def save_data(df, output_path):
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
