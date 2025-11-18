import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    """Carrega o dataset e retorna um DataFrame."""
    return pd.read_csv(path)

def clean_data(df):
    """Realiza limpeza básica no dataset Titanic."""
    
    # Remover colunas com excesso de nulos
    if "Cabin" in df.columns:
        df = df.drop(columns=["Cabin"])

    # Remover colunas inúteis (não ajudam o modelo)
    drop_cols = [col for col in ["Name", "Ticket", "PassengerId"] if col in df.columns]
    df = df.drop(columns=drop_cols)

    # Preencher idade com a mediana
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Preencher porto de embarque com a moda
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Preencher tarifa com mediana (caso precise)
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    return df

def encode_data(df):
    """Codifica variáveis categóricas com LabelEncoder."""
    
    le = LabelEncoder()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

def split_data(df):
    """Divide os dados em treino e teste."""
    
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)