import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset do Titanic a partir de um CSV."""
    return pd.read_csv(path)

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza limpeza simples: trata valores faltantes e padroniza colunas."""
    
    df = df.copy()

    # Preencher idade com mediana
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # Preencher porto de embarque com o mais frequente
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Cabin tem muitos valores nulos, vamos descartar a coluna
    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    # Ticket e Name não serão usados diretamente
    if "Ticket" in df.columns:
        df.drop(columns=["Ticket"], inplace=True)

    return df
