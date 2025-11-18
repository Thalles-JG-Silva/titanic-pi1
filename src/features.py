import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features novas e úteis para o modelo."""
    
    df = df.copy()

    # Tamanho da família
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Passageiro viaja sozinho?
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Extrair título do nome (Mr, Miss, Mrs, etc)
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\s*\.", expand=False)

    # Agrupar títulos raros
    df["Title"] = df["Title"].replace([
        "Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"
    ], "Rare")

    return df
