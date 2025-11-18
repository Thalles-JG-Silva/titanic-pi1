import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features novas e úteis para o modelo."""
    
    df = df.copy()

    # Tamanho da família
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Passageiro viaja sozinho?
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Extrair título do nome (Mr, Miss, Mrs, etc)
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r', (\w+)\.', expand=False)
        
        # Agrupar títulos raros
        common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
        df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Rare')
        
        # Codificar títulos
        title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
        df['Title'] = df['Title'].map(title_mapping)

    return df