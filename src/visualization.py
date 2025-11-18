# src/visualization.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer

def plot_survival_count(df):
    sns.countplot(data=df, x="Survived")
    plt.title("Contagem de Sobreviventes")
    plt.xlabel("Sobreviveu")
    plt.ylabel("Quantidade")
    plt.show()

def plot_age_distribution(df):
    sns.histplot(data=df, x="Age", kde=True)
    plt.title("Distribuição de Idade")
    plt.xlabel("Idade")
    plt.ylabel("Frequência")
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlação entre variáveis")
    plt.show()


# === NOVO — EXTRAI NOMES DAS FEATURES DO PREPROCESSOR ===

def get_feature_names(preprocessor, numeric_features, categorical_features):
    output_features = []

    # Numéricas — mantêm nome original
    output_features.extend(numeric_features)

    # Categóricas — obter nomes do OneHotEncoder
    if categorical_features:
        ohe = preprocessor.named_transformers_['cat']
        ohe_names = list(ohe.get_feature_names_out(categorical_features))
        output_features.extend(ohe_names)

    return output_features


# === AJUSTADO — PLOT CORRETO DE FEATURES ===

def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        print("Este modelo não possui atributo feature_importances_.")
        return

    importance = model.feature_importances_
    if len(importance) != len(feature_names):
        print("Tamanho incompatível entre features e importâncias.")
        print(f"{len(importance)} importâncias vs {len(feature_names)} nomes")
        return

    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp['feature'], feature_imp['importance'])
    plt.title('Importância das Features')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.show()
