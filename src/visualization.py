import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_survival_count(df):
    sns.countplot(data=df, x="Survived")
    plt.title("Distribuição de Sobrevivência")
    plt.show()

def plot_age_distribution(df):
    sns.histplot(df["Age"], kde=True)
    plt.title("Distribuição de Idades")
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=False, cmap="Blues")
    plt.title("Correlação entre Variáveis Numéricas")
    plt.show()
