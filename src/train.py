# src/train.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, clean_data, encode_data
from src.models import create_preprocessing_pipeline, train_logistic_regression, train_random_forest
from src.evaluate import evaluate_classification_model, plot_confusion_matrix, plot_roc_curve, plot_precision_recall
from src.visualization import plot_feature_importance, get_feature_names

def main():
    df = load_data("data/train.csv")
    print("Dados carregados. Shape:", df.shape)

    df = clean_data(df)
    print("Dados limpos. Shape:", df.shape)

    df = encode_data(df)
    print("Dados codificados.")

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    print("Features numéricas:", numeric_features)
    print("Features categóricas:", categorical_features)

    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

    print("\n=== Treinando Regressão Logística ===")
    lr_model = train_logistic_regression(X_train, y_train, preprocessor)

    print("\n=== Treinando Random Forest ===")
    rf_model = train_random_forest(X_train, y_train, preprocessor)

    print("\n=== Avaliação Regressão Logística ===")
    y_pred_lr, y_proba_lr = evaluate_classification_model(lr_model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred_lr)
    plot_roc_curve(y_test, y_proba_lr)
    plot_precision_recall(y_test, y_proba_lr)

    print("\n=== Avaliação Random Forest ===")
    y_pred_rf, y_proba_rf = evaluate_classification_model(rf_model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred_rf)
    plot_roc_curve(y_test, y_proba_rf)
    plot_precision_recall(y_test, y_proba_rf)

    print("\n=== Importância das Features (Random Forest) ===")
    rf_classifier = rf_model.named_steps['classifier']

    # ✨ Obter nomes reais das features
    feature_names = get_feature_names(
        preprocessor,
        numeric_features,
        categorical_features
    )

    plot_feature_importance(rf_classifier, feature_names)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
