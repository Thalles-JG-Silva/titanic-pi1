from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Cria pipeline para transformar dados antes do modelo."""
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

def train_logistic_regression(X, y, preprocessor):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])
    model.fit(X, y)
    dump(model, "models/logistic_regression.joblib")
    print("Regressão Logística treinada e salva.")
    return model

def train_random_forest(X, y, preprocessor):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    dump(model, "models/random_forest.joblib")
    print("Random Forest treinada e salva.")
    return model