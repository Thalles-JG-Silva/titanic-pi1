from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Cria pipeline para transformar dados antes do modelo."""
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

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
        ("classifier", LogisticRegression(max_iter=200))
    ])
    model.fit(X, y)
    dump(model, "models/logistic_regression.joblib")
    return model

def train_random_forest(X, y, preprocessor):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X, y)
    dump(model, "models/random_forest.joblib")
    return model
