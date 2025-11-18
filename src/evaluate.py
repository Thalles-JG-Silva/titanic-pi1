import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)


def evaluate_classification_model(model, X_test, y_test):
    """
    Exibe métricas de classificação completas.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("ACCURACY:", accuracy_score(y_test, y_pred))
    print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))

    return y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred):
    """
    Plota a matriz de confusão.
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.show()


def plot_roc_curve(y_test, y_proba):
    """
    Plota a curva ROC.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Curva ROC")
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.legend()
    plt.show()


def plot_precision_recall(y_test, y_proba):
    """
    Plota a curva Precision-Recall.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.plot(recall, precision)
    plt.title("Curva Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
