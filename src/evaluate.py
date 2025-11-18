from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    print("\nACCURACY:", accuracy_score(y_test, preds))
    print("\nCLASSIFICATION REPORT:\n")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()
