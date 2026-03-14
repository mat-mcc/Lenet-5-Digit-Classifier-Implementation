import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def showConfusionMatrix(preds, targets):
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=range(10)
    )

    disp.plot(cmap="viridis")
    plt.title("Confusion Matrix")
    plt.show()