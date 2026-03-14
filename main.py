import torch
import os
import torch.optim as optim

from lenet_models import LeNet5, ModifiedLeNet5
from dataset_loader import loadData
from trainer import train, test
from metrics import showConfusionMatrix

import draw_digit_ui


def runModel(model, modelPath, optimizer, trainLoader, testLoader, device):

    if os.path.exists(modelPath):
        print("Pretained Model Exists! Loading pretrained model...\n")
        model.load_state_dict(torch.load(modelPath, weights_only=True))
        print("Launching UI\n")
        draw_digit_ui.main()

    else:
        print("Pretrained model does NOT exist! Training model...")

        for epoch in range(1, 11):
            train(model, device, trainLoader, optimizer, epoch)

        torch.save(model.state_dict(), modelPath)
        print("Model saved.")
        showConfusionMatrix(preds, targets)

    accuracy, preds, targets = test(model, device, testLoader)



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainLoader, testLoader = loadData()
    """
    print("Running Original LeNet")
    model1 = LeNet5().to(device)
    optimizer1 = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)

    runModel(model1,"saved_models/original_lenet5.pth",optimizer1,trainLoader,testLoader,device)
    """
    print("Running Improved LeNet Model\n")
    model2 = ModifiedLeNet5().to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    runModel(model2,"saved_models/modified_lenet5.pth",optimizer2,trainLoader,testLoader,device)

if __name__ == "__main__":
    main()