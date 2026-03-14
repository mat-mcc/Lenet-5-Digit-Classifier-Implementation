import torch
import torch.nn as nn


def train(model, device, trainLoader, optimizer, epoch):

    model.train()
    lossFunction = nn.CrossEntropyLoss()

    correct = 0
    totalLoss = 0

    for data, target in trainLoader:

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossFunction(output, target)
        loss.backward()
        optimizer.step()
        totalLoss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    avgLoss = totalLoss / len(trainLoader.dataset)
    accuracy = 100 * correct / len(trainLoader.dataset)
    print(f"Epoch {epoch} | Loss: {avgLoss:.4f} | Accuracy: {accuracy:.2f}%")

    return avgLoss, accuracy


def test(model, device, testLoader):

    model.eval()
    lossFunction = nn.CrossEntropyLoss(reduction="sum")

    testLoss = 0
    correct = 0

    allPreds = []
    allTargets = []

    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += lossFunction(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            allPreds.extend(pred.cpu().numpy())
            allTargets.extend(target.cpu().numpy())

    testLoss /= len(testLoader.dataset)
    accuracy = 100 * correct / len(testLoader.dataset)
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy, allPreds, allTargets