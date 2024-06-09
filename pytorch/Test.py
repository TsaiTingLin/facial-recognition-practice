import torch

from pytorch.EmotionNet import model
from pytorch.FER2013Dataset import device, test_loader


def evaluate_model(model, test_loader):
    model.eval()
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {acc:.4f}')

evaluate_model(model, test_loader)