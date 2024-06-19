import copy
import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn
from pytorch.FER2013Dataset import train_loader, val_loader, test_loader, emotions
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from pytorch.model.AlexNet import AlexNet
from pytorch.model.DarkCovidNet import DarkCovidNet
from pytorch.model.MobileNetV2 import MobileNetV2
from pytorch.model.VGG13 import VGG13
import seaborn as sns


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, device, num_epochs=1):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs

        # Initialize the best model weights
        self.best_model_wts = copy.deepcopy(model.state_dict())

        # Initialize TensorBoard writer
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', model.__class__.__name__, current_time)
        self.writer = SummaryWriter(log_dir)

    def train_model(self):
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    data_loader = self.train_loader
                else:
                    self.model.eval()
                    data_loader = self.val_loader

                running_loss = 0.0
                running_corrects = 0

                for batch_idx, (inputs, labels) in enumerate(data_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = running_corrects.float() / len(data_loader.dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Record the loss and accuracy to TensorBoard
                self.writer.add_scalars('loss', {phase: epoch_loss}, epoch)
                self.writer.add_scalars('accuracy', {phase: epoch_acc}, epoch)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

        self.writer.close()

        print(f'Best val Acc: {best_acc:.4f}')
        self.model.load_state_dict(self.best_model_wts)

    def save_model(self, path, script, example):
        if script:  # If script is True, save a scripted version of the model
            original_device = next(self.model.parameters()).device  # Store the original device
            self.model.to(torch.device('cpu'))  # Switch to CPU
            traced_model = torch.jit.trace(self.model, example)
            traced_model.save(path)
            self.model.to(original_device)  # Switch back to the original device
        else:  # If script is False, save the model normally
            torch.save(self.model.state_dict(), path)

    def show_predictions(self, test_loader, emotions):
        # Display the confusion matrix
        self.show_confusion_matrix(test_loader)
        # Fetch images and labels from the test loader
        images, labels = next(iter(test_loader))

        # Ensure the image tensor is on CPU before converting it to numpy array
        images = images.cpu()

        # Get model predictions
        samples = images.to(self.device)
        outputs = self.model(samples)
        _, preds = torch.max(outputs, 1)

        fig, axs = plt.subplots(2, 5, figsize=(25, 10))

        for i, ax in enumerate(axs.flat):
            # Normalize the image array for visualization
            img = images[i]  # fetch the i-th image from the batch
            img = np.transpose(img.numpy(), (1, 2, 0))
            img = (img * 0.5) + 0.5

            # Display image along with its predicted and actual labels
            ax.imshow(img)
            pred_label = emotions[preds[i].item()]
            true_label = emotions[labels[i].item()]
            ax.set_title(f'Pred: {pred_label}, True: {true_label}', fontsize=20)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.show()

    def show_confusion_matrix(self, test_loader):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Print the confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print('Confusion Matrix: ')
        print(cm)

        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, ax=ax)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.show()


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")


model = DarkCovidNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = ModelTrainer(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=30)
trainer.train_model()

trainer.show_predictions(test_loader, emotions)

# export model for app,mps train export cpu
example = torch.randn(1, 3, 224, 224).to(torch.device('cpu'))
save_dir = "export"
model_name = "unet_"+model.__class__.__name__ + ".pt"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(os.path.join(save_dir, model_name), script=True, example=example)
