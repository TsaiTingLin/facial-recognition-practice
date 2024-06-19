import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from matplotlib import pyplot as plt
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import os
from datetime import datetime
from pytorch.pretrain.UNet import UNet


class FacialRecognition:
    def __init__(self, filename, num_classes=7, start_neurons=64):
        self.filename = filename
        self.num_classes = num_classes
        self.start_neurons = start_neurons
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        self.label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.names = ['emotion', 'pixels', 'usage']
        self.df = pd.read_csv(filename, names=self.names, na_filter=False)

        # Assuming you load the data in the constructor
        self.X, self.Y = self._get_data()
        self.X = self.X.reshape(self.X.shape[0], 48, 48, 1)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42, shuffle=True)
        self.x_train = self.x_train[:5000]
        self.x_test = self.x_test[:5000]

        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.

        self.noise_factor = 0.1
        self.x_train_noisy = self.x_train + self.noise_factor * np.random.normal(loc=0.0, scale=1.0,
                                                                                 size=self.x_train.shape)
        self.x_test_noisy = self.x_test + self.noise_factor * np.random.normal(loc=0.0, scale=1.0,
                                                                               size=self.x_test.shape)

        self.x_train_noisy = np.clip(self.x_train_noisy, 0., 1.)
        self.x_test_noisy = np.clip(self.x_test_noisy, 0., 1.)

        # Convert data to PyTorch tensors and create DataLoader
        self._prepare_data()

        # Initialize UNet model
        self.model_unet = self._initialize_model()

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.model_unet.parameters())
        self.criterion = nn.MSELoss()

        self.predictions = []
        # Initialize TensorBoard writer
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('../runs', self.model_unet.__class__.__name__, current_time)
        self.writer = SummaryWriter(log_dir)

    def _get_data(self):
        Y = []
        X = []
        first = True
        for line in open(self.filename):
            if first:
                first = False
            else:
                row = line.split(',')
                Y.append(int(row[0]))
                X.append([int(p) for p in row[1].split()])

        X, Y = np.array(X), np.array(Y)
        return X, Y

    def _prepare_data(self):
        self.x_train_noisy = torch.from_numpy(self.x_train_noisy.transpose(0, 3, 1, 2)).float()
        self.x_train = torch.from_numpy(self.x_train.transpose(0, 3, 1, 2)).float()
        self.x_test_noisy = torch.from_numpy(self.x_test_noisy.transpose(0, 3, 1, 2)).float()
        self.x_test = torch.from_numpy(self.x_test.transpose(0, 3, 1, 2)).float()

        self.train_dataset = TensorDataset(self.x_train_noisy, self.x_train)
        self.test_dataset = TensorDataset(self.x_test_noisy, self.x_test)

        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)



    def _initialize_model(self):
        model = UNet(in_channels=1, start_neurons=self.start_neurons)
        model.to(self.device)
        return model

    def train(self, epochs=1):
        self.model_unet.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model_unet(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.writer.add_scalars('loss', {'epoch_loss': epoch_loss}, epoch)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
        self.writer.close()

    def evaluate(self):
        self.model_unet.eval()
        with torch.no_grad():
            for inputs, _ in self.test_loader:
                outputs = self.model_unet(inputs.to(self.device))
                self.predictions.append(outputs.cpu().numpy())

        self.predictions = np.concatenate(self.predictions, axis=0)

        # Calculate PSNR
        x_test_numpy = self.x_test.cpu().numpy()
        x_test_numpy = x_test_numpy.astype(np.float32)
        predictions_numpy = self.predictions.astype(np.float32)

        total_psnr = 0.0
        num_images = len(x_test_numpy)
        for original_image, generated_image in zip(x_test_numpy, predictions_numpy):
            psnr_value = compare_psnr(original_image, generated_image)
            total_psnr += psnr_value

        average_psnr = total_psnr / num_images
        print(f"Average PSNR: {average_psnr:.2f} dB")

    def plot_images(self, n=10):
        indices = random.sample(range(len(self.x_test)), n)
        plt.figure(figsize=(60, 60))
        plt.suptitle("Comparison of Original, Noised, and Generated Test Images")

        plt.subplot(3, n, 1)
        plt.title("Original", fontsize=22)
        for i in range(n):
            plt.subplot(3, n, i + 1)
            plt.imshow(self.x_test[indices[i]].reshape(48, 48))
            plt.gray()
            plt.axis('off')

        plt.subplot(3, n, n + 1)
        plt.title("Noised", fontsize=22)
        for i in range(n):
            plt.subplot(3, n, n + i + 1)
            plt.imshow(self.x_test_noisy[indices[i]].reshape(48, 48))
            plt.gray()
            plt.axis('off')

        plt.subplot(3, n, 2 * n + 1)
        plt.title("Generated", fontsize=22)
        for i in range(n):
            plt.subplot(3, n, 2 * n + i + 1)
            plt.imshow(self.predictions[indices[i]].reshape(48, 48))
            plt.gray()
            plt.axis('off')

        plt.show()


# Usage example:
filename = '../../fer2013.csv'
facial_recognition = FacialRecognition(filename)
facial_recognition.train(epochs=50)
facial_recognition.evaluate()
facial_recognition.plot_images()
torch.save(facial_recognition.model_unet.state_dict(), 'unet_model_trained.pth')
