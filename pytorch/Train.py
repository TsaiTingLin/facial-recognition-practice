import copy
import torch
from pytorch.FER2013Dataset import device, train_loader, val_loader
from EmotionNet import model, criterion, optimizer


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if batch_idx % 100 == 0:
                    batch_acc = torch.sum(preds == labels.data).float() / len(inputs)  # 将准确率计算转换为float32
                    print(
                        f'[{epoch}/{num_epochs - 1}] [{phase}] Batch {batch_idx}/{len(data_loader)} Loss: {loss.item():.4f} Acc: {batch_acc:.4f}')

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)
# 在 Python 中
model.eval()  # 確保模型在評估模式
example = torch.rand(1, 3, 224, 224)  # 依據您的模型輸入調整
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("mobilenet_v2_model.pt")
