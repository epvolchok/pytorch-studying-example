import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from PIL import Image

import os
import imghdr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#data loading

transforms = transforms.Compose([
    # resize an image to a standart size of 64x64
    # to accelerate clcultions on gpu
    transforms.Resize((64, 64)),
    # translate to a tensor form
    transforms.ToTensor(),
    # normalization to avoid the problem of the exploding gradients
    # these are the mean and std through images of ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

def is_valid_image(file_path):
    """
    Проверяет, является ли файл валидным изображением.
    
    Args:
        file_path (str): Путь к файлу.

    Returns:
        bool: True, если файл валиден, иначе False.
    """
    # Проверка расширения
    allowed_extensions = (".jpg", ".jpeg", ".png")
    if not file_path.lower().endswith(allowed_extensions):
        return False

    # Проверка размера файла
    min_size = 1  # байт
    if os.path.getsize(file_path) < min_size:
        return False

    # Проверка формата изображения
    if imghdr.what(file_path) not in ["jpeg", "png"]:
        return False

    return True

#data for learning
train_data_path = './train'
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transforms, is_valid_file=is_valid_image)

#data fro validation
val_data_path = './val/'
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=transforms, is_valid_file=is_valid_image)

#data to test the model
test_data_path = './test/'
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transforms, is_valid_file=is_valid_image)

def datasets_test(train_data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        img = img.permute(1,2,0)
        plt.imshow(img, cmap="gray")
    plt.show()



batch_size = 64
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

def test_dataloader(train_data_loader):
    X, y = next(iter(train_data_loader))

    img = X[0].permute(1,2,0)
    plt.imshow(img, cmap="gray")
    plt.show()

# simple model
class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()
        # describe the model
        self.model_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*64*64, 84),
            nn.ReLU(),
            nn.Linear(84, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            # to categorize we use Cross EntropyLoss(), it already includes softmax()
            #nn.Softmax(),
        )
        

    def forward(self, x):
        results = self.model_stack(x)
        return results

simplenet=SimpleNet()
# optimizer
# Adam is with a varying learning rate
optimizer = optim.Adam(simplenet.parameters(), lr=0.001)

# CrossEntropyLoss combines nn.NLLLoss (Negative Log Likelihood) - loss func for classifications
# and nn.LogSoftmax
loss_fn = nn.CrossEntropyLoss()

train_size = len(train_data_loader.dataset)
val_size = len(val_data_loader.dataset)

# learning
def train_loop(train_loader, model, optimizer, loss_fn):
    
    # Unnecessary in this situation but added for best practices
    model.train()

    for batch, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * batch_size + len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")

def validation_loop(val_loader, model, loss_fn):

    model.eval()

    test_loss, correct = 0, 0
    num_batches = len(val_loader)
    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= val_size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_data_loader, simplenet, optimizer, loss_fn)
    validation_loop(val_data_loader, simplenet, loss_fn)
print("Done!")

labels = ['cat','fish']

img = Image.open('./test/fish/wilderness_beach.jpg')

img = transforms(img)
img = img.unsqueeze(0)
prediction = simplenet(img)
prediction = prediction.argmax()
print(labels[prediction])