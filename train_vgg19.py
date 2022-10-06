from torch import nn
import numpy as np
from dataset import Covid_dataset
from torch.utils.data import DataLoader
import torch
from model import VGG_19
import torchvision.transforms as trans

mytransformer = trans.Resize([224,224])
# prepare both datasets

whole_dataset = Covid_dataset("/home/jas0n/PycharmProjects/covid_ct/COVID-CT/Data-split/non_covid.csv",
                              "/home/jas0n/PycharmProjects/covid_ct/COVID-CT/Data-split/covid.csv",
                              "/home/jas0n/PycharmProjects/covid_ct/COVID-CT/all_image_resized",
                              )
training_ratio = 0.8
batch_size = 16
epochs = 50
train_data, test_data = torch.utils.data.random_split(whole_dataset, [int(len(whole_dataset) * training_ratio),
                                                                      len(whole_dataset)-int(len(whole_dataset) * training_ratio)])
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
plot_loss = []

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
#prepare the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model_vgg = VGG_19().to(device)
print(model_vgg)

#define train and test function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_vgg.parameters(), lr=1e-2)
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.float()

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.float()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    plot_loss.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
test(test_dataloader, model_vgg, loss_fn)
test(train_dataloader, model_vgg, loss_fn)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model_vgg, loss_fn, optimizer)
    test(test_dataloader, model_vgg, loss_fn)
    test(train_dataloader, model_vgg, loss_fn)
    torch.save(model_vgg,'./weights/vgg/{}.pth'.format(t))
np.save("./loss/vgg19.npy",plot_loss)
print("Done!")
