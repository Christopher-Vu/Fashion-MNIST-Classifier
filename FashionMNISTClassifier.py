import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

%matplotlib inline
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

epochs = 100
al_graph_step = 3 # show accuracy/loss graph every n epochs
wbstd_graph_step = 1 # show weight+bias standard deviation graph every n epochs
show_final_graphs = True

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Display sample data
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    idx = torch.randint(len(test_data), size=(1,)).item()
    img, label = test_data[idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct, test_loss # accuracy and avg loss

accuracies, avg_losses, epoch_ind = [], [], [i+1 for i in range(epochs)]

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    accuracy, avg_loss = test(test_dataloader, model)
    
    accuracies.append(accuracy)
    avg_losses.append(avg_loss * 1000)
    epoch_ind_con = epoch_ind[:len(accuracies)]
    
    if show_final_graphs or (t+1)%al_graph_step==0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epoch_ind_con, accuracies, color='skyblue', linewidth=2, label='Accuracy (%)')
        ax.plot(epoch_ind_con, avg_losses, color='salmon', linewidth=2, label='Average Loss (magnified by 1000x)')
        ax.fill_between(epoch_ind_con, accuracies, color='skyblue', alpha=0.3)
        ax.fill_between(epoch_ind_con, avg_losses, color='salmon', alpha=0.3)
        
        plt.style.use('dark_background')
        
        plt.title('Accuracy and Average Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Values')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    if show_final_graphs or (t+1)%wbstd_graph_step==0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epoch_ind_con, accuracies, color='skyblue', linewidth=2, label='Accuracy (%)')
        ax.plot(epoch_ind_con, avg_losses, color='salmon', linewidth=2, label='Average Loss (magnified by 1000x)')
        ax.fill_between(epoch_ind_con, accuracies, color='skyblue', alpha=0.3)
        ax.fill_between(epoch_ind_con, avg_losses, color='salmon', alpha=0.3)
        
        plt.style.use('dark_background')
        
        plt.title('Accuracy and Average Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Values')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

print("Done!")  

model = NeuralNetwork()
torch.save(model.state_dict(), "data/model.pth")

model.load_state_dict(torch.load("data/model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
