# -*- coding: utf-8 -*-
"""Copy of Gavin-Kai Vida - PyTorch Fundamentals Module 1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-pxoDql5xTzy8Yi17eGqC9SyP6HtU0vR

Machin workflows involve working with data, creating models, using hyperparameters to optimize models, and saving and inferencing with those models.

Tensors are arrays with any dimensionality.
"""

import torch
import numpy as np

#can be created from normal dataytypes
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

#can be created from numpy arrays
data = np.array(data)
x_data = torch.from_numpy(data)

#tensor created from other

shape = (2,3)
rand_tensor= torch.rand(shape)

#tensors can be moved to gpu processing
#if torch.cuda.is_available():
#  tensor = torch.tensor.to('cuda')

ones = torch.ones(4,4)
print(ones[0])
print(ones[:,0])
print(ones[...,-1])
ones[:,1]=0
print(ones)

#concatenating torches
t1 = torch.cat([ones,ones,ones],dim=1)
t2 = torch.cat([ones,ones,ones],dim=0)
print(t1)
print(t2)

"""Tensor multiplication is like standard numpy syntax. There is also syntax that bridges numpy and tensors

### Datasets and dataloaders
Pytorch provides DataLoader and Dataset. Dataset stores samples and labels and wraps an iterable around the Dataset to enable easy access to the samples.

PyTorch also offers prefloaded datasets for prototyping an dbenchmarking (image, text, and audio)
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

'''
root is the path where the train/test data is stored.
train specifies training or test dataset.
download=True downloads the data from the Internet if it's not available at root.
transform and target_transform specify the feature and label transformations.
'''

print(training_data[0])
#for examination of the structure of a single datapoint

sample_idx = torch.randint(len(training_data),size=(9,))
print(sample_idx)

print(sample_idx[0].item())

"""Iteration and visualization"""

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
} #used to label subplots
figure = plt.figure(figsize=(8, 8)) #initiation
cols, rows = 3, 3 #used for loop calculation and subplot labelling
for i in range(1,cols*rows+1):
  img,label = training_data[sample_idx[i-1].item()] #select a random item in training_data from list of random numbers generated earlier
  figure.add_subplot(rows,cols,i)
  plt.title(labels_map[label])
  plt.axis("off")
  plt.imshow(img.squeeze(), cmap="gray")
plt.show()

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
#batch_size, number of datapoints/batch; shuffle -> ranodmly sample data by indices

"""Convert dataloader to iterable and iterate through it."""

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
print(img)
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
label_name = list(labels_map.values())[label]
print(f"Label: {label_name}")

"""#Normalization
Scales data to ensure equal weight on all features/datapoints
"""

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
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

"""nn.Module -> provides useful methods for neural networks and automatically does what is defined in the foward method above

nn.Flatten -> flattens input to 1D tensor.

nn.Sequential -> defines the sequences of functions that will be applied to the initial n=784 tensor

nn.Linear -> first parameter: number of input_features, second_parameter: number of output features (comes built in with biases and weights)
"""

# model = NeuralNetwork().to(device) #moves calculations of this nn instance to device defined earlier
# print(model)

# x = torch.rand(1,28,28,device=device) #creates a 1x28x28 tensor with 784 randomly distributed numbers between 0 and 1
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits) #converts raw ouput to probability
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

"""torch.autograd -> automatically calculates gradients during trainig

When making tensors, you may use the parameter "requires_grad" to compute the gradients of tensors derived from that tensor later on. The functoin that computes this gradient is stored in grad_fn of tensors.

The "backward" method of a tensor computes the gradients. The appriopriate coeffecients for parameters can then be accessed through their corresponding tensors with the "grad" method.

"""

import torch
import numpy as np

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

loss.backward() #loss is derived from w and b. So the backward method calculates gradients with respect to those variables
print(w.grad) #gradients can then be accessed with grade method
print(b.grad)
weights = w.grad.numpy()
biases = b.grad.numpy()
output = np.dot(weights,biases)
print(output)

"""Gradient tracking can be disabled with the detach method or with the torch.no_grad() block.

You may want to disable parameters in your program when dealing with frozen parameters (important for pre-trained networks) or for optimization of forward-passes

Autograd uses these methods to automatically calculate gradients.

Hyperparameters -> parameters that control the optimization process
"""

learning_rate = 1e-3
batch_size = 32
epochs = 5

"""#Optimization Loop
Each epoch consists of training loop (converge on optimized parameters)
Testing loop check network performance and monitor improvements

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) #stochastic gradient descent
print(model.parameters())
"""

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward() #calculates and aggregates
        optimizer.step()

        if batch % 100 == 0: #periodically print loss
            optimizer.step() #applies gradients
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") #resets gradients

def test_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    state = optimizer.state_dict()["state"]

    contributions = [np.zeros(shape=(state[i]["momentum_buffer"].shape[0])) for i in range(0,len(state)-2,2)]
    for X, y in dataloader:
        pred = model(X)
        
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        
        test_loss += loss.item() #accumulates loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        state = optimizer.state_dict()["state"]
        for i,layer in enumerate(contributions):
            #print("layer",layer.shape)
            #print("buffer",weights:=torch.abs(state[i*2]["momentum_buffer"]).sum(1).squeeze().shape.numpy())
            #print("biases",biases:=state[(i*2+1)]["momentum_buffer"].numpy())
            weights=torch.abs(state[i*2]["momentum_buffer"]).sum(1).squeeze().numpy()
            biases=torch.abs(state[(i*2+1)]["momentum_buffer"]).numpy()
            #print(layer.shape,weights.shape,biases.shape)
            #print(weights,biases)
            contributions[i] = (contributions[i]+weights+biases)
            #print(layer)

    test_loss /= size #averages loss
    correct /= size #finds accuracy percent
    #state = optimizer.state_dict()["state"]
    #print([layer for layer in contributions])
    #contributions =[torch.abs(state[i]["momentum_buffer"]).sum(1).squeeze() for i in range(0,len(state)-2,2)]
    #print([layer for layer in contributions])

    # distribution = np.dot(weights,biases).squeeze()
    # print(distribution)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return contributions, correct

model = NeuralNetwork()


# for name, param in model.named_parameters():
#   print(name, param)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum = 0.5)


variations = np.array([])
accuracies = np.array([])
epochs = 1000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    contributions, accuracy = test_loop(test_dataloader, model, loss_fn,optimizer)
    variations = np.append(variations,np.std(contributions))
    accuracies = np.append(accuracies,accuracy)
    
    plt.scatter(accuracies,variations)
    p = np.polyfit(accuracies,variations,2)
    plt.plot(accuracies,p[0]*accuracies**2+p[1]*accuracies**1+p[2])
    plt.show()
    print((package := np.concatenate(contributions)).shape)
    plt.hist(package, color='lightgreen', ec='black')
    plt.show()
    

    circle_radius = 3
    x=0; y=0
    plt.axis([0,circle_radius * 3 * len(contributions),0,circle_radius*3*max([len(layer) for layer in contributions])])
    #plt.axis("equal")
    for layer in contributions:
        y = 0
        x += 2*circle_radius
        biggest = max(layer)
        for neuron in layer:
            y += 3*circle_radius
            strength = (neuron/biggest).item()
            plt.gca().add_artist(plt.Circle((x,y),radius=circle_radius,color=(strength,0,1-strength)))
    plt.show()
            
        
    
    # print(optimizer.state_dict()["state"])
    # for i in range(1,len(optimizer.state_dict()["state"].values())/2-1):
    #   print(optimizer.state_dict()["state"].values()[i]["momentum_buffer"])


print("Done!")



torch.save(model.state_dict(),"data/model.pth") #stores learned parameters in specified path
print("Saved PyTorch Model state to model.pth")

"""Model can also be loaded from saved parametesr

"""

model = NeuralNetwork()
model.load_state_dict(torch.load('data/model.pth'))

"""Things to do:
- Identify plateau in neural network
- Graph variation of contributions with respect to accuracy and regress(are neural networks with a more diverse set of weight gradients more accurate? How strong is this relationship)
- Graph variation of contritbution of each neuron across samples (How sensitive is the contribution fo a neuron to the structure of data being passed through it? Is individual neuron contribution consistent, and therefore an actually reliable measurement of the importance of of a neuron)
  - Average neuron variations and plot with respect to accuracy and regret
- Remove varying proportions of neurons based on contribution measurement and observe loss/gain in accuracy and loss/gain in computational cost (how strong is the relationship? regress!)

"""