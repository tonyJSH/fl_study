import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
import flwr as fl
import math
import sys

    
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0') # 해당 조의 GPU 번호로 변경 ex) 1조 : cuda:1
else:
    DEVICE = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.dense2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x

def train(model, epoch, train_loader, optimizer, log_interval, loss_fn):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                                                    epoch, batch_idx * len(image), 
                                                    len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                                                    loss.item()))
            
def evaluate(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += loss_fn(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= (len(test_loader.dataset) / BATCH_SIZE)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


train_set = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = True,
    download = True,
    transform = transfroms.Compose([
        transfroms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)
test_set = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = False,
    download = True,
    transform = transfroms.Compose([
        transfroms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

BATCH_SIZE = 128

def list_split(arr, n):
    num = math.trunc(len(arr) / n)
    return [arr[i: i + num] for i in range(0, len(arr), num)]

def non_iid_split(client_num:int, ratio:float, dataset):
    client_x = [[] for i in range(client_num)]
    client_y = [[] for i in range(client_num)]
    for i in range(client_num):
        data = dataset.data[dataset.targets == i]
        targets = dataset.targets[dataset.targets == i]
        ratio_num = math.ceil(ratio * len(targets))
        client_x[i].append(data[:ratio_num])
        client_y[i].append(targets[:ratio_num])
        if len(data[ratio_num:]) / (client_num - 1) != 0:
            data_tail = list_split(data[ratio_num:], client_num -1)
            targets_tail = list_split(targets[ratio_num:], client_num -1)
            for l in [n for n in range(client_num) if n != i]:
                client_x[l].append(data_tail.pop(0))
                client_y[l].append(targets_tail.pop(0))
    
    return [torch.concat(t, dim=0) for t in client_x], [torch.concat(t, dim=0) for t in client_y]

num_clients = 10

with open('ratio.txt', 'r') as f:
    r = float(f.readline())
    ratio = r

x_train_list, y_train_list = non_iid_split(num_clients, ratio, train_set)
x_test_list, y_test_list = non_iid_split(num_clients, ratio, test_set)

class MnistDataSet(torch.utils.data.Dataset):
    def __init__(self, images, labels, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        data = self.X[i, :]
        data = np.array(data).astype(np.float32).reshape(1, 28, 28)
        return (data, self.y[i])

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, tarinloader, testloader, opt, loss_fn):
        self.model = model
        self.train_loader = tarinloader
        self.test_loader = testloader
        self.optimizer = opt
        self.loss_fn = loss_fn

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters): # pytorch 모델에 파라미터를 적용하는 코드가 복잡하여 함수로 정의
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters) # 위에서 정의한 set_parameters함수를 사
        train(self.model, 1, self.train_loader, self.optimizer, 200, self.loss_fn)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.test_loader, self.loss_fn)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}
    
model_fl = MLP().to(DEVICE)
criterion_fl = nn.CrossEntropyLoss().to(DEVICE)
optimizer_fl = torch.optim.Adam(model_fl.parameters())

client_num = int(sys.argv[1])

train_dataset_fl = MnistDataSet(x_train_list[client_num], y_train_list[client_num])
test_dataset_fl = MnistDataSet(x_test_list[client_num], y_test_list[client_num])
BATCH_SIZE = 128
train_loader_fl = DataLoader(train_dataset_fl, batch_size=BATCH_SIZE, shuffle=True)
test_loader_fl = DataLoader(test_dataset_fl, batch_size=BATCH_SIZE, shuffle=True)

flwr_client = FlowerClient(model_fl, train_loader_fl, test_loader_fl, optimizer_fl, criterion_fl)

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=flwr_client)