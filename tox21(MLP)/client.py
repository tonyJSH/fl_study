import deepchem as dc
import numpy as np
import torch
from collections import OrderedDict
import flwr as fl
import sys


client_num = int(sys.argv[1])
x_train = np.load(f'./data/client{client_num}/x_train.npy')
y_train = np.load(f'./data/client{client_num}/y_train.npy')
x_val = np.load(f'./data/client{client_num}/x_val.npy')
y_val = np.load(f'./data/client{client_num}/y_val.npy')

train_set = dc.data.DiskDataset.from_numpy(x_train, y_train)
val_set = dc.data.DiskDataset.from_numpy(x_val, y_val)

model_fl = dc.models.MultitaskClassifier(n_tasks=12, n_features=1024, layer_sizes=[1000])

metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, val_dataset, metric):
        self.model = model
        self.trainset = train_dataset
        self.valset = val_dataset
        self.metric = metric

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.model.state_dict().items()] # 모델의 파라미터 반환
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters) # 서버에서 받은 parameters 모델 적용
        self.model.fit(self.trainset, nb_epoch=1) # 모델 학습
        train_roc = self.model.evaluate(self.trainset, [self.metric])
        return self.get_parameters(config={}), len(self.trainset), train_roc # 필수 반환

    def evaluate(self, parameters, config):
        self.set_parameters(parameters) # 서버에서 받은 parameters 모델 적용
        val_roc = self.model.evaluate(self.valset, [self.metric])
        return 0.1, len(self.valset), val_roc # 필수 반환 (loss값은 임의로 0.1 입력)
    
fl_client = FlowerClient(model_fl, train_set, val_set, metric)

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=fl_client)
