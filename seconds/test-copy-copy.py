import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from torch import optim
import random
import sklearn.datasets
import matplotlib.pyplot as plt
import time, json, pathlib
import torchvision.models
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
# from sklearn.datasets import make_regression




class Corr_loss(nn.Module):
    def forward(self, x, y):

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        # cost = (torch.sum(vx * vy)**2 + 1e-10) / ((torch.sum(vx ** 2))*(torch.sum(vy ** 2))+ 1e-10)
        # cost = (torch.sum(vx * vy) ) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1)
        # cost = (torch.sum(vx * vy) ) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1)

        cost = torch.sum(vx * vy)
        # cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2))
        #                              * torch.sqrt(torch.sum(vy ** 2)))
        # cost = (torch.sum(vx * vy) / torch.sqrt(torch.sum(vx ** 2))) * (torch.sum(vx * vy) / torch.sqrt(torch.sum(vy ** 2)))
        return cost
class Corvari_loss(nn.Module):
    def forward(self, x, y):

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        # cost = (torch.sum(vx * vy)**2 + 1e-10) / ((torch.sum(vx ** 2))*(torch.sum(vy ** 2))+ 1e-10)
        # cost = (torch.sum(vx * vy) ) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1)
        # cost = (torch.sum(vx * vy) ) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1)

        cost = torch.sum(vx * vy)
        # cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2))
        #                              * torch.sqrt(torch.sum(vy ** 2)))
        # cost = (torch.sum(vx * vy) / torch.sqrt(torch.sum(vx ** 2))) * (torch.sum(vx * vy) / torch.sqrt(torch.sum(vy ** 2)))
        return cost
class lastModule_(nn.Module):
    def __init__(self,in_features,out_features,ensamble = 4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d = {}

        self.relu = nn.ReLU()
        for i in range(out_features):

            self.d[i] = [nn.Sequential(
                    nn.Linear(in_features,in_features),
                    self.relu,
                    nn.Linear(in_features,1),
                    # nn.BatchNorm1d(1),

                    ) for _ in range(ensamble)]
        for i in range(out_features):
            for j in range(ensamble):
                setattr(self,f"model_{str(i)}_{str(j)}",self.d[i][j])
        self.in_features = in_features
        self.out_features = out_features
        self.ensamble = ensamble
        # self.last_bn = nn.BatchNorm1d(self.ensamble)
    def forward(self,X):
        ret_X = []
        for i in range(self.out_features):
            X_list = [fc(X) for fc in self.d[i]]
            rr = torch.concat(X_list,dim=-1)
            ret_X.append(rr)
        ret = torch.stack(ret_X,dim=-1).squeeze()
        # ret = self.last_bn(ret)
        # print(ret.shape)
        # ret = ret[:,0,:,:]
        # r
        return ret
class shallow_MLP(nn.Module):
    def __init__(self, input_size, output_size = 1,ensamble = 10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc1_1 = nn.Linear(128, 128)
        self.fc1_2 = nn.Linear(128, 128)
        self.fc1_3 = nn.Linear(128, 128)


        self.last = lastModule_(128,output_size,ensamble)
        self.weight_layer = nn.Sequential(nn.ReLU(),nn.Linear(128,ensamble))
        self.bn = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn(x)
        x = torch.relu(self.fc1_1(x))
        x = self.bn2(x)
        x = torch.relu(self.fc1_2(x))
        x = torch.relu(self.fc1_3(x))
        return X
# class shallow_CNN(nn.Module):
    # def __init__(self)

class large_lastModule_(nn.Module):
    def __init__(self,in_features,out_features,ensamble = 4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d = {}

        self.relu = nn.ReLU()
        for i in range(out_features):

            self.d[i] = [nn.Sequential(
                    nn.Linear(in_features,in_features),
                    self.relu,
                    nn.Linear(in_features,in_features),
                    self.relu,
                    nn.Linear(in_features,in_features),
                    self.relu,                
                    nn.Linear(in_features,in_features),
                    self.relu,
                    nn.Linear(in_features,1),
                    # nn.BatchNorm1d(1),

                    ) for _ in range(ensamble)]
        for i in range(out_features):
            for j in range(ensamble):
                setattr(self,f"model_{str(i)}_{str(j)}",self.d[i][j])
        self.in_features = in_features
        self.out_features = out_features
        self.ensamble = ensamble
        # self.last_bn = nn.BatchNorm1d(self.ensamble)
    def forward(self,X):
        ret_X = []
        for i in range(self.out_features):
            X_list = [fc(X) for fc in self.d[i]]
            rr = torch.concat(X_list,dim=-1)
            ret_X.append(rr)
        ret = torch.stack(ret_X,dim=-1).squeeze()
        # ret = self.last_bn(ret)
        # print(ret.shape)
        # ret = ret[:,0,:,:]
        # r
        return ret


class No_share_MLP(nn.Module):
    def __init__(self,input_size, output_size = 1,ensamble = 10) -> None:
        super().__init__()
        self.last = large_lastModule_(input_size,output_size,ensamble)
        self.weight_layer = large_lastModule_(input_size,out_features=ensamble,ensamble=1)
    def forward(self, x):
        return self.last(x),self.weight_layer(x)


class NCL(nn.Module):
    def __init__(self,input_dim = 128,output_size = 1,ensamble = 16) -> None:
        super().__init__()
        self.last = lastModule_(128,output_size,ensamble)
        self.weight_layer = nn.Sequential(nn.ReLU(),nn.Linear(128,ensamble))
    def forward(self,x):
        ret = self.last(x)
        weight = self.weight_layer(x)
        return ret,weight


class MLP(nn.Module):
    def __init__(self, input_size, output_size = 1,ensamble = 10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc1_1 = nn.Linear(128, 128)
        self.fc1_2 = nn.Linear(128, 128)
        self.fc1_3 = nn.Linear(128, 128)


        self.last = lastModule_(128,output_size,ensamble)
        # self.weight_layer = nn.Sequential(nn.Linear(128,ensamble),)
        self.weight_layer = nn.Sequential(nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,ensamble))
        
        self.bn = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn(x)
        x = torch.relu(self.fc1_1(x))
        x = self.bn2(x)
        x = torch.relu(self.fc1_2(x))
        x = self.bn3(x)
        x = torch.relu(self.fc1_3(x))

        ret = self.last(x)
        weight = self.weight_layer(x)
        return ret,weight

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize images to match ResNet input size
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the data
])




class CNN(nn.Module):
    def __init__(self,input_size = None,output_size = 1,ensamble = 10):
        super(CNN, self).__init__()
        # self.fc1 = nn.Linear(input_size, 128)
        # self.fc1_1 = nn.Linear(128, 128)
        # self.fc1_2 = nn.Linear(128, 128)
        # self.fc1_3 = nn.Linear(128, 128)
        #here you load resnet model.
        self.resnet = torchvision.models.resnet18(pretrained=True,)
        self.last = lastModule_(1000,output_size,ensamble)
        self.weight_layer = nn.Sequential(nn.ReLU(),nn.Linear(1000,ensamble))

    def forward(self, x):
        # x = self.resnet(x)
        
        with torch.no_grad():
            x = self.resnet(x)
        ret = self.last(x)
        weight = self.weight_layer(x)
        return ret,weight







class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Classifier, self).__init__()

        # Fully connected layer for classification
        self.relu = nn.ReLU()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):

        # Flatten the embedded representation
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layer
        x = self.relu(x)
        output = self.fc(x)

        # Apply a softmax activation to get class probabilities
        # class_probabilities = torch.softmax(output, dim=1)

        return output



def weight_and_pred(weight,estimate):
  max_values, _ = torch.max(weight, dim=1, keepdim=True)
  mask = torch.where(weight == max_values, torch.tensor(1), torch.tensor(0))
  return (estimate * mask).sum(dim = 1)

def weight_and_pred_2(weight,estimate):
  weight = nn.Softmax(1)(weight)
  simple_estimate = torch.mean(estimate,dim = 1)
  diff = estimate - torch.stack([simple_estimate] * estimate.shape[1],dim = 1)
  diff_weighted = torch.mean(diff * weight,dim = 1)
  weighted_diff_estimate = simple_estimate + diff_weighted
  return weighted_diff_estimate

def weight_and_pred_3(weight,estimate):
  weight = nn.Softmax(1)(weight)
  simple_estimate = torch.mean(estimate,dim = 1)
  diff = estimate - torch.stack([simple_estimate] * estimate.shape[1],dim = 1)
  diff_weighted = torch.mean(diff * np.exp(weight),dim = 1)
  weighted_diff_estimate = simple_estimate + diff_weighted
  return weighted_diff_estimate




def simple_agg(weight,estimate):
    return torch.mean(estimate,dim = 1)

def weight_average(weight,estimate):
    return torch.mean(estimate * weight,dim = 1)

def plot_loss_values(loss_data_string,title):
    # Split the input string by lines to get each line of data
    lines = loss_data_string.strip().split('\n')

    # Initialize empty lists to store the loss values for plotting
    epochs = []
    training_loss = []
    simple_test_loss = []
    weighted_test_loss = []
    new_weighted_test_loss = []

    # Parse each line of data and extract the values
    for line in lines:
      # print(line)
      try:
        parts = line.split(',')[1:]
        loss, simple_test_loss_val, weighted_test_loss_val, new_weighted_test_loss_val = map(
            lambda x: float(x.split(':')[1]), parts)
        # epochs.append(epoch)
        training_loss.append(loss)
        simple_test_loss.append(simple_test_loss_val)
        weighted_test_loss.append(weighted_test_loss_val)
        new_weighted_test_loss.append(new_weighted_test_loss_val)
      except:pass
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label='Training Loss', marker='o')
    plt.plot(simple_test_loss, label='Simple Test Loss', marker='o')
    plt.plot(weighted_test_loss, label='Weighted Test Loss', marker='o')
    plt.plot(new_weighted_test_loss, label='New Weighted Test Loss', marker='o')

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.yscale('log')

    # Show the plot
    plt.grid(True)
    plt.savefig(str(title) + ".png")
    plt.close()


def plot_loss_values_2(indicators,title,plot_range):
    
    df = pd.DataFrame(indicators)
    df = df.set_index(df["epoch"])    


    fig,ax = plt.subplots(1,1,figsize=(10, 6))

    for c in df.drop(columns="epoch").columns:
        # ax.set_ylim(min(plot_range),max(plot_range))
        ax.plot(df[c], label=c, marker='o')
        ax.xaxis.set_label("Epoch")
        ax.yaxis.set_label("Loss")

    # Add labels and legend

    # fig.title(str(title))
    # fig.
    fig.legend()
    # fig.yscale('log')

    # Show the plot
    # fig.grid(True)
    fig.savefig(str(title) + ".png")
    plt.close()
    df.to_csv(str(title) + ".csv")










class Experiment:
    def __init__(self,X_train,y_train,X_test,y_test,classify = False,corr = False,corr_abs = False,baseLine = False,
                 loss = nn.MSELoss(),lr = 0.001,class_n = 1,iterate = 1000,weighted_learn = True,
                 aggregation = simple_agg,title = str(time.time()),lr_adjusting = False,freezing = False,ensemble_n = 8,
                 weight_norm_dim = 1,plot_range = (0,1),negative = True,NET = MLP,ensemble_with_simple = False) -> None:
        self.settings = {}
        self.NET = NET
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.device = device
        if classify:
            y_train = torch.eye(class_n)[y_train]
            y_test = torch.eye(class_n)[y_test]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.device = device
        self.X_train = torch.Tensor(X_train).to(device)
        self.X_test = torch.Tensor(X_test)
        self.y_test = torch.Tensor(y_test)
        self.y_train = torch.Tensor(y_train).to(device)
        self.aggrigation = aggregation
        self.losser = loss
        self.plot_range = plot_range

        self.title = title            
        self.class_n = class_n
        self.classify = classify
        self.corr = corr
        self.corr_abs = corr_abs
        self.base_line = baseLine
        self.lr = lr
        self.iterate = iterate
        self.weighted_learn = weighted_learn
        self.lr_adjusting = lr_adjusting
        self.freezing = freezing
        self.ensemble_n = ensemble_n
        self.weight_norm_dim = weight_norm_dim
        self.negative = negative
        self.ensemble_with_simple = ensemble_with_simple
        
        self.settings["aggregate"] = aggregation.__name__
        
        self.settings["negative"] = negative
        self.settings["title"] = title
        self.settings["class_n"] = class_n
        self.settings["classify"] = classify
        self.settings["corr"] = corr
        self.settings["corr_abs"] = corr_abs
        self.settings["base_line"] = baseLine
        self.settings["lr"] = lr
        self.settings["iterate"] = iterate
        self.settings["weighted_learn"] = weighted_learn
        self.settings["lr_adjusting"] = lr_adjusting
        self.settings["freezing"] = freezing
        self.settings["ensemble_n"] = ensemble_n
        self.settings["weight_norm_dim"] = weight_norm_dim
        self.settings["ensemble_with_simple"] = ensemble_with_simple

        self.save_setting
    def save_setting(self):
        pd.Series(self.settings).to_json(str(self.title) + ".json")
        

    def baseLine_train(self):
        pass        
    
    def train(self):
        print(self.X_train.shape,self.y_train.shape)
        dataset = TensorDataset(torch.Tensor(self.X_train), torch.Tensor(self.y_train))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = self.X_train.shape[1]
        learning_rate = self.lr
        batch_size = 32
        ensamble = self.ensemble_n
        out_text = ""        
        weight_norm_dim = self.weight_norm_dim

        if  self.lr_adjusting:

            if weight_norm_dim == 1:
                lr_adjust = ensamble
            else:
                lr_adjust = batch_size
        else:
            lr_adjust = 1
        
        if self.classify:
            out = 32
        else:
            out = 1
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.NET(input_size,out,ensamble)
        model.to(device)
        criterion = self.losser  # Mean Squared Error loss for regression
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(),lr = learning_rate)
        indicators = []
        
        if self.ensemble_with_simple:
            simple_model = self.NET(input_size,out,ensamble)
            simple_model.to(device)
            simple_optimizer = optim.SGD(simple_model.parameters(),lr = learning_rate)
        
        
        if self.classify:
            clss = Classifier(out,self.class_n)
            clss.to(device)
            if self.ensemble_with_simple:
                cls_simple = Classifier(out,self.class_n)
                cls_simple.to(device)
        for epoch in range(self.iterate):
            indicators_d  = {}
            indicators_d["epoch"] = epoch
            self.out_text = out_text
            
            total_loss = 0.0
            first_loss_sum = 0.0
            second_loss_sum = 0.0
            
            ii = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if self.freezing:
                    do_corr = random.uniform(0,1) > 0.2
                    if do_corr:
                        model.last.requires_grad_(False)
                        model.weight_layer.requires_grad_(True)
                    else:
                        model.last.requires_grad_(True)
                        model.weight_layer.requires_grad_(False)
                y_pred,weight_raw = model(inputs)
                # weight = nn.Softmax(dim=weight_norm_dim)(-(weight_raw**2))
                weight = nn.Softmax(dim=weight_norm_dim)(weight_raw)
                if self.classify:
                    y_preds = [clss(y_pred[:,i,:]) for i in range(ensamble)]
                    y_pred = torch.stack(y_preds,axis = 1)
                # weight_estimate = torch.mean(y_pred * weight,dim = 1)
                # loss_estimate = criterion(weight_estimate, labels)
                if not self.base_line:
                    simple_estimate = torch.mean(y_pred,dim = 1)
                else:
                    simple_estimate = y_pred
                if self.classify:
                    weight = torch.stack([weight] * self.class_n,dim=-1)
                # est = weight_and_pred_2(weight,y_pred)
                # est = weight_and_pred(weight,y_pred)
                if self.base_line:
                    est = y_pred
                else:
                    est = self.aggrigation(weight,y_pred)
                
                # loss_estimate = criterion(est,labels) * weight
                if self.classify:
                    est = nn.Softmax(dim=1)(est)
                if self.weighted_learn:
                    loss_estimate = torch.mean(criterion(est, labels) * weight)
                else:
                    loss_estimate = criterion(est, labels)
                loss_corr = 0
                first_loss = 0
                second_loss = 0
                if self.corr and (not self.base_line):
                # if self.corr and (not self.base_line) and not ((self.freezing) and (not do_corr)):
                    if y_pred.shape != labels.shape:
                        L = (y_pred - torch.stack([labels for _ in range(y_pred.shape[1])],dim=-1)).detach()**2
                    else:                    
                        L = (y_pred - labels).detach()**2
                    # first_grad = weight * weight * (L - torch.mean(L,dim=1,keepdim=True))
                    LL = (L - torch.mean(L,dim=1,keepdim=True))
                    first_grad = (weight - weight.mean(dim=1,keepdim=True)) *  torch.stack([(weight[:,j] -weight.mean(dim=1)) *LL.mean(dim = 1) for j in range(y_pred.shape[1])],dim=1)                    
                    # first_loss = first_grad*(weight - weight.mean(dim = 1,keepdim = True))
                    # first_loss = first_grad*(weight_raw - weight_raw.mean(dim = 1,keepdim = True))
                    
                    # first_loss = first_grad.abs()
                    first_loss = first_grad
                    first_loss = first_loss.mean(dim = 0).mean()
                    first_loss_sum += first_loss.item()
                    # loss += first_loss
                    loss_corr = 0
                    n = 1
                    corrs = []
                    # for i in range(weight.shape[1]):
                    #     for ii in range(weight.shape[1]):
                    #         if i == ii:continue
                    #         stacked_w_i = torch.stack([weight[:,i] for _ in range(weight.shape[1])],dim = 1)
                    #         stacked_w_ii = torch.stack([weight[:,ii] for _ in range(weight.shape[1])],dim = 1)
                    #         # corr_grad =  - stacked_w_ii* (weight*(L - torch.mean(L,dim=1,keepdim=True)).detach()) 
                            
                    #         corr_grad = stacked_w_i*stacked_w_ii*L - stacked_w_ii*  torch.stack([weight[:,j]*LL.mean(dim = 1) for j in range(y_pred.shape[1])],dim=1)  
                    #         corr_loss = corr_grad*(weight_raw - weight_raw.mean(dim = 1,keepdim = True))**2
                    #         corr_loss = corr_loss.mean().mean().abs()
                    #         # if torch.isnan(corr_loss):
                    #         #     print(corr_loss)
                    #         # else:
                    #         corrs.append(corr_loss)
                    #         # corrs.append(corr_loss.mean(dim = 0).mean())
                    # l = abs(y_pred - labels).detach()**2
                    # LL = (l - torch.mean(l,dim=1,keepdim=True))
                    first_grad = 0
                    for j in range(weight.shape[1]):
                        # first_grad += torch.stack([weight_value[:,j] for _ in range(yy_pred.shape[1])],dim = 1) *  torch.stack([weight_value[:,k]*(l[:,j] - l[:,k]) for k in range(yy_pred.shape[1])],dim=1) 
                        first_grad += weight *  torch.stack([weight[:,k]*(L[:,j] - L[:,k]) for k in range(weight.shape[1])],dim=1) 
                    first_grad = first_grad/weight.shape[1]    
                    # first_grad = (weight - weight.mean(dim=1,keepdim=True)) *  torch.stack([(weight[:,j] -weight.mean(dim=1)) *LL.mean(dim = 1) for j in range(yy_pred.shape[1])],dim=1)                    

                    # first_grad = weight_stacked * weight_stacked * (l - torch.mean(l,dim=1,keepdim=True))
                    first_loss = first_grad * (weight_raw - torch.log(weight.mean(dim = 0,keepdim = True)))
                    # first_loss = first_grad
                    # first_loss = first_grad*(weight - weight.mean(dim = 1,keepdim = True))
                    
                    first_loss = first_loss.mean(dim = 0).mean()
                    for i in range(weight.shape[1]):
                        for ii in range(i,weight.shape[1]):
                            stacked_w_i = torch.stack([weight[:,i] for _ in range(weight.shape[1])],dim = 1)
                            stacked_w_ii = torch.stack([weight[:,ii] for _ in range(weight.shape[1])],dim = 1)
                            
                            stacked_w_i_raw = torch.stack([weight_raw[:,i] for _ in range(weight.shape[1])],dim = 1)
                            stacked_w_ii_raw = torch.stack([weight_raw[:,ii] for _ in range(weight.shape[1])],dim = 1)
                            corr_grad = stacked_w_i*(stacked_w_ii - stacked_w_ii**2)*(L - torch.stack([L[:,ii] for _ in range(weight.shape[1])],dim=1)).detach()
                            # corr_loss = corr_grad*(weight_raw - torch.log(weight.mean(dim = 0,keepdim = True)))**2
                            corr_loss = corr_grad*(stacked_w_i_raw - np.log(weight.shape[1]))*(stacked_w_ii_raw - np.log(weight.shape[1]))
                            corrs.append(corr_loss.mean(dim = 0).mean())
                    second_loss = torch.stack(corrs).mean()
                    second_loss_sum += second_loss.item()
                loss = loss_estimate
                # if first_loss:
                #     loss = loss - first_loss
                # if loss_corr:
                #     loss = loss - loss_corr
                # loss = loss + abs(first_loss) + abs(second_loss)
                loss = loss  - (second_loss)
                
                # loss = loss + abs(first_loss) 
                
                loss = loss * lr_adjust
                # if loss_corr:
                #     loss = (loss_estimate -first_loss - loss_corr)*lr_adjust
                # else:
                #     loss = loss_estimate * lr_adjust            
                if torch.isnan(loss):
                    raise Exception
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss_estimate.item()
                if self.ensemble_with_simple:
                    y_pred,weight_raw = simple_model(inputs)
                    # weight = nn.Softmax(dim=weight_norm_dim)(-(weight_raw**2))
                    weight = nn.Softmax(dim=weight_norm_dim)(weight_raw)
                    if self.classify:
                        y_preds = [cls_simple(y_pred[:,i,:]) for i in range(ensamble)]
                        y_pred = torch.stack(y_preds,axis = 1)
                    # weight_estimate = torch.mean(y_pred * weight,dim = 1)
                    # loss_estimate = criterion(weight_estimate, labels)
                    if not self.base_line:
                        simple_estimate = torch.mean(y_pred,dim = 1)
                    else:
                        simple_estimate = y_pred
                    if self.classify:
                        weight = torch.stack([weight] * self.class_n,dim=-1)
                    # est = weight_and_pred_2(weight,y_pred)
                    # est = weight_and_pred(weight,y_pred)
                    if self.base_line:
                        est = y_pred
                    else:
                        est = simple_agg(weight,y_pred)
                    
                    # loss_estimate = criterion(est,labels) * weight
                    if self.classify:
                        est = nn.Softmax(dim=1)(est)
                    loss_estimate = criterion(est, labels)
                    loss = loss_estimate
                    simple_optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            average_loss = total_loss / len(dataloader)/batch_size
            # indicators_d["train_loss"] = average_loss
            if self.classify:
                with torch.no_grad():
                        
                    y_test = torch.Tensor(self.y_test)
                    y_pred,weight = model(torch.Tensor(self.X_test.to(device)))

                    weight = nn.Softmax(1)(weight)
                    # weight = nn.Softmax(dim=1)(-(weight**2))
                    
                    y_preds = [clss(y_pred[:,i,:]) for i in range(ensamble)]
                    y_pred = torch.stack(y_preds,axis = 1)
                    y_pred = y_pred.to("cpu")
                    weight = weight.to("cpu")
                    simple_estimate = torch.mean(y_pred,dim = 1)
                    diff = y_pred - torch.stack([simple_estimate] * y_pred.shape[1],dim = 1)
                    weight = torch.stack([weight] * self.class_n,dim = -1)

                    diff_weighted = torch.mean(diff * weight,dim = 1)
                    weighted_diff_estimate = simple_estimate + diff_weighted
                    new_weighted_estimate = weight_and_pred(weight,y_pred)
                    # weight_average_estimate = weight_average(weight,y_pred)
                    if self.ensemble_with_simple:
                        y_pred,weight = model(torch.Tensor(self.X_test.to(device)))
                        y_preds = [clss(y_pred[:,i,:]) for i in range(ensamble)]
                        y_pred = torch.stack(y_preds,axis = 1)
                        simple_simple_estimate = simple_agg(weight,y_pred)
                        simple_estimate = (simple_estimate + simple_simple_estimate)/2
                        weighted_diff_estimate = (weighted_diff_estimate + simple_simple_estimate)/2                    
                        new_weighted_estimate = (new_weighted_estimate + simple_simple_estimate)/2                    
                                            
                    l =  nn.BCEWithLogitsLoss()
                    # weight_average_testLoss = l(y_test,weight_average_estimate)
                    simple_testLoss = l(simple_estimate,y_test)
                    weighted_testLoss = l(weighted_diff_estimate,y_test)
                    # new_weighted_testLoss = l(new_weighted_estimate,y_test)
                    new_weighted_testLoss = l(new_weighted_estimate,y_test)
                    

                    
                    print(y_test[0,:])
                    print(new_weighted_estimate[0,:])
                    
                    # indicators_d["weight_average_testLoss"] = weight_average_testLoss.item()
                    indicators_d["simple_testLoss"] = simple_testLoss.item()
                    indicators_d["weighted_testLoss"] = weighted_testLoss.item()
                    indicators_d["new_weighted_testLoss"] = new_weighted_testLoss.item()
                    
                    
                    # t = f"Epoch {epoch + 1}, Loss: {average_loss:.4f},simpleTestLoss: {simple_testLoss.item():.4f},weightedTestLoss: {weighted_testLoss.item():.4f},newWeightedTestLoss: {new_weighted_testLoss.item():.4f}"
                    # out_text = out_text +"\n" +  t
                    # print(t)
            else:
                with torch.no_grad():
                    y_test = torch.Tensor(self.y_test)
                    y_pred,weight = model(torch.Tensor(self.X_test.to(device)))
                    if self.base_line:
                        l = criterion(y_test,y_pred)
                        t = f"Epoch {epoch + 1}, Loss: {average_loss:.4f},simpleTestLoss: {l.item():.4f},weightedTestLoss: {l.item():.4f},newWeightedTestLoss: {l.item():.4f}"
                        out_text = out_text +"\n" +  t
                        print(t)
                        continue
                    weight = nn.Softmax(1)(weight)
                    y_pred = y_pred.to("cpu")
                    weight = weight.to("cpu")
                    simple_estimate = torch.mean(y_pred,dim = 1)
                    diff = y_pred - torch.stack([simple_estimate] * y_pred.shape[1],dim = 1)
                    diff_weighted = torch.mean(diff * weight,dim = 1)
                    weighted_diff_estimate = simple_estimate + diff_weighted
                    new_weighted_estimate = weight_and_pred(weight,y_pred)
                    # weighted_diff_estimate = weight_and_pred_3(weight,y_pred)
                    
                    if self.ensemble_with_simple:
                        y_pred,weight = model(torch.Tensor(self.X_test.to(device)))
                        y_pred = y_pred.to("cpu")
                        weight = weight.to("cpu")
                        simple_simple_estimate = simple_agg(weight,y_pred)
                        simple_estimate = (simple_estimate + simple_simple_estimate)/2
                        weighted_diff_estimate = (weighted_diff_estimate + simple_simple_estimate)/2                    
                        new_weighted_estimate = (new_weighted_estimate + simple_simple_estimate)/2                    
                                     
                    simple_testLoss = criterion(y_test,simple_estimate)
                    weighted_testLoss = criterion(y_test,weighted_diff_estimate)
                    new_weighted_testLoss = criterion(y_test,new_weighted_estimate)
                    indicators_d["simple_testLoss"] = simple_testLoss.item()
                    indicators_d["weighted_testLoss"] = weighted_testLoss.item()
                    indicators_d["new_weighted_testLoss"] = new_weighted_testLoss.item()
                    indicators_d["train_est_loss"] = total_loss
                    indicators_d["train_first_loss"] = first_loss_sum
                    indicators_d["train_second_loss"] = second_loss_sum

                    
            indicators.append(indicators_d)
            print(indicators_d)
            print(self.settings)
            # self.plot(indicators)
        try:
            self.plot(indicators)
        except:pass
    def plot(self,indicators):
        self.plot_loss_values_2(indicators,self.title,self.plot_range)
    def plot_loss_values_2(self,indicators,title,plot_range):
        
        
        d = self.settings.copy()
        d["indicators"] = indicators
        with open(f"{self.title}.result.json",mode="w") as f:
            json.dump(d,f)
        df = pd.DataFrame(indicators)
        df = df.set_index(df["epoch"])    
        fig,ax = plt.subplots(1,1,figsize=(10, 6))

        for c in df.drop(columns="epoch").columns:
            # ax.set_ylim(min(plot_range),max(plot_range))
            ax.plot(df[c], label=c, marker='o')
            ax.xaxis.set_label("Epoch")
            ax.yaxis.set_label("Loss")

        # Add labels and legend

        # fig.title(str(title))
        # fig.
        fig.legend()
        # fig.yscale('log')

        # Show the plot
        # fig.grid(True)
        fig.savefig(str(title) + ".png")
        plt.close()
        # df.to_csv(str(title) + ".csv")

import torchvision.datasets

class Experiment_imagenet(Experiment):
    def __init__(self,classify=False, corr=False, corr_abs=False, baseLine=False, loss=nn.MSELoss(), lr=0.001, class_n=1, 
                 iterate=1000, weighted_learn=True, aggregation=simple_agg, title=str(time.time()),
                 lr_adjusting=False, freezing=False, ensemble_n=8, weight_norm_dim=1, plot_range=(0, 1), negative=True, NET=CNN,
                 save_dir = "experiments") -> None:
        
        train_loader = torchvision.datasets.CIFAR10(root='./data', train=True,download = True)
        test_loader = torchvision.datasets.CIFAR10(root='./data', train=False,download = True)        
        
        

        
        print(train_loader.data.shape)
        X_train = torch.Tensor(train_loader.data).permute(0, 3, 1, 2)[:32*32]
        y_train = train_loader.targets[:32*32]
        X_test = torch.Tensor(test_loader.data).permute(0, 3, 1, 2)
        y_test = test_loader.targets
        
        super().__init__(X_train, y_train, X_test, y_test, classify, corr, corr_abs, baseLine, loss, lr, class_n, iterate, weighted_learn, aggregation, title, lr_adjusting, freezing, ensemble_n, weight_norm_dim, plot_range, negative, NET)
        
    def train(self):
        print(self.X_train.shape,self.y_train.shape)
        dataset = TensorDataset(torch.Tensor(self.X_train), torch.Tensor(self.y_train))

        input_size = self.X_train.shape[1]
        learning_rate = self.lr
        batch_size = 32
        ensamble = self.ensemble_n
        out_text = ""        
        weight_norm_dim = self.weight_norm_dim

        if  self.lr_adjusting:

            if weight_norm_dim == 1:
                lr_adjust = ensamble
            else:
                lr_adjust = batch_size
        else:
            lr_adjust = 1
        
        if self.classify:
            out = 32
        else:
            out = 1
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.NET(input_size,out,ensamble)
        model.to(self.device)
        criterion = self.losser  # Mean Squared Error loss for regression
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(),lr = learning_rate)
        indicators = []
        
        if self.classify:
            clss = Classifier(out,self.class_n)
            clss.to(self.device)
        for epoch in range(self.iterate):
            indicators_d  = {}
            indicators_d["epoch"] = epoch
            self.out_text = out_text
            
            total_loss = 0.0
            ii = 0

            for inputs, labels in dataloader:
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                ii += 1
                if ii > 10:
                    break    
                if self.freezing:
                    do_corr = random.uniform(0,1) > 0.2
                    if do_corr:
                        model.last.requires_grad_(False)
                        model.weight_layer.requires_grad_(True)
                    else:
                        model.last.requires_grad_(True)
                        model.weight_layer.requires_grad_(False)
                y_pred,weight_raw = model(inputs)
                # weight = nn.Softmax(dim=weight_norm_dim)(-(weight_raw**2))
                weight = nn.Softmax(dim=weight_norm_dim)(weight_raw)
                if self.classify:
                    y_preds = [clss(y_pred[:,i,:]) for i in range(ensamble)]
                    y_pred = torch.stack(y_preds,axis = 1)
                # weight_estimate = torch.mean(y_pred * weight,dim = 1)
                # loss_estimate = criterion(weight_estimate, labels)
                if not self.base_line:
                    simple_estimate = torch.mean(y_pred,dim = 1)
                else:
                    simple_estimate = y_pred
                if self.classify:
                    weight = torch.stack([weight] * self.class_n,dim=-1)
                # est = weight_and_pred_2(weight,y_pred)
                # est = weight_and_pred(weight,y_pred)
                if self.base_line:
                    est = y_pred
                else:
                    est = self.aggrigation(weight,y_pred)
                
                # loss_estimate = criterion(est,labels) * weight
                if self.classify:
                    est = nn.Softmax(dim=1)(est)
                if self.weighted_learn:
                    loss_estimate = torch.mean(criterion(est, labels) * weight)
                else:
                    loss_estimate = criterion(est, labels)

                

                loss_corr = 0
                n = 1
                if self.corr and (not self.base_line):
                # if self.corr and (not self.base_line) and not ((self.freezing) and (not do_corr)):
                
                    for i in range(weight.shape[1]):
                        for j in range(i+1,weight.shape[1]):

                            # loss_corr += -losser_corr(weight[:,i],weight[:,j])
                            # loss_corr += -nn.MSELoss()(weight[:,i],weight[:,j])
                            if self.corr_abs:                            
                                loss_corr += abs(Corr_loss()(weight[:,i],weight[:,j]))
                                # loss_corr += abs(Corr_loss()(weight_raw[:,i],weight_raw[:,j]))
                                # loss_corr += -abs(nn.MSELoss()(weight[:,i],weight[:,j]))
                            else:
                                loss_corr += Corr_loss()(weight[:,i],weight[:,j])
                                # loss_corr += Corr_loss()(weight_raw[:,i],weight_raw[:,j])
                                # loss_corr += Corr_loss()(weight[:,i],weight[:,j])
                                # loss_corr += -(nn.MSELoss()(weight[:,i],weight[:,j]))
                                
                                
                            # loss_corr += abs(Corr_loss()(weight_raw[:,i],weight_raw[:,j]))



                            n += 1
                loss = loss_estimate * lr_adjust + loss_corr/n * 2
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss_estimate.item()

            average_loss = total_loss / len(dataloader)/batch_size
            # indicators_d["train_loss"] = average_loss
            if self.classify:
                with torch.no_grad():
                        
                    y_test = torch.Tensor(self.y_test)
                    y_pred,weight = model(torch.Tensor(self.X_test.to(self.device)))

                    weight = nn.Softmax(1)(weight)
                    # weight = nn.Softmax(dim=1)(-(weight**2))
                    
                    y_preds = [clss(y_pred[:,i,:]) for i in range(ensamble)]
                    y_pred = torch.stack(y_preds,axis = 1)
                    y_pred = y_pred.to("cpu")
                    weight = weight.to("cpu")
                    simple_estimate = torch.mean(y_pred,dim = 1)
                    diff = y_pred - torch.stack([simple_estimate] * y_pred.shape[1],dim = 1)
                    weight = torch.stack([weight] * self.class_n,dim = -1)

                    diff_weighted = torch.mean(diff * weight,dim = 1)
                    weighted_diff_estimate = simple_estimate + diff_weighted
                    new_weighted_estimate = weight_and_pred(weight,y_pred)
                    # weight_average_estimate = weight_average(weight,y_pred)
                    
                    l =  nn.BCEWithLogitsLoss()
                    # weight_average_testLoss = l(y_test,weight_average_estimate)
                    simple_testLoss = l(simple_estimate,y_test)
                    weighted_testLoss = l(weighted_diff_estimate,y_test)
                    # new_weighted_testLoss = l(new_weighted_estimate,y_test)
                    new_weighted_testLoss = l(new_weighted_estimate,y_test)
                    
                    
                    print(y_test[0,:])
                    print(new_weighted_estimate[0,:])
                    
                    # indicators_d["weight_average_testLoss"] = weight_average_testLoss.item()
                    indicators_d["simple_testLoss"] = simple_testLoss.item()
                    indicators_d["weighted_testLoss"] = weighted_testLoss.item()
                    indicators_d["new_weighted_testLoss"] = new_weighted_testLoss.item()
                    
                    # t = f"Epoch {epoch + 1}, Loss: {average_loss:.4f},simpleTestLoss: {simple_testLoss.item():.4f},weightedTestLoss: {weighted_testLoss.item():.4f},newWeightedTestLoss: {new_weighted_testLoss.item():.4f}"
                    # out_text = out_text +"\n" +  t
                    # print(t)
            else:
                with torch.no_grad():
                    y_test = torch.Tensor(self.y_test)
                    y_pred,weight = model(torch.Tensor(self.X_test.to(self.device)))
                    if self.base_line:
                        l = criterion(y_test,y_pred)
                        t = f"Epoch {epoch + 1}, Loss: {average_loss:.4f},simpleTestLoss: {l.item():.4f},weightedTestLoss: {l.item():.4f},newWeightedTestLoss: {l.item():.4f}"
                        out_text = out_text +"\n" +  t
                        print(t)
                        continue
                    weight = nn.Softmax(1)(weight)
                    y_pred = y_pred.to("cpu")
                    weight = weight.to("cpu")
                    simple_estimate = torch.mean(y_pred,dim = 1)
                    diff = y_pred - torch.stack([simple_estimate] * y_pred.shape[1],dim = 1)
                    diff_weighted = torch.mean(diff * weight,dim = 1)
                    weighted_diff_estimate = simple_estimate + diff_weighted
                    new_weighted_estimate = weight_and_pred(weight,y_pred)
                    simple_testLoss = criterion(y_test,simple_estimate)
                    weighted_testLoss = criterion(y_test,weighted_diff_estimate)
                    new_weighted_testLoss = criterion(y_test,new_weighted_estimate)
                    indicators_d["simple_testLoss"] = simple_testLoss.item()
                    indicators_d["weighted_testLoss"] = weighted_testLoss.item()
                    indicators_d["new_weighted_testLoss"] = new_weighted_testLoss.item()
            indicators.append(indicators_d)
            print(indicators_d)
            self.plot(indicators)
            # try:
            #     self.plot(indicators)
            # except:pass

class Experiment_califolnia_housing(Experiment):
    def __init__(self, classify=False, corr=False, corr_abs=False, baseLine=False, loss=nn.MSELoss(), lr=0.001, class_n=1, iterate=1000, weighted_learn=True, aggregation=simple_agg, title=str(time.time()), lr_adjusting=False, freezing=False, ensemble_n=8, weight_norm_dim=1, plot_range=(0, 1), negative=True, NET=MLP) -> None:
        iris = sklearn.datasets.fetch_california_housing()
        X = iris.data
        y = iris.target
        X = (X - X.mean())/(X.std())
        # y = (y - y.mean())/(y.std())


        train_index = sorted(list(random.sample(list(range(X.shape[0])),X.shape[0] * 9 //10,)))
        test_index = [i for i in range(X.shape[0]) if i not in train_index]

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]
        super().__init__(X_train, y_train, X_test, y_test, classify, corr, corr_abs, baseLine, loss, lr, class_n, iterate, weighted_learn, aggregation, title, lr_adjusting, freezing, ensemble_n, weight_norm_dim, plot_range, negative, NET)
        
class Experiment_news(Experiment):
    def __init__(self,  classify=True, corr=False, corr_abs=False, baseLine=False, loss=nn.MSELoss(), lr=0.001, class_n=20, iterate=1000, weighted_learn=True, aggregation=simple_agg, title=str(time.time()), lr_adjusting=False, freezing=False, ensemble_n=8, weight_norm_dim=1, plot_range=(0, 1), negative=True, NET=MLP) -> None:
        iris = sklearn.datasets.fetch_20newsgroups_vectorized()
        X = iris.data
        y = iris.target
        X = X.toarray()
        X = (X - X.mean())/(X.std())
        # y = (y - y.mean())/(y.std())


        train_index = sorted(list(random.sample(list(range(X.shape[0])),X.shape[0] * 9 //10,)))
        test_index = [i for i in range(X.shape[0]) if i not in train_index]

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]
        super().__init__(X_train, y_train, X_test, y_test, classify, corr, corr_abs, baseLine, loss, lr, class_n, iterate, weighted_learn, aggregation, title, lr_adjusting, freezing, ensemble_n, weight_norm_dim, plot_range, negative, NET)

class Experiment_wine(Experiment):
    def __init__(self,  classify=True, corr=False, corr_abs=False, baseLine=False, loss=nn.MSELoss(), lr=0.001, class_n=3, iterate=1000, weighted_learn=True, aggregation=simple_agg, title=str(time.time()), lr_adjusting=False, freezing=False, ensemble_n=8, weight_norm_dim=1, plot_range=(0, 1), negative=True, NET=MLP) -> None:
        iris = sklearn.datasets.load_wine()
        X = iris.data
        y = iris.target
        X = (X - X.mean())/(X.std())
        # y = (y - y.mean())/(y.std())


        train_index = sorted(list(random.sample(list(range(X.shape[0])),X.shape[0] * 9 //10,)))
        test_index = [i for i in range(X.shape[0]) if i not in train_index]

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]
        super().__init__(X_train, y_train, X_test, y_test, classify, corr, corr_abs, baseLine, loss, lr, class_n, iterate, weighted_learn, aggregation, title, lr_adjusting, freezing, ensemble_n, weight_norm_dim, plot_range, negative, NET,ensemble_with_simple=True)
        
        
class Experiment_diabate(Experiment):
    def __init__(self,classify=False, corr=False, corr_abs=False, baseLine=False, loss=nn.MSELoss(), lr=0.001, class_n=1, iterate=1000, weighted_learn=True, aggregation=simple_agg, title=str(time.time()), lr_adjusting=False, freezing=False, ensemble_n=8, weight_norm_dim=1, plot_range=(0, 1), negative=True, NET=MLP) -> None:
        iris = sklearn.datasets.load_diabetes()
        X = iris.data
        y = iris.target
        X = (X - X.mean())/(X.std())
        # y = (y - y.mean())/(y.std())


        train_index = sorted(list(random.sample(list(range(X.shape[0])),X.shape[0] * 9 //10,)))
        test_index = [i for i in range(X.shape[0]) if i not in train_index]

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]
        super().__init__(X_train, y_train, X_test, y_test, classify, corr, corr_abs, baseLine, loss, lr, class_n, iterate, weighted_learn, aggregation, title, lr_adjusting, freezing, ensemble_n, weight_norm_dim, plot_range, negative, NET)
        
class Experiment_digit(Experiment):
    def __init__(self,  classify=True, corr=False, corr_abs=False, baseLine=False, loss=nn.MSELoss(), lr=0.001, class_n=10, iterate=1000, weighted_learn=True, aggregation=simple_agg, title=str(time.time()), lr_adjusting=False, freezing=False, ensemble_n=8, weight_norm_dim=1, plot_range=(0, 1), negative=True, NET=MLP) -> None:
        iris = sklearn.datasets.load_digits()
        X = iris.data
        y = iris.target
        X = (X - X.mean())/(X.std())
        # y = (y - y.mean())/(y.std())


        train_index = sorted(list(random.sample(list(range(X.shape[0])),X.shape[0] * 9 //10,)))
        test_index = [i for i in range(X.shape[0]) if i not in train_index]

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]
        super().__init__(X_train, y_train, X_test, y_test, classify, corr, corr_abs, baseLine, loss, lr, class_n, iterate, weighted_learn, aggregation, title, lr_adjusting, freezing, ensemble_n, weight_norm_dim, plot_range, negative, NET)


class Experiment_artificial(Experiment):
    def __init__(self, classify=False, corr=False, corr_abs=False, baseLine=False, loss=nn.MSELoss(), lr=0.001, class_n=1, iterate=1000, weighted_learn=True, aggregation=simple_agg, title=str(time.time()), lr_adjusting=False, freezing=False, ensemble_n=8, weight_norm_dim=1, plot_range=(0, 1), negative=True, NET=MLP, ensemble_with_simple=False) -> None:        
        features =5
        noise = 120
        # m_random = 42
        
        # # Generate correlated dummy data for regression
        X_1, y_1 = sklearn.datasets.make_regression(
            n_samples=100,
            n_features=features,
            noise=noise,
            random_state=42,
            bias=random.randint(0,100)
        )

        X_2, y_2 = sklearn.datasets.make_regression(
            n_samples=100,
            n_features=features,
            noise=noise,
            random_state=43,
            # bias = 50
            bias=random.randint(0,100)

        )

        X_3, y_3 = sklearn.datasets.make_regression(
            n_samples=100,
            n_features=features,
            noise=noise,
            random_state=44,
            # bias = 50,
            bias=random.randint(0,100)

        )
        X_trains = []
        y_trains = []
        X_tests = []
        y_tests = []
        for X,y,train_ratio in zip([X_1,X_2,X_3],[y_1,y_2,y_3],[0.9,0.9,0.9]):
        # for X,y,train_ratio in zip([X_1,X_2,X_3],[y_1,y_2,y_3],[0.1,0.9,0.9]):
        
        # for X,y,train_ratio in zip([X_1],[y_1],[0.9]):
            X = (X - X.mean())/(X.std())
            y = (y - y.mean())/(y.std())  
            train_index = sorted(list(random.sample(list(range(X.shape[0])),int(X.shape[0] * train_ratio,))))
            test_index = [i for i in range(X.shape[0]) if i not in train_index]
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]
            X_trains.append(X_train)
            y_trains.append(y_train)
            X_tests.append(X_test)
            y_tests.append(y_test)
            
        X_train = np.concatenate(X_trains,axis=0)
        y_train = np.concatenate(y_trains,axis=0)
        X_test = np.concatenate(X_tests,axis=0)
        y_test = np.concatenate(y_tests,axis=0)
        
  
        super().__init__(X_train, y_train, X_test, y_test, classify, corr, corr_abs, baseLine, loss, lr, class_n, iterate, weighted_learn, aggregation, title, lr_adjusting, freezing, ensemble_n, weight_norm_dim, plot_range, negative, NET,ensemble_with_simple=ensemble_with_simple)
        

def reg_test():
    import lightgbm
    # iris = sklearn.datasets.
    iris = sklearn.datasets.load_diabetes()
    X = iris.data
    y = iris.target
    X = (X - X.mean())/(X.std())
    # y = (y - y.mean())/(y.std())


    train_index = sorted(list(random.sample(list(range(X.shape[0])),X.shape[0] * 9 //10,)))
    test_index = [i for i in range(X.shape[0]) if i not in train_index]

    X_train = X[train_index]
    y_train = y[train_index]

    X_test = X[test_index]
    y_test = y[test_index]
    
    reg = lightgbm.LGBMRegressor()
    reg.fit(X_train,y_train)
    ret = np.mean((reg.predict(X_test) - y_test)**2)
    print(ret)
    
    inst = Experiment(X_train,y_train,X_test,y_test,classify=False,baseLine=False,corr=1,corr_abs=1,lr=0.001,iterate=100,weighted_learn=0,
                      aggregation=weight_and_pred,title = str(time.time()),lr_adjusting = True,plot_range = (1,0),freezing=False,NET=MLP)
    inst.train()
    # inst.plot()

# preprocess for resnet18, automatically detect image size, take arg as numpy.array
def preprocess_color(image):
    # import required libraries

    # image = Image.open(image_path)
    image = torch.Tensor(image)
    image = transforms.Resize((224, 224))(image)
    # image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(image)
    image = torch.unsqueeze(image, 0)
    return image

# preprocess for resnet18, automatically detect image size, stack 3 times as if a color image
def preprocess_gray(image):
    # import required libraries

    image = torch.Tensor(image)
    image = transforms.Resize((224, 224))(image)
    # image = transforms.ToTensor()(image)
    image = torch.stack([image,image,image],dim = 1)

    image = torch.squeeze(image, 0)
    return image





def class_test():
    import lightgbm
    # iris = sklearn.datasets.fetch_20newsgroups_vectorized()
    iris = sklearn.datasets.load_wine()
    X = iris.data
    y = iris.target
    # numpy sparce matrix to numpy array
    # X = X.toarray()
    X = (X - X.mean())/(X.std())
    # X = preprocess_gray(X)

    train_index = sorted(list(random.sample(list(range(X.shape[0])),X.shape[0] * 9 //10,)))
    test_index = [i for i in range(X.shape[0]) if i not in train_index]

    X_train = X[train_index]
    y_train = y[train_index]

    X_test = X[test_index]
    y_test = y[test_index]
    
    # reg = lightgbm.LGBMRegressor()
    # reg.fit(X_train,y_train)
    # ret = np.mean((reg.predict(X_test) - y_test)**2)
    # print(ret)
    
    inst = Experiment(X_train,y_train,X_test,y_test,classify=True,baseLine=False,corr=1,corr_abs=0,lr=0.1,iterate=100,weighted_learn=0,
                      aggregation=weight_and_pred,title = str(time.time()),lr_adjusting = 1,freezing=False,class_n=3,NET = MLP)
    inst.train()
    # inst.plot()

def imagenet_test():
    inst = Experiment_imagenet(classify=True,baseLine=False,corr=0,corr_abs=1,lr=0.01,iterate=100,weighted_learn=0,
                               aggregation=simple_agg,title = str(time.time()),lr_adjusting = 0,freezing=0,class_n=10,NET = CNN)

    inst.train()

def aaa():
    # for corr,agg,weight,freeze,absolute,i in 
    ds = itertools.product([0,1],[simple_agg,weight_and_pred,weight_and_pred_2,weight_average],[0,1],[0,1],[0,1],list(range(10)))
    ds = list(ds)
    ds = [1,simple_agg]
    for d in random.sample(ds,len(ds)): 
        corr,agg,weight,freeze,absolute,i = d
        # inst = Experiment_wine(corr=corr,corr_abs=absolute,lr=0.1,iterate=1000,weighted_learn=weight,
        #                 aggregation=agg,title = str(time.time()),lr_adjusting = 1,freezing=freeze,NET = MLP)

        inst = Experiment_califolnia_housing(corr=corr,corr_abs=absolute,lr=0.01,iterate=1000,weighted_learn=weight,
                        aggregation=agg,title = str(time.time()),lr_adjusting = 1,freezing=freeze)
        inst.train()
import itertools,random
if __name__ == "__main__":
    # ds = itertools.product([0,1],[simple_agg,weight_and_pred,weight_and_pred_2,weight_average],[0,1],[0,1],[0,1])
    # ds = list(ds)
    # ds = ds * 10
    # ds = [[0,simple_agg,0,0,0,1] for i in range(10)] + [[1,simple_agg,1,0,1,1] for i in range(10)]+ [[1,weight_and_pred,0,0,1,1] for i in range(10)]
    # ds = [[0,simple_agg,0,0,0,1] for i in range(10)] + [[1,simple_agg,1,0,1,1] for i in range(10)]
    # ds = [[0,simple_agg,0,0,0,1] for i in range(10)] + [[1,weight_and_pred,0,0,0,1] for i in range(10)]
    # ds = [[1,weight_and_pred,0,0,1,1] for i in range(10)] + [[0,weight_and_pred,0,0,0,1] for i in range(10)] + [[0,simple_agg,0,0,0,1] for i in range(10)] 
    # ds = [[1,simple_agg,1,1,1,1] for i in range(10)] + [[1,weight_and_pred,0,1,0,1] for i in range(10)]  
    # ds = [[1,simple_agg,1,1,1,1] for i in range(10)] +[[0,simple_agg,0,0,0,1] for i in range(10)]
    # ds = [[1,simple_agg,1,1,1,1] for i in range(10)]
    ds =  [[1,weight_and_pred,0,0,0,1] for i in range(10)]   
     
    # ds = [[1,simple_agg,1,1,1,1] for i in range(10)] 
    # ds = [[0,simple_agg,0,0,0,1] for i in range(10)] + [[1,weight_and_pred_2,0,1,1,1] for i in range(10)]
    # ds = [[0,simple_agg,0,0,0,1] for i in range(10)] + [[1,weight_and_pred,0,1,1,1] for i in range(10)]+ [[1,weight_and_pred,0,0,1,1] for i in range(10)]
    
    
    for d in random.sample(ds,len(ds)): 
        corr,agg,weight,freeze,absolute,i = d
        # inst = Experiment_wine(corr=corr,corr_abs=absolute,lr=0.1,iterate=300,weighted_learn=weight,
        #                 aggregation=agg,title = str(time.time()),lr_adjusting = 1,freezing=freeze,NET = MLP)        
        
        inst = Experiment_artificial(corr=corr,corr_abs=absolute,lr=0.01,iterate=1000,weighted_learn=weight,
                        aggregation=agg,title = str(time.time()),lr_adjusting = 0,freezing=freeze,NET = MLP,ensemble_with_simple=False)
        inst.train()
        # try:
        #     # inst = Experiment_diabate(corr=corr,corr_abs=absolute,lr=0.001,iterate=1000,weighted_learn=weight,
        #     #             aggregation=agg,title = str(time.time()),lr_adjusting = 1,freezing=freeze)
        #     inst.train()
        # except Exception as e:
        #     print(e)