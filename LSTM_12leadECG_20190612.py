#!/home/desktop/anaconda3/bin/python3.6
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from biosppy.signals import ecg

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))
#print(torch.cuda.current_device())
#############################################

## Load img & label array   
with open('/home/desktop/thesis/ls_ilabel.pkl','rb') as file:
    label_f = pickle.load(file)
np_label = np.array(label_f)

np_img = np.load('/home/desktop/thesis/patient_lead.npy')

#print(np_img)
#print(np_label)
#print(np_label.shape)           #(2540,)
#print(np.unique(np_label))
#print(np_img.shape)             #(2540,1,12,5000)
#print(np_img[-1])
#print(len(np_label))
#print(list(np_label).count(0))


## Split dataset
x = np_img
y = np_label

np.random.seed(7)
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

#print(x)
#print(y)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

kf = KFold(n_splits=10)
#
for train_index, test_index in kf.split(x):
    #print("Trian:",train_index ,"test:",test_index)
    x_train ,x_test = x[train_index],x[test_index]
    y_train ,y_test = y[train_index],y[test_index]

#print(x_train.shape)         
#print(x_test.shape)          
#print(y_train.shape)         
#print(y_test.shape)          
#print("x_train = ", x_train)
#print("y_train = ", y_train)


##################################################################
## Hyper Parameters
Batch_size = 32
learning_rate = 0.0001
num_epochs = 50

##  Dataset
class ecg_dataset(Dataset):
    def __init__(self,x,y,transform=None):   #initial processes,reading data
        super(ecg_dataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__ (self, idx):       #return one item on the index 
        return idx, self.x[idx], self.y[idx]
      

    def __len__(self):   #return the data length
        return len(self.x)

## DataLoader
tr_dataset = ecg_dataset(x_train,y_train)
ts_dataset = ecg_dataset(x_test,y_test)
tr_dataloader = DataLoader(tr_dataset, Batch_size,shuffle=True)
ts_dataloader = DataLoader(ts_dataset, Batch_size,shuffle=False)

#print(tr_dataset.__len__())
#print(len(ts_dataset))
#print(next(iter(tr_dataloader)))
#print(next(iter(ts_dataloader)))


#print(x.shape)
#x = x.squeeze(x,1)
#print(x.shape)


# Define  Model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.conv = nn.Conv1d()
        self.rnn = nn.LSTM(input_size=5000,hidden_size=128,num_layers=2,batch_first=True,dropout=0.5,)
        self.dense = nn.Linear(128,2)
          
 
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)  
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)

        out = self.dense(r_out[:, -1, :])
        return out


if torch.cuda.is_available():
        model = LSTM().cuda()
else:
        model = LSTM()

print(model)

## Loss Function & Optimization
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(True), lr=learning_rate)

## early stop
loss_history = []
count = 0

## Training Model
model.train()
total_step = len(tr_dataloader)
tr_total = 0
tr_correct = 0
tr_loss = 0.0
tr_acc = 0.0
print("\nTraining Model...")
for epoch in range(num_epochs):
        i = 0
        for i,x_train, y_train in (tr_dataloader):
                if torch.cuda.is_available():
                        input = Variable(x_train.float(), requires_grad=True).cuda()
                        target = Variable(y_train).cuda()
                else:
                        input = Variable(x_train.float(), requires_grad=True)
                        target = Variable(y_train)                
                input = input.reshape(-1,12,5000)
                output = model(input)
                loss = criterion(output,target)
                #print("out", out.shape, "\n", out)
                #print("target", target.shape, "\n", target)
                #print("pred", pred.shape, "\n", pred)
                optimizer.zero_grad()  #reset gradients
                loss.backward()        #backward pass
                optimizer.step()       #update parameters
                tr_loss = loss.item()
                _,pred = torch.max(output.data,1)
                tr_total += target.size(0)
                tr_correct += (pred == target).sum().item()

        if (epoch + 1) % 5 == 0:
                print('num_epochs [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, tr_loss))

        loss_history.append(tr_loss)
        num_early_stop = 5
        if(num_early_stop>0):
                if(len(loss_history)>1 and loss_history[-1]>loss_history[-2]):
                    count += 1
                    if count>num_early_stop:
                        break


#print("Train Accurary:",tr_correct.cpu().numpy()/len(tr_dataset))
print("Train Accurary: {:.4f} %".format(100*tr_correct/tr_total))


#############################################################
print("\nTesting Model...")
## Testing Model
model.eval()

ts_total = 0
ts_correct = 0
#for epoch in range(num_epochs):
for i,x_test,y_test in (ts_dataloader):
        if torch.cuda.is_available():
                input  = Variable(x_test.float()).cuda()
                target  = Variable(y_test).cuda()
        else:
                input = Variable(x_test.float())
                target = Variable(y_test)

        input = input.reshape(-1,12,5000)
        output = model(input)
        loss = criterion(output,target)
        _,pred = torch.max(output.data,1)
        ts_loss = loss.item()
        ts_total += target.size(0)
        ts_correct += (pred == target).sum().item()

        print("pred",pred)
        print("targer",target)

        if (epoch + 1) % 5 == 0:
                print('num_epochs [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, ts_loss))

#print("Test Accurary:",100*ts_correct.cpu().numpy()/np.int(ts_total))
print("Test Accurary: {:.4f} %".format(100*ts_correct/ts_total))

