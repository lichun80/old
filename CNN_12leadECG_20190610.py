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

np.random.seed(1)
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

#print(x_train.shape)         #(1778,1,12,5000)
#print(x.shape)          #(762,1,12,5000)
#print(y_train.shape)         #(1778,)
#print(y_test.shape)          #(762,)
#print("x_train = ", x_train)
#print("y_train = ", y_train)

##################################################################
## Hyper Parameters
Batch_size = 32
learning_rate = 0.001
num_epochs = 30

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

# Define  Model
class CNN(torch.nn.Module):
        def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=1,out_channels=16 ,kernel_size=3 , stride=1 ,padding=1),
                        nn.InstanceNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
                self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels=16,out_channels=32 ,kernel_size=3 , stride=1 ,padding=1),
                        nn.InstanceNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
                self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels=32,out_channels=64 ,kernel_size=3 , stride=1 ,padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
                self.conv4 = nn.Sequential(
                        nn.Conv2d(in_channels=64,out_channels=128 ,kernel_size=3 , stride=1 ,padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
                self.conv5 = nn.Sequential(
                        nn.Conv2d(in_channels=128,out_channels=256 ,kernel_size=3 , stride=1 ,padding=1),
                        nn.InstanceNorm2d(256),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))


                self.dense = torch.nn.Sequential(
                        nn.Linear(40000,2))
        
        def forward(self,x):
                #print(x.shape)
                x = self.conv1(x)
                #print(x.shape)
                x = self.conv2(x)
                #print(x.shape)
                x = self.conv3(x)
                #print(x.shape)
                x = self.conv4(x)
                #print(x.shape)
                x = self.conv5(x)
                #print(x.shape)

                fc_input = x.view(x.size(0),-1)  #reshape tensor
                #print(fc_input.shape)
                fc_output = self.dense(fc_input)
                #print(fc_output.shape)
                return fc_output



if torch.cuda.is_available():
        model = CNN().cuda()
else:
        model = CNN()

#print(model)


## Loss Function & Optimization
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## early stop
loss_history = []
count = 0

## Training Model
total_step = len(tr_dataloader)
tr_total = 0
tr_correct = 0
print("\nTraining Model...")
for epoch in range(num_epochs):
        for i, x_train, y_train in tr_dataloader:
                #print(x_train.shape, y_train.shape)
                #print("i = ", i)
                #print("x_train =", x_train)
                #print("y_train =", y_train)
                if torch.cuda.is_available():
                        input = Variable(x_train.float(), requires_grad=True).cuda()
                        target = Variable(y_train).cuda()
                else:
                        input = Variable(x_train.float(), requires_grad=True)
                        target = Variable(y_train)
                #print(input.shape)
                #print(target.shape)
                
                output = model(input)
                
                loss = criterion(output,target)
                _,pred = torch.max(output.data,1)
                #print("output", output.shape, "\n", output)
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
#for epoch in range(num_epoch):
for i,x_test,y_test in ts_dataloader:
        if torch.cuda.is_available():
                input  = Variable(x_test.float()).cuda()
                target  = Variable(y_test).cuda()
        else:
                input = Variable(x_test.float())
                target = Variable(y_test)

        output = model(input)
        loss = criterion(output,target)
#       print(out.data)
        _,pred = torch.max(output.data,1)
#       print(pred)
        ts_loss = loss.item()
        ts_total += target.size(0)
        ts_correct += (pred == target).sum().item()

        if (epoch + 1) % 5 == 0:
                print('num_epochs [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, ts_loss))
                #print("pred",pred)
                #print("targer",target)

#print("Test Accurary:",100*ts_correct.cpu().numpy()/np.int(ts_total))
print("Test Accurary: {:.4f} %".format(100*ts_correct/ts_total))

exit()


def plot_acc_loss(tr_loss, ts_loss, tr_acc, ts_acc):
    
    # loss
    plt.figure(figsize=(20,6))
    plt.title('Learning Curve')
    plt.xlabel('num_epochs')
    plt.ylabel('crossentropy')
    plt.plot(tr_loss, label = 'training loss')
    plt.plot(ts_loss, label = 'test loss')
    plt.legend()
    plt.show()

    # accuracy
    plt.figure(figsize=(20,6))
    plt.title('Accuracy')
    plt.xlabel('num_epochs')
    plt.ylabel('Accuracy')
    plt.plot(tr_acc, label = 'training acc')
    plt.plot(ts_acc, label = 'test acc')
    plt.legend()
    plt.show()