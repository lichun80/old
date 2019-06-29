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
from sklearn.metrics import confusion_matrix

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))
#print(torch.cuda.current_device())
#############################################

## Load img & label array   
with open('/home/desktop/thesis/labels.pkl','rb') as file:
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
#x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

kf = KFold(n_splits=10)
#
for train_index, val_index in kf.split(x):
    #print("Trian:",train_index ,"val:",val_index)
    x_train ,x_val = x[train_index],x[val_index]
    y_train ,y_val = y[train_index],y[val_index]

#print(x_train.shape)         #(2286,1,12,5000)
#print(x_test.shape)          
#print(y_train.shape)         #(2286,)
#print(y_test.shape)          

##################################################################
## Hyper Parameters
Batch_size = 64
learning_rate = 0.001
num_epochs = 100

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
tv_dataset = ecg_dataset(x_val,y_val)
tr_dataloader = DataLoader(tr_dataset, Batch_size,shuffle=True)
tv_dataloader = DataLoader(tv_dataset, Batch_size,shuffle=False)

#print(tr_dataset.__len__())
#print(len(tv_dataset))
#print(next(iter(tr_dataloader)))
#print(next(iter(tv_dataloader)))

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
                self.conv6 = nn.Sequential(
                        nn.Conv2d(in_channels=256,out_channels=512 ,kernel_size=3 , stride=1 ,padding=1),
                        nn.InstanceNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
                self.conv7 = nn.Sequential(
                        nn.Conv2d(in_channels=512,out_channels=1024 ,kernel_size=3 , stride=1 ,padding=1),
                        nn.InstanceNorm2d(1024),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))

                self.dense = torch.nn.Sequential(
                        nn.Linear(39936,2))
        
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
                x = self.conv6(x)
                #print(x.shape)
                x = self.conv7(x)
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


        # loss_history.append(tr_loss)
        # num_early_stop = 5
        # if(num_early_stop>0):
        #         if(len(loss_history)>1 and loss_history[-1]>loss_history[-2]):
        #             count += 1
        #             if count>num_early_stop:
        #                 break


#print("Train Accurary:",tr_correct.cpu().numpy()/len(tr_dataset))

print("Train Accurary: {:.4f} %".format(100*tr_correct/tr_total))

confusion = confusion_matrix(target.cpu(), pred.cpu())
print("Confusion Maxtrix", confusion)
# True Positives
TP = confusion[1, 1]
# True Negatives
TN = confusion[0, 0]
# False Positives
FP = confusion[0, 1]
# False Negatives
FN = confusion[1, 0]

print("Train Sensitivity : {:.4f} %".format(100* TP / float(TP + FN)))
print("Train Specificity : {:.4f} %".format(100* TN / float(TN + FP)))

#############################################################
print("\nVailding Model...")
## Vailding Model
model.eval()
tv_total = 0
tv_correct = 0
#for epoch in range(num_epoch):
for i,x_val,y_val in tv_dataloader:
        if torch.cuda.is_available():
                input  = Variable(x_val.float()).cuda()
                target  = Variable(y_val).cuda()
        else:
                input = Variable(x_val.float())
                target = Variable(y_val)

        output = model(input)
        loss = criterion(output,target)
        _,pred = torch.max(output.data,1)
#       print(pred)
        tv_loss = loss.item()
        tv_total += target.size(0)
        tv_correct += (pred == target).sum().item()

        #if (epoch + 1) % 5 == 0:
        print('num_epochs [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, tv_loss))
        #print("pred",pred)
        #print("targer",target)

#print("Vaild Accurary:",100*ts_correct.cpu().numpy()/np.int(ts_total))
print("Vaild Accurary: {:.4f} %".format(100*tv_correct/tv_total))

## Cofusion Matrix
confusion = confusion_matrix(target.cpu(), pred.cpu())
print("Confusion Maxtrix", confusion)

# True Positives
TP = confusion[1, 1]
# True Negatives
TN = confusion[0, 0]
# False Positives
FP = confusion[0, 1]
# False Negatives
FN = confusion[1, 0]

print("Vaild Sensitivity : {:.4f} %".format(100* TP / float(TP + FN)))
print("Vaild Specificity : {:.4f} %".format(100* TN / float(TN + FP)))
