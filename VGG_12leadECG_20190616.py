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
num_epoch = 300

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
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),      
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),     
            # nn.MaxPool2d(kernel_size=2, stride=2))      

        self.classifier = nn.Sequential(
            nn.Linear(159744, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024,1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024,2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if torch.cuda.is_available():
        model = VGG().cuda()
else:
        model = VGG()

#print(model)


## Loss Function & Optimization
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(True), lr=learning_rate)

## Training Model
total_step = len(tr_dataloader)
tr_total = 0
tr_correct = 0
tr_loss = 0.0
print("\nTraining Model...")
for epoch in range(num_epoch):
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
                
                output = model(input)
                loss = criterion(output,target)
                _,pred = torch.max(output.data,1)
                #print("out", out.shape, "\n", out)
                #print("target", target.shape, "\n", target)
                #print("pred", pred.shape, "\n", pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                tr_total += len(x_train)
                tr_correct += torch.sum(pred == target)
                
                #if (i+1) % 50  == 0:
                #        print("Epoch:[{}/{}], Step[{}/{}], Loss:{:.4f}".format(epoch+1,num_epoch, i+1, total_step, tr_loss))
                #        print("Train Accuracy is :{:.4f}".format(tr_correct.cpu().numpy()/np.int(tr_total)))
        
#print(tr_total)
#print(tr_correct)
#        print("Loss is:{:.4f}, Train Accurary is:{:.4F}%".format(loss/(len(data), 100*loss/len(data))))
        
print("Train Accurary:",tr_correct.cpu().numpy()/np.int(tr_total))

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
#       ts_loss += loss.item()
        ts_total += len(x_test)
        ts_correct += torch.sum(pred == target.data)
        print("pred:", pred)
        print("label",target)

#print(ts_total)
#print(ts_correct)
print("Test Accurary:",ts_correct.cpu().numpy()/np.int(ts_total))

