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

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
##################################

# load label array   
with open('/home/desktop/thesis/ls_ilabel_AF_AFL.pkl','rb') as file:
	label_f = pickle.load(file)
np_label = np.array(label_f)
#print(np_label)
print(np_label.shape)
print(np.unique(np_label))

# load numpy array 
np_img = np.load('/home/desktop/thesis/patient_lead_AF_AFL.npy')
print(np_img.shape)

x = np_img
y = np_label


# Make Dataset
class ecg_dataset(Dataset):
    def __init__(self,x,y,transform=None):   #initial processes,reading data
      super(ecg_dataset, self).__init__()
      self.x = x
      self.y = y 
      #print(self.x)
      #print(self.y) 

    def __getitem__ (self, idx):       #return one item on the index 

      return idx, self.x[idx], self.y[idx]
    
    def __len__(self):   #return the data length
      return len(self.x)

def train():


# creat DataLoader
r_dataset = ecg_dataset(x,y)
r_dataloader = DataLoader(dataset=r_dataset, batch_size=16, shuffle=True, num_workers=1)


# Define Model
class CNN_model(torch.nn.Module):
  def __init__ (self):
    super(CNN_model, self).__init__()
    # network layer
    layer1 = nn.Sequential()
    layer1.add_module('conv1',nn.Conv2d(in_channels=1,out_channels=32 ,kernel_size=3 , stride=1 ,padding=1)) 
    layer1.add_module('relu1',nn.ReLU(True))
    layer1.add_module('pool1', nn.MaxPool2d(2,2))
    self.layer1 = layer1

  def forward(self, x):
    conv1 = self.layer1(x)
    return out

if torch.cuda.is_available():
  model = CNN_model().cuda()
else:
  model = CNN_model()

print(model)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum = 0.9) 

#j=0
#for i, x_train, y_train in r_dataloader:
#  j+=1
#  print(type(x_train))

#print(j)



exit()
for epoch in range(100):
  for i, x_train, y_train in r_dataloader:
    #print(type(x_train)
    #print(type(y_train))
  
    if torch.cuda.is_available():
      input = torch.autograd.Variable(x_train).cuda()
      target = torch.autograd.Variable(y_train).cuda()
    else:
      input = torch.autograd.Variable(x_train)
      target = torch.autograd.Variable(y_train)
    out = model(input)
    loss = criterion(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if(epoch+1) % 10 == 0:
    print('{}/100, loss:{:.6f}' .format(epoch+1, loss.data[0]))


















