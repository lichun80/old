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
from sklearn.model_selection import train_test_split


#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))
#print(torch.cuda.current_device())
##################################

## load label array   
with open('/home/desktop/thesis/ls_ilabel.pkl','rb') as file:
	label_f = pickle.load(file)
np_label = np.array(label_f)
#print(np_label)
#print(np_label.shape)
#print(np.unique(np_label))

## load numpy array 
np_img = np.load('/home/desktop/thesis/patient_lead.npy')
#print(np_img.shape)


x = np_img
y = np_label


np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

#print(x_train.shape)
#print(x_val.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_val.shape)
#print(y_test.shape)



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


# creat DataLoader
tr_dataset = ecg_dataset(x_train,y_train)
ts_dataset = ecg_dataset(x_test,y_test)
tr_dataloader = DataLoader(tr_dataset, batch_size=16, shuffle=True, num_workers=1)
#print(next(iter(tr_dataloader)))

# Define Model
class CNN_model(torch.nn.Module):
	def __init__ (self):
		super(CNN_model, self).__init__()
		# network layer
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=1,out_channels=16 ,kernel_size=3 , stride=1 ,padding=1
			),
			nn.ReLU(True),
			nn.MaxPool2d(2,2)
		)
		print(self.conv1)
#		self.conv2 = nn.Sequential(
#			nn.Conv2d(16,32, 3, 1, 1),
#			nn.ReLU(True),
#			nn.MaxPool2d(2)
#		)
		self.out = nn.Linear(240000,2)
			
		
	def forward(self, x):
		x = self.conv1(x)
#		x = self.conv2(x)
		x = x.view(x.size(0),-1)
		output = self.out(x)
		return output, x

if torch.cuda.is_available():
  model = CNN_model().cuda()
else:
  model = CNN_model()

#print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum = 0.9) 

#j=0
#for i, x_train, y_train in r_dataloader:
#  j+=1
#  print(type(x_train))

#print(j)

for epoch in range(1):
	for i, x_train, y_train in tr_dataloader:
		#print(x_train.shape, y_train.shape)
	#print(type(y_train))
		if torch.cuda.is_available():
			input = torch.autograd.Variable(x_train.float(), requires_grad=True).cuda()
			target = torch.autograd.Variable(y_train).cuda()
		else:
      			input = torch.autograd.Variable(x_train.float(), requires_grad=True)
      			target = torch.autograd.Variable(y_train)
		#print(input.shape,target.shape)

		out = model(input)
		exit()
		#print(out, target)
		loss = criterion(out, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if(epoch+1) % 10 == 0:
		print('{}/100, loss:{:.6f}' .format(epoch+1, loss.data[0]))

#Testing Model
i, x_test, y_test = ts_dataset[:]
model.eval()
if torch.cuda.is_available():
  predict = model(torch.autograd.Variable(torch.from_numpy(x_test)).cuda())
predict = predict.data.cpu().numpy()


plt.plot(x_test, y_test, 'ro', label="Ori.")
plt.plot(x_test, predict, label='Fitting')
plt.show()


