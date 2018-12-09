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
from torch.autograd import Variable

#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))
#print(torch.cuda.current_device())
##################################3###########

## Load img&label array   
with open('/home/desktop/thesis/ls_ilabel.pkl','rb') as file:
	label_f = pickle.load(file)
np_label = np.array(label_f)

np_img = np.load('/home/desktop/thesis/patient_lead.npy')

#print(np_label[-1])
print(np_label.shape)
#print(np.unique(np_label))
print(np_img.shape)
#print(np_img[-1])

## Split dataset
x = np_img
y = np_label

np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

##############################################
## Hyper Parameters
Batch_size = 16
learning_rate = 1e-4
num_epoch = 10


##  Dataset
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

## DataLoader
tr_dataset = ecg_dataset(x_train,y_train)
ts_dataset = ecg_dataset(x_test,y_test)
tr_dataloader = DataLoader(tr_dataset, Batch_size, shuffle=True)
ts_dataloader = DataLoader(ts_dataset, Batch_size, shuffle=False)
#print(next(iter(tr_dataloader)))
#print(next(iter(ts_dataloader)))


# Define  Model
class CNN_model(torch.nn.Module):
	def __init__ (self):
		super(CNN_model, self).__init__()
		# neework layer
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1,out_channels=32 ,kernel_size=3 , stride=1 ,padding=1),
			nn.ReLU(True),
			nn.AvgPool2d(2))
		self.conv2 = nn.Sequential(
			nn.Conv2d(32,64,3,1,1),
			nn.ReLU(),
			nn.AvgPool2d(2,15))
		#self.con3 = nn.Sequential(
		#	nn.Conv2d(128,,5,2,1),
		#	nn.ReLU(),
		#	nn.AvgPool2d(2))
		self.dense = nn.Sequential(
			nn.Linear(167*64,512),
			nn.ReLU(),
			nn.Dropout(p = 0.5),
			nn.Linear(512,2))
	def forward(self, x):
#		print(x.shape)
		x = self.conv1(x)
#		print(x.shape)
		x = self.conv2(x)
#		print(x.shape)
		fc_input = x.view(x.size(0),-1)  #reshape tensor
		#print(fc_input.shape)
		fc_output = self.dense(fc_input)
		#print(fc_output)
		return fc_output
		

if torch.cuda.is_available():
  model = CNN_model().cuda()
else:
  model = CNN_model()

#print(model)

## Loss Function & Optimization
criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum = 0.9) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

## Training Model
tr_loss = 0.0
tr_correct = 0
print("\nTrainging Model...")
for epoch in range(num_epoch):
	for i, x_train, y_train in tr_dataloader:
		#print(x_train.shape, y_train.shape)
		if torch.cuda.is_available():
			input = Variable(x_train.float(), requires_grad=True).cuda()
			target = Variable(y_train).cuda()
		else:
      			input = Variable(x_train.float(), requires_grad=True)
      			target = Variable(y_train)
		out = model(input)
		#print(out.data)
		_,pred = torch.max(out.data,1)
		#print("out", out.shape, "\n", out)
		#print("target", target.shape, "\n", target)
		loss = criterion(out, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()	
		#tr_loss += loss.data[0]
		#print(pred)
		#print(target)
		tr_loss += loss.item()	
		#print(target.data)
		#print(pred)
		tr_correct += torch.sum(pred == target.data)
		#print(int(tr_correct)
		
		tr_acc =100 * tr_correct / len(tr_dataset)

	if(epoch+1) %2 == 0:
		print("Epoch:{}/{}".format(epoch+1,num_epoch))
		print("Train loss is :{:.4f}, Train Accuracy is :{:.2f}%".format(tr_loss/len(tr_dataset),tr_acc))


exit()

## Testing Model
model.eval()

ts_loss = 0.0
ts_correct = 0.0
for i,x_test,y_test in ts_dataloader:
	if torch.cuda.is_available():
		input  = Variable(x_test.float()).cuda()
		target = Variable(y_test).cuda()	
	else:
		input = Variable(x_test.float()).cuda
		target = Variable(y_test)	

	out = model(input)
	_,pred = torch.max(out.data,1)
	loss = criterion(out,target)
	
	ilabel = int(target[0][0])
	iprediction = int(pred)	
	print(ilabel)
	print(iprediction) 

	ts_loss += loss.item()
	tr_correct += torch.sum(pred == target.data)

print("\nTesting Model...")
print("Test loss is :{:.4f}, Test Accuracy is :{:.2f}%".format(ts_loss/len(ts_dataset),ts_correct/len(ts_dataset)))


