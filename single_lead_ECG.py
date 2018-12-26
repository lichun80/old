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
#############################################

## Load img&label array   
with open('/home/desktop/thesis/ls_ilabel.pkl','rb') as file:
	label_f = pickle.load(file)
np_label = np.array(label_f)

np_img = np.load('/home/desktop/thesis/patient_lead.npy')

#print(np_label[-1])
#print(np_label.shape)           #(2540,)
#print(np.unique(np_label))
#print(np_img.shape)             #(2540,1,12,5000)
#print(np_img[-1])

## Split dataset
x = np_img
y = np_label

np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#print(x_train.shape)         #(2032,1,12,5000)
#print(x_test.shape)          #(508,1,12,5000)
#print(y_train.shape)         #(2032,)
#print(y_test.shape)	     #(508,)

##############################################
## Hyper Parameters
Batch_size = 16
learning_rate = 1e-4
num_epoch = 20


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
tr_dataloader = DataLoader(tr_dataset, Batch_size, shuffle=True)
ts_dataloader = DataLoader(ts_dataset, Batch_size, shuffle=False)
#print(tr_dataset.__len__())
#print(len(ts_dataset))
#print(next(iter(tr_dataloader)))
#print(next(iter(ts_dataloader)))

# Define  Model
class CNN_model(torch.nn.Module):
	def __init__ (self):
		super(CNN_model, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1,out_channels=16 ,kernel_size=3 , stride=1 ,padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.MaxPool2d(2))
		self.conv2 = nn.Sequential(
			nn.Conv2d(16,32,3,1,1),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.MaxPool2d(2))
		self.conv3 = nn.Sequential(
			nn.Conv2d(32,64,3,1,1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.AvgPool2d(2))
		self.dense = nn.Sequential(
			nn.Linear(40000,2))
			#nn.ReLU(),
			#nn.Dropout(p = 0.5),
			#nn.Linear(1024,2))
	def forward(self, x):
#		print(x.shape)
		x = self.conv1(x)
#		print(x.shape)
		x = self.conv2(x)
#		print(x.shape)
		x = self.conv3(x)
#		print(x.shape)
		fc_input = x.view(x.size(0),-1)  #reshape tensor
#		print(fc_input.shape)
		fc_output = self.dense(fc_input)
#		print(fc_output.shape)
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
tr_total = 0
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
		loss = criterion(out,target)
		_,pred = torch.max(out.data,1)
		#print("out", out.shape, "\n", out)
		#print("target", target.shape, "\n", target)
		#print("pred", pred.shape, "\n", pred)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()	

		#print(pred)
		#print(target)
		#print(pred == target)
		#print(torch.sum(pred == target))
		#tr_loss += loss.item()
		tr_total += len(x_train)
		tr_correct += torch.sum(pred == target)
		#print(torch.sum(pred == target))
		#print(pred)
		#print(target)
		
		
#	if (epoch+1) %2 == 0:
#		print("Epoch:{}/{}".format(epoch+1,num_epoch))
#		print("Train Accuracy is :{:.4f}".format(tr_correct.cpu().numpy()/np.int(tr_total)))

#print(tr_total)
#print(tr_correct)
print("Train Accurary:",tr_correct.cpu().numpy()/np.int(tr_total))

## Testing Model
model.eval()

ts_total = 0
ts_correct = 0
for epoch in range(num_epoch):
	for i,x_test,y_test in ts_dataloader:
		if torch.cuda.is_available():
			input  = Variable(x_test.float()).cuda()
			target  = Variable(y_test).cuda()	
		else:
			input = Variable(x_test.float()).cuda
			target = Variable(y_test)	

		out = model(input)
		loss = criterion(out,target)

		_,pred = torch.max(out.data,1)
#		ts_loss += loss.item()
		ts_total += len(x_test)
		ts_correct += torch.sum(pred == target.data)

print("\nTesting Model...")
#print("Test loss is :{:.4f}, Test Accuracy is :{:.4f}".format(ts_loss/len(ts_dataset),ts_acc))
#print("Test Accuracy is :{:.4f}".format(ts_correct.cpu().numpy()/np.int(ts_total)))

#print(ts_total)
#print(ts_correct)
print("Test Accurary:",ts_correct.cpu().numpy()/np.int(ts_total))

