#!/home/desktop/anaconda3/bin/python3.6
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available())  
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())

x = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59], [2.167], [7.042], [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53], [1.221], [2.827], [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

#plt.figure()
#plt.scatter(x, y, marker="o")
#plt.show()

np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4) 

'''
np.random.seed(1234)
#total = np.arange(len(x))
#print(total)
np.random.shuffle(x)
np.random.shuffle(y)
#print(x)
#print(y)

x_train = x[:12] 
y_train = y[:12]
x_test = x[-3:]
y_test = y[-3:]
'''
#print(sorted(x_train))
#print(sorted(y_train))
#print(x_test)
#print(y_test)

class dataset(Dataset):
    def __init__(self, x, y):
      super(dataset, self).__init__()
      self.x = x
      self.y = y
      #print(0)
      #print(self.x_train)
      #print(self.y_train)
    def __getitem__ (self, idx):
      #print(111)
      return idx, self.x[idx], self.y[idx]
    
    def __len__(self):
      #print(222)
      return len(self.x)

class LR(torch.nn.Module):
  def __init__ (self):
    super(LR, self).__init__()
    self.linear = torch.nn.Linear(1,1)  #input and output is 1 dimension
    #self.linear1 = torch.nn.Linear(1,3)
    #self.linear2 = torch.nn.Linear(3,1)
  def forward(self, x):
    out = self.linear(x)
    #out = self.linear1(x)
    #out = self.linear2(out)
    return out

#print("call class")
train_dataset = dataset(x_train, y_train)
test_dataset = dataset(x_test, y_test)
#print("call array 0", r_dataset[0])
#print("Length", len(r_dataset))
#print(r_dataset.__getitem__(1))

tr_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=2)
ts_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=True, num_workers=2)
#for epoch in range(2):
#  for i, x_train, y_train in r_dataloader:
#    print(epoch, i)
#exit()
#print (r_dataloader.__len__())

if torch.cuda.is_available():
  model = LR().cuda()
else:
  model = LR()

criterion = torch.nn.MSELoss()   #define loss function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  #define optimizer

for epoch in range(100):
	for i, x_train, y_train in tr_dataloader:
	#print(i, x_train)
	#print(type(x_train))
		if torch.cuda.is_available():
			input = torch.autograd.Variable(x_train).cuda()
			target = torch.autograd.Variable(y_train).cuda()
		else:
			input = torch.autograd.Variable(x_train)
			target = torch.autograd.Variable(y_train)
		#forward
		out = model(input)
		loss = criterion(out, target)  
		#backward
		optimizer.zero_grad() #reset gradients 
		loss.backward()  #backward pass
		optimizer.step()  # update parameters

	if(epoch+1) % 10 == 0:
		print('Epoch[{}/100], loss:{:.6f}' .format(epoch+1, loss.data[0]))

#Testing Model
i, x_test, y_test = test_dataset[:]
model.eval()   #predict mode
if torch.cuda.is_available():
  predict = model(torch.autograd.Variable(torch.from_numpy(x_test)).cuda())
predict = predict.data.cpu().numpy()


plt.plot(x_test, y_test, 'ro', label="Ori.")
plt.plot(x_test, predict, label='Fitting')
plt.show()


