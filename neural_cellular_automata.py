import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeurAutoNet(nn.Module):
	def __init__(self):
		super(NeurAutoNet, self).__init__()
		self.set_modules()

	def set_modules(self):
		sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
		sobel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

		self.C_S_x = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1, bias = False, groups = 16)
		self.C_S_x.weight = nn.Parameter(sobel_x.unsqueeze(0).repeat(16, 1, 1, 1), requires_grad = False)

		self.C_S_y = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1, bias = False, groups = 16)
		self.C_S_y.weight = nn.Parameter(sobel_y.unsqueeze(0).repeat(16, 1, 1, 1), requires_grad = False)

		self.D1 = nn.Linear(48, 128)
		nn.init.kaiming_normal(self.D1.weight)

		self.D2 = nn.Linear(128, 16)
		self.D2.weight = nn.Parameter(torch.zeros(16, 128) + 1e-12)

		self.Pool = nn.MaxPool2d(3, stride = 1, padding = 1)

	def forward(self, x):
		x_i = x

		x1 = self.C_S_x(x).permute(0, 2, 3, 1).view(x.size(0), -1, 16)	#SobelX
		x2 = self.C_S_y(x).permute(0, 2, 3, 1).view(x.size(0), -1, 16)	#Sobel Y
		x3 = x.permute(0, 2, 3, 1).view(x.size(0), -1, 16)				#Identity
		x = torch.cat([x1, x2, x3], dim = 2)

		x = F.relu(self.D1(x))
		x = self.D2(x)
		x = x.view(x_i.shape)
		x = x_i + F.dropout(x, p = 0.2)

		alpha = self.Pool(x[:, 3]) > 0.1
		alpha = alpha.type(torch.FloatTensor)
		x[:, 3] = alpha * x[:, 3]
		return x

	def norm_gradients(self):
		x1, x2 = self.D1.weight.grad, self.D2.weight.grad
		x1_norm, x2_norm = torch.norm(x1), torch.norm(x2)
		#self.D1.weight.grad = (x1 - x1.mean())/(x1.std() + 1e-12)
		#self.D2.weight.grad = (x2 - x2.mean())/(x2.std() + 1e-12)
		self.D1.weight.grad = x1/x1_norm
		self.D2.weight.grad = x2/x2_norm
		return

#Let's load that image bois########################################

img = cv2.imread("appletun.PNG")
img = cv2.resize(img, (32, 32))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

alpha = np.zeros((32, 32))

for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		s = img[i,j].mean()
		if s < 1.0:
			alpha[i][j] = 1.0

img = np.transpose(img, (2, 0, 1))

target_img = np.ones((16, 32, 32))
target_img[:3] = img
target_img[3] = alpha

#Quick check if the alpha-ing worked #####
#test = np.transpose(target_img, (1, 2, 0))
#plt.imshow(test[:, :, 3])
#plt.show()

#Make that seed #####################################################

alpha = np.zeros((32, 32))
alpha[15, 15] = 1
initial_seed = np.ones((16, 32, 32))
initial_seed[:3, 15, 15] = 0
initial_seed[3] = alpha

target_img, initial_seed = torch.Tensor([target_img]), torch.Tensor([initial_seed])
initial_seed = initial_seed.repeat(4, 1, 1, 1)

#Setup that network#################################################
net = NeurAutoNet()
optimizer = optim.Adam(net.parameters(), lr = 0.0001)

iterations = 10

for i in range(iterations):
	net.zero_grad()
	out = initial_seed

	for k in range(64):
		out = net(out)
		#print(out[:, :3].max(), out[:, :3].min())

	L = torch.mean((out[:, :4] - target_img[:, :4])**2)
	L.backward()

	net.norm_gradients()

	optimizer.step()

	#if i%10 == 0:
	print("Loss: ", L.item())

plt.imshow(out[:, 3].detach().permute(1, 2, 0))
plt.show()