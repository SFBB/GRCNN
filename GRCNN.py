import torch
import math
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules import padding


class BidirectionalLSTM(nn.Module):

	def __init__(self, nIn, nHidden, nOut):
		super(BidirectionalLSTM, self).__init__()

		self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
		self.embedding = nn.Linear(nHidden * 2, nOut)

	def forward(self, input):
		recurrent, _ = self.rnn(input)
		T, b, h = recurrent.size()
		t_rec = recurrent.view(T * b, h)

		output = self.embedding(t_rec)  # [T * b, nOut]
		output = output.view(T, b, -1)

		return output


# 学长
class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size=128, batch_size=-1, num_layers=2, bidirectional=True):
		super(LSTM,self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size
		if bidirectional:
			self.num_layers = num_layers * 2
		else:
			self.num_layers = num_layers
		self.batch_size = batch_size
		self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.6, bidirectional=bidirectional)
		self.embedding = nn.Linear(hidden_size * 2, 128)
		#self.hidden = self.init_hidden()

	# def init_hidden(self):
	#     if torch.cuda.is_available():
	#         return (Variable(torch.randn(self.num_layers,self.batch_size,self.hidden_size)).cuda(),
	#                 Variable(torch.randn(self.num_layers,self.batch_size,self.hidden_size)).cuda())
	#     else:
	#         return (Variable(torch.randn(self.num_layers,self.batch_size,self.hidden_size)),
	#                 Variable(torch.randn(self.num_layers,self.batch_size,self.hidden_size)))

	def forward(self, inputs, hidden=None):
		# print("inputs shape is {}".format(inputs.shape()))
		recurrent, hidden = self.rnn(inputs, hidden)
		T, b, h = recurrent.size()
		t_rec = recurrent.view(T * b, h)
		output = self.embedding(t_rec)  # [T * b, nOut]
		output = output.view(T, b, -1)
		#self.hidden = hidden
		return output


class GRCL(nn.Module):
	def __init__(self, in_channels, out_channels, n_iter = 3, kernel_size=3, padding=(1, 1), stride=(1, 1)):
		super(GRCL, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.n_iter = n_iter

		self.conv_r = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
		self.conv_g_r = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

		self.conv_f = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
									nn.BatchNorm2d(out_channels))

		self.conv_g_f = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
										 nn.BatchNorm2d(out_channels))

		self.bn_rec = nn.ModuleList()
		self.bn_gate_rec = nn.ModuleList()
		self.bn_gate_mul = nn.ModuleList()
		for ii in range(n_iter):
			self.bn_rec.append(nn.BatchNorm2d(out_channels))
			self.bn_gate_rec.append(nn.BatchNorm2d(out_channels))
			self.bn_gate_mul.append(nn.BatchNorm2d(out_channels))



	def forward(self, x):
		conv_gate_f = self.conv_g_f(x)
		bn_f = self.conv_f(x)
		x = F.relu(bn_f)

		for ii in range(self.n_iter):
			c_gate_rec = self.bn_gate_rec[ii](self.conv_g_r(x))
			gate = F.sigmoid(conv_gate_f + c_gate_rec)

			c_rec = self.bn_rec[ii](self.conv_r(x))
			x = F.relu(bn_f + self.bn_gate_mul[ii](c_rec*gate))

		return x

# https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
# https://blog.csdn.net/u012348774/article/details/109293450
class GRUConvCell(nn.Module):

	def __init__(self, input_channel, output_channel, hidden_channel, kernel_size, stride):

		super(GRUConvCell, self).__init__()

		self.W_z = nn.Conv2d(input_channel, output_channel, kernel_size, stride)
		self.U_z = nn.Conv2d(hidden_channel, output_channel, kernel_size, stride)
		# self.



		# filters used for gates
		gru_input_channel = input_channel + output_channel
		self.output_channel = output_channel

		self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size=3, padding=1)
		self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)
		self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

		# filters used for outputs
		self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size=3, padding=1)
		self.output_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

		self.activation = nn.Tanh()

	# 公式1，2
	def gates(self, x, h):

		# x = N x C x H x W
		# h = N x C x H x W

		# c = N x C*2 x H x W
		c = torch.cat((x, h), dim=1)
		f = self.gate_conv(c)

		# r = reset gate, u = update gate
		# both are N x O x H x W
		C = f.shape[1]
		r, u = torch.split(f, C // 2, 1)

		rn = self.reset_gate_norm(r)
		un = self.update_gate_norm(u)
		rns = torch.sigmoid(rn)
		uns = torch.sigmoid(un)
		return rns, uns

	# 公式3
	def output(self, x, h, r, u):

		f = torch.cat((x, r * h), dim=1)
		o = self.output_conv(f)
		on = self.output_norm(o)
		return on

	def forward(self, x, h = None):

		N, C, H, W = x.shape
		HC = self.output_channel
		if(h is None):
			h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
		r, u = self.gates(x, h)
		o = self.output(x, h, r, u)
		y = self.activation(o)
		
		# 公式4
		return u * h + (1 - u) * y

class GRUNet(nn.Module):

	def __init__(self, hidden_size=64):

		super(GRUNet,self).__init__()

		self.gru_1 = GRUConvCell(input_channel=4,          output_channel=hidden_size)
		self.gru_2 = GRUConvCell(input_channel=hidden_size,output_channel=hidden_size)
		self.gru_3 = GRUConvCell(input_channel=hidden_size,output_channel=hidden_size)

		# 传统的FC
		self.fc = nn.Conv2d(in_channels=hidden_size,out_channels=1,kernel_size=3,padding=1)
		# 论文中提到的卷积层方法实现
		self.cnn_layers = nn.Sequential(
			# Defining a 2D convolution layer
			nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(4),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# Defining another 2D convolution layer
			nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(4),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)

	def forward(self, x, h):

		if h is None:
			h = [None,None,None]

		h1 = self.gru_1( x,h[0])
		h2 = self.gru_2(h1,h[1])
		h3 = self.gru_3(h2,h[2])

		o = self.cnn_layers(h3)

		return o,[h1,h2,h3]





# CNN Network for SNM features 最终版本
class CNN_SNM(nn.Module):
	def __init__(self):
		super().__init__()
		
		pass

	def forward(self, x):
		pass









# https://gist.github.com/daskol/05439f018465c8fb42ae547b8cc8a77b
class Maxout(nn.Module):
	"""Class Maxout implements maxout unit introduced in paper by Goodfellow et al, 2013.
	
	:param in_feature: Size of each input sample.
	:param out_feature: Size of each output sample.
	:param n_channels: The number of linear pieces used to make each maxout unit.
	:param bias: If set to False, the layer will not learn an additive bias.
	"""
	
	def __init__(self, in_features, out_features, n_channels, bias=True):
		super().__init__()
		
		self.in_features = in_features
		self.out_features = out_features
		self.n_channels = n_channels

		print("n_channels * out_features, in_features: ", n_channels * out_features, in_features)

		self.weight = nn.Parameter(torch.Tensor(n_channels * out_features, in_features))
		
		if bias:
			self.bias = nn.Parameter(torch.tensor(n_channels * out_features))
		else:
			self.register_parameter('bias', None)
			
		self.reset_parameters()
	
	def forward(self, input):
		print("input shape: ", input.shape)
		print("weight shape: ", self.weight.shape)
		a = nn.functional.linear(input, self.weight, self.bias)
		b = nn.functional.max_pool1d(a.unsqueeze(-3), kernel_size=self.n_channels)
		return b.squeeze()
	
	def reset_parameters(self):
		irange = 0.005
		nn.init.uniform_(self.weight, -irange, irange)
		if self.bias is not None:
			nn.init.uniform_(self.bias, -irange, irange)
	
	def extra_repr(self):
		return (f'in_features={self.in_features}, '
				f'out_features={self.out_features}, '
				f'n_channels={self.n_channels}, '
				f'bias={self.bias is not None}')

class ConvGRUCell(nn.Module):
	"""
	Generate a convolutional GRU cell
	"""

	def __init__(self, input_size, hidden_size, kernel_size):
		super().__init__()
		padding = kernel_size // 2
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.reset_gate = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
		self.reset_gate_ = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
		self.reset_maxout = Maxout(hidden_size, hidden_size/2, 3)
		self.update_gate = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
		self.update_gate_ = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
		self.update_maxout = Maxout(hidden_size, hidden_size/2, 3)
		self.out_gate = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
		self.out_gate_ = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
		self.out_maxout = Maxout(hidden_size, hidden_size/2, 3)



		init.orthogonal(self.reset_gate.weight)
		init.orthogonal(self.update_gate.weight)
		init.orthogonal(self.out_gate.weight)
		init.constant(self.reset_gate.bias, 0.)
		init.constant(self.update_gate.bias, 0.)
		init.constant(self.out_gate.bias, 0.)


	def forward(self, input_, prev_state):

		# get batch and spatial sizes
		batch_size = input_.data.size()[0]
		spatial_size = input_.data.size()[2:]

		# generate empty prev_state, if None is provided
		if prev_state is None:
			state_size = [batch_size, self.hidden_size] + list(spatial_size)
			if torch.cuda.is_available():
				prev_state = Variable(torch.zeros(state_size)).cuda()
			else:
				prev_state = Variable(torch.zeros(state_size))

		# data size is [batch, channel, height, width]
		# stacked_inputs = torch.cat([input_, prev_state], dim=1)
		# update = F.sigmoid(self.update_maxout(self.update_gate(stacked_inputs)))
		# reset = F.sigmoid(self.reset_maxout(self.reset_gate(stacked_inputs)))
		# # out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))

		z_t_n = F.sigmod(self.update_maxout(self.update_gate(input_)+self.update_gate_(prev_state)))
		r_t_n = F.sigmod(self.reset_maxout(self.reset_gate(input_)+self.reset_gate_(prev_state)))
		hidden_ = F.tanh(self.out_maxout(self.out_gate(input_)+self.out_gate_(r_t_n * prev_state)))

		new_state = z_t_n * prev_state + (1 - z_t_n) * hidden_

		return new_state

# https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
# https://www.oreilly.com/library/view/deep-learning-with/9781789534092/73698368-c6ba-4ec6-a84e-cba8417a2410.xhtml
# https://github.com/covarep/covarep/blob/5a2be5d6b776f14a0b275c69fde90eb13849e60d/feature_extraction/modified_group_delay_feature.m
class ConvGRU(nn.Module):

	def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
		'''
		Generates a multi-layer convolutional GRU.
		Preserves spatial dimensions across cells, only altering depth.
		Parameters
		----------
		input_size : integer. depth dimension of input tensors.
		hidden_sizes : integer or list. depth dimensions of hidden state.
			if integer, the same hidden size is used for all cells.
		kernel_sizes : integer or list. sizes of Conv2d gate kernels.
			if integer, the same kernel size is used for all cells.
		n_layers : integer. number of chained `ConvGRUCell`.
		'''

		super(ConvGRU, self).__init__()

		self.input_size = input_size

		if type(hidden_sizes) != list:
			self.hidden_sizes = [hidden_sizes]*n_layers
		else:
			assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
			self.hidden_sizes = hidden_sizes
		if type(kernel_sizes) != list:
			self.kernel_sizes = [kernel_sizes]*n_layers
		else:
			assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
			self.kernel_sizes = kernel_sizes

		self.n_layers = n_layers

		cells = []
		for i in range(self.n_layers):
			if i == 0:
				input_dim = self.input_size
			else:
				input_dim = self.hidden_sizes[i-1]

			cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
			name = 'ConvGRUCell_' + str(i).zfill(2)

			setattr(self, name, cell)
			cells.append(getattr(self, name))

		self.cells = cells


	def forward(self, x, hidden=None):
		'''
		Parameters
		----------
		x : 4D input tensor. (batch, channels, height, width).
		hidden : list of 4D hidden state representations. (batch, channels, height, width).
		Returns
		-------
		upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
		'''
		if not hidden:
			hidden = [None]*self.n_layers

		input_ = x

		upd_hidden = []

		for layer_idx in range(self.n_layers):
			cell = self.cells[layer_idx]
			cell_hidden = hidden[layer_idx]

			# pass through layer
			upd_cell_hidden = cell(input_, cell_hidden)
			upd_hidden.append(upd_cell_hidden)
			# update input_ to the last updated hidden layer for next pass
			input_ = upd_cell_hidden

		# retain tensors in list to allow different hidden sizes
		return upd_hidden


class GRCNN(nn.Module):
	def __init__(self, n_class=37):
		super(GRCNN, self).__init__()
		self.n_class = n_class
		self.conv_layer_1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
										  nn.BatchNorm2d(64), nn.ReLU())
		self.GRCL_layer_1 = GRCL(64, 64, kernel_size=3, stride=(1, 1), padding=(1, 1))
		self.GRCL_layer_2 = GRCL(64, 128, kernel_size=3, stride=(1, 1), padding=(1, 1))
		self.GRCL_layer_3 = GRCL(128, 256, kernel_size=3, stride=(1, 1), padding=(1, 1))

		self.conv_layer_2 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=0),
										  nn.BatchNorm2d(512), nn.ReLU())
		self.rnn = nn.Sequential(
			BidirectionalLSTM(512, 256, 256),
			BidirectionalLSTM(256, 256, self.n_class))

	def forward(self, x):
		x = self.conv_layer_1(x)
		x = F.max_pool2d(x, kernel_size=2, stride=2)
		x = self.GRCL_layer_1(x)
		x = F.max_pool2d(x, kernel_size=2, stride=2)
		x = self.GRCL_layer_2(x)
		x = F.max_pool2d(x	, kernel_size=2, stride=(2, 1), padding=(0, 1))
		x = self.GRCL_layer_3(x)
		x = F.max_pool2d(x, kernel_size=2, stride=(2, 1), padding=(0, 1))
		conv = self.conv_layer_2(x)

		b, c, h, w = conv.size()
		assert h == 1, "the height of conv must be 1"
		conv = conv.squeeze(2)
		conv = conv.permute(2, 0, 1)  # [w, b, c]
		conv = self.rnn(conv)
		return conv



# https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()

		# 输入的为音频，一维，所以使用1维的相关函数，输入数据为3维，分别为时间、频道和样本数
		self.cnn_layers = nn.Sequential(
			# Defining a 2D convolution layer 时域的卷积
			nn.Conv1d(1, 3, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm1d(3),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),
			# Defining another 2D convolution layer 频域卷积
			nn.Conv1d(3, 3, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm1d(3),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),
		)

		# 第一个参数要与输入的第二个参数相同。作用是将输入[x, y]根据参数[y, z]，转换为[x, z]
		self.linear_layers = nn.Sequential(
			nn.Linear(24000, 10)
		)

	def forward(self, x):
		x = self.cnn_layers(x)
		# print("x shape is {}".format(x.shape))
		# x = x.view(x.size(0), -1)
		# print("x shape is {}".format(x.shape))
		# x shape is torch.Size([16, 72000])
		x = self.linear_layers(x)
		return x



# # https://medium.com/@aungkyawmyint_26195/multi-layer-perceptron-mnist-pytorch-463f795b897a
# class DNN(nn.Module):
# 	def __init__(self):
# 		super(DNN,self).__init__()
# 		# number of hidden nodes in each layer (512)
# 		hidden_1 = 512
# 		hidden_2 = 512
# 		# linear layer (784 -> hidden_1)
# 		self.fc1 = nn.Linear(1280, 512)
# 		# linear layer (n_hidden -> hidden_2)
# 		self.fc2 = nn.Linear(512,512)
# 		# linear layer (n_hidden -> 10)
# 		self.fc3 = nn.Linear(512,2)
# 		# dropout layer (p=0.2)
# 		# dropout prevents overfitting of data
# 		self.droput = nn.Dropout(0.2)
# 		self.sm = nn.Sequential(
# 				nn.Softmax(dim=1),
# 			)

# 	def forward(self, x):
# 		# flatten image input
# 		# x = x.view(-1,16*1280)
# 		# print("x shape is {}".format(x.shape))
# 		# x shape is torch.Size([16, 1280])
# 		# add hidden layer, with relu activation function
# 		x = F.relu(self.fc1(x))
# 		# print("x shape is {}".format(x.shape))
# 		# add dropout layer
# 		x = self.droput(x)
# 		 # add hidden layer, with relu activation function
# 		x = F.relu(self.fc2(x))
# 		# add dropout layer
# 		x = self.droput(x)
# 		# add output layer
# 		x = self.fc3(x)
# 		x = self.sm(x)
# 		return x
		


class Model(nn.Module):
	def __init__(self, args, num_classes=2):
		super().__init__()
		self.args = args

		# self.convs = nn.Sequential(
		# 	CNN(),
		# 	LSTM(),
			
		# 	)

		self.grcnn = GRCNN()
		self.lstm = LSTM(3, hidden_size=64, batch_size=16, num_layers=2)
		# self.dnn = DNN()
		# self.convs = nn.Sequential(
		# 	MCNN(1, 32, ker_size=[127], stride=2),
		# 	nn.LeakyReLU(inplace=True),
		# 	nn.BatchNorm1d(32),

		# 	nn.MaxPool1d(2, 2),

		# 	MCNN(32, 75, ker_size=[63], stride=2),
		# 	nn.LeakyReLU(inplace=True),
		# 	nn.BatchNorm1d(75),

		# 	nn.MaxPool1d(2, 2),

		# 	MCNN(75, 40, ker_size=[7, 15, 31], stride=2),
		# 	nn.LeakyReLU(inplace=True),
		# 	nn.BatchNorm1d(120),

		# 	nn.MaxPool1d(2, 2),

		# 	MCNN(120, 128, ker_size=[7, 15], stride=2),
		# 	nn.LeakyReLU(inplace=True),
		# 	nn.BatchNorm1d(256),

		# 	nn.MaxPool1d(2, 2),

		# 	MCNN(256, 256, ker_size=[3, 7], stride=2),
		# 	nn.LeakyReLU(inplace=True),
		# 	nn.BatchNorm1d(512),

		# 	nn.MaxPool1d(2, 2),

		# 	MCNN(512, 512, ker_size=[3, 7]),
		# 	nn.LeakyReLU(inplace=True),
		# 	nn.BatchNorm1d(1024),

		# 	nn.MaxPool1d(2, 2),

		# 	MCNN(1024, 1024, ker_size=[1, 3]),
		# 	nn.LeakyReLU(inplace=True),
		# 	nn.BatchNorm1d(2048),
		# )

		# self.t_lstm = LSTM(t_lstm_input_size, hidden_size=64, batch_size=self.args.batch_size, num_layers=2)
		# self.t_lstm_hidden = self.t_lstm.init_hidden()

		# self.f_lstm = LSTM(f_lstm_input_size, hidden_size=64, batch_size=batch_size, num_layers=2)
		# self.f_lstm_hidden = self.f_lstm.init_hidden()

		# self.fcs = nn.Sequential(
		# 	nn.Dropout(p=0.75),
		# 	nn.Linear(dim_fc, 1500),
		# 	nn.LeakyReLU(inplace=True),
		# 	nn.BatchNorm1d(1500),

		# 	nn.Dropout(p=0.75),
		# 	nn.Linear(1500, 256),
		# 	nn.LeakyReLU(inplace=True),
		# 	nn.BatchNorm1d(256),
		# )
		
		# if args.loss == 'FocalLoss':
		# 	self.last_fc = nn.Sequential(
		# 		nn.Linear(256, num_classes),
		# 		nn.Softmax(dim=1),
		# 	)
		# else:
		# 	self.last_fc = nn.Sequential(
		# 		nn.Linear(256, num_classes),
		# 	)
		
		# self.init_weight()

	def forward(self, x):
		x = x.float()
		batch_size = x.shape[0]
		x = x.reshape(batch_size, 1, -1)

		if False:
			print(x.shape)
			x0 = x
			print('\n')
			for i in range(len(self.convs)):
				if i % 4 == 0:
					print('\n')

				x0 = self.convs[i](x0)
				print(x0.shape)
				

		x = self.grcnn(x)
		x = self.grcnn(x)
		x = self.grcnn(x)
		# x = x.view(batch_size, -1)
		### batch_size*nchannel*F*T -> T*batch_size*F*nchannel
		# x = x.transpose(1,3).contiguous()
		# x = x.transpose(0,1).contiguous()
		# x = x.permute(3, 0, 2, 1).contiguous()
		
		# print("x shape is {}".format(x.shape))
		tx = x.permute(2, 0, 1).contiguous()
		# fx = x.permute(2, 0, 3, 1).contiguous() # F * batch_size * T * nchannel
		del x
		
		
		# lstm_input: [T, batch_size, input_num]
		# t_lstm_out, t_lstm_hidden = self.t_lstm(tx, self.t_lstm_hidden)
		# print("tx shape is {}".format(tx.shape))
		t_lstm_out, _ = self.lstm(tx)
		# f_lstm_out, f_lstm_hidden = self.f_lstm(fx, self.f_lstm_hidden)
		

		# lstm_out: [T, batch_size, hidden_num]
		# self.t_lstm_hidden = (t_lstm_hidden[0].detach(), t_lstm_hidden[1].detach())
		# self.f_lstm_hidden = (f_lstm_hidden[0].detach(), f_lstm_hidden[1].detach())
		

		# print('1 lstm_out', lstm_out.size())
		# print("t_lstm_out shape is {}".format(t_lstm_out.shape))
		t_lstm_out = t_lstm_out.transpose(0,1).contiguous()
		t_lstm_out = t_lstm_out.reshape(batch_size, -1)
		feat = t_lstm_out
		feat = nn.functional.pad(feat, (0, 6016-1280))

		# f_lstm_out = f_lstm_out.transpose(0,1).contiguous()
		# f_lstm_out = f_lstm_out.reshape(batch_size, -1)

		# lstm_out = torch.cat([t_lstm_out, f_lstm_out], axis=1)
		# print("t_lstm_out shape is {}".format(t_lstm_out.shape))
		# x = self.dnn(t_lstm_out)

		# x = F.normalize(x, p=2, dim=1)
		# feat = x
		# x = self.last_fc(x)

		# if self.args.loss == 'CE' and not self.training:
		# 	x = F.softmax(x, dim=1)

		# w = self.last_fc.weight.data
		# w = self.last_fc.weight

		# print('output', y.size())
		# return x, feat, w
		return x, feat, None
		
	def init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# print('conv: ', m)
				# nn.init.kaiming_normal_(m.weight.data)
				
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				# print('linear: ', m)
				# nn.init.kaiming_normal_(m.weight.data)
				
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.zero_()


if __name__ == "__main__":
	print("Hello World!")
	input = torch.randn(10, 20, 30)
	print(input.shape)

	# maxout = Maxout(30, 15, 3, False)
	# output = maxout(input)
	# print(output.shape)
	# m = Model("")
	# m.train()
	# lstm = LSTM()
	# lstm.train()
	# dnn = DNN()
	# dnn.train()