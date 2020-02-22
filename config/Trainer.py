# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm


class Trainer(object):

	def __init__(self, 
				 model = None,
				 data_loader = None,
				 train_times = 1000,
				 alpha = 0.5,
				 use_gpu = True,
				 opt_method = "sgd",
				 save_steps = None,
				 checkpoint_dir = None):

		self.work_threads = 8  # 工作进程数
		self.train_times = train_times # 训练轮数

		self.opt_method = opt_method  # 优化器 的名字 字符串
		self.optimizer = None # 相对应的 优化器的对象
		self.lr_decay = 0 # 学习率衰减
		self.weight_decay = 0 # 是否L2正则化
		self.alpha = alpha # 学习率

		self.model = model  # 训练使用的模型（一般是负采样model）
		self.data_loader = data_loader # 一个对象，训练使用的数据集 ，产生一个 epoch 的数据，即 nbatches 个batches的数据
		self.use_gpu = use_gpu # 是否使用gpu
		self.save_steps = save_steps #
		self.checkpoint_dir = checkpoint_dir # 检查点保存地址

	#训练一个batch 数据的过程
	def train_one_step(self, data):
		self.optimizer.zero_grad()
		# 负采样model
		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode']
		}) # size is 1
		loss.backward()
		self.optimizer.step()
		return loss.item()

	#整个训练过程的 配置，以及调用 一个训练batcch方法的框架
	def run(self):
		if self.use_gpu:
			self.model.cuda()
		# 配置优化器
		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay, # L2正则化
			)
		print("Finish initializing...")

		# 显示进度条
		training_range = tqdm(range(self.train_times)) #进度条
		# 在训练模型
		for epoch in training_range: # 迭代 train_times 次 epoch
			res = 0.0
			for data in self.data_loader: # 迭代 nbatches 次 batch
				# data 是一个字典，
				# 			"batch_h": self.batch_h,  size [batches_size] np
				# 			"batch_t": self.batch_t,
				# 			"batch_r": self.batch_r,
				# 			"batch_y": self.batch_y,
				# 			"mode": "normal"
				loss = self.train_one_step(data) # size is 1
				res += loss
			training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
			
			if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

	def set_model(self, model):
		self.model = model

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	def set_train_times(self, train_times):
		self.train_times = train_times

	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir