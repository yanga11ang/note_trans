import torch
import torch.nn as nn
from ..BaseModule import BaseModule

# 用于 实现 模型的父类，为了 规整，
class Model(BaseModule):

	def __init__(self, ent_tot, rel_tot):
		super(Model, self).__init__()
		self.ent_tot = ent_tot
		self.rel_tot = rel_tot

	def forward(self):
		raise NotImplementedError
	
	def predict(self):
		raise NotImplementedError