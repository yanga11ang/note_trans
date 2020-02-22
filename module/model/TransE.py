import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

# 父类 是nn.Module 因为用到了forward 的特殊用法
class TransE(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransE, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim   # embedding dim
		self.margin = margin #
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		# 初始化参数
		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]),
				requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

	#打分 求2范数的平方
	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			# size is (batches_size, embedding_size)
			h = F.normalize(h, 2, -1)  # Performs L2 normalization of inputs over -1 dimension.
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1]) # size is (1, batches_size, embedding_size)
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])

		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t # (batches_size, embedding_size)

		score = torch.norm(score, self.p_norm, -1) # 在最后一维上 求1范数的平方 size is (batches_size)
		score.flatten() # size is (batches_size)
		return score

	def forward(self, data):
		batch_h = data['batch_h'] # size is (batches_size)
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		# embedding rel and entity
		h = self.ent_embeddings(batch_h) # size is (batches_size, embedding_size)
		t = self.ent_embeddings(batch_t) # size is (batches_size, embedding_size)
		r = self.rel_embeddings(batch_r) # size is (batches_size, embedding_size)
		score = self._calc(h ,t, r, mode) # 得到 score
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()