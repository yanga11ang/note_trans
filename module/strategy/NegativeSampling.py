from .Strategy import Strategy

# 优化目标函数， 负采样
# 父类 是nn.Module 因为用到了forward 的特殊用法
class NegativeSampling(Strategy):

	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
		super(NegativeSampling, self).__init__()
		self.model = model  # 正常的模型 trans系列模型
		self.loss = loss	# 损失函数 nn.Module类 ，有forward 函数
		self.batch_size = batch_size
		self.regul_rate = regul_rate	# 正则系数
		self.l3_regul_rate = l3_regul_rate	#l3正则系数

	# 前 batch_size 是正例
	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0) # 排列维度， size is (batch_size,1)
		return positive_score

	# batch_size之后是负例
	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0) # size is (batch_size, neg_rate)
		return negative_score

	def forward(self, data):
		score = self.model(data) # 调用 trans系列模型，算出 每个三元组的score。size is (batches_size)
		p_score = self._get_positive_score(score) # size is (batch_size,1)
		n_score = self._get_negative_score(score) # size is (batch_size, neg_rate)
		loss_res = self.loss(p_score, n_score) # loss 是nn.Module类 size is (1)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res