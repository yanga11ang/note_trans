# coding:utf-8
import os
import ctypes  #python 和 C 混合调用的库
import numpy as np
import platform

# 每个epoch 是一个 TrainDataSampler ，nbatches次调用 datasampler，每次调用产生一个batch的数据
class TrainDataSampler(object):
	# 第二个参数返回 调用 产生一个batches 的函数
	def __init__(self, nbatches, datasampler):
		self.nbatches = nbatches
		self.datasampler = datasampler
		self.batch = 0

	def __iter__(self):
		return self

	def __next__(self):
		self.batch += 1 
		if self.batch > self.nbatches:
			raise StopIteration()
		return self.datasampler()

	def __len__(self):
		return self.nbatches

#用于加载 训练数据，低层实现是调用C++的DLL，上层主要实现，batch数据迭代
class TrainDataLoader(object):

	def __init__(self, in_path = "./", batch_size = None, nbatches = None, threads = 8, sampling_mode = "normal", bern_flag = 0, filter_flag = 1, neg_ent = 1, neg_rel = 0):
		#连接 动态链接库
		if platform.system() == 'Windows':
			base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.dll"))
			print(base_file)
			self.lib = ctypes.cdll.LoadLibrary(base_file)
		elif platform.system() == 'Linux':
			base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
			self.lib = ctypes.cdll.LoadLibrary(base_file)


		"""argtypes"""
		self.lib.sampling.argtypes = [
			ctypes.c_void_p, # 定义指针
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64
		]
		"""set essential parameters"""
		self.in_path = in_path   # 训练数据存放位置
		self.work_threads = threads # C代码所用线程数目
		self.nbatches = nbatches # 每个epoch 有多少个batches ，因为 batches 是随机抽取，而不是枚举
		self.batch_size = batch_size # 每个batch的样例数
		self.bern = bern_flag # 是否用负例采样优化
		self.filter = filter_flag # 是否过滤 假负例
		self.negative_ent = neg_ent # 负例比率 （替换h 或t）
		self.negative_rel = neg_rel # 负例比率 （替换 r）
		self.sampling_mode = sampling_mode # 采样方式 正态
		self.cross_sampling_flag = 0 # 随机 替换 h t ，而不是所以都替换h 或者 t
		self.read()

	def read(self):
		self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2)) #设置路径
		self.lib.setBern(self.bern) # 设置是否采用更优的 负样采集方法
		self.lib.setWorkThreads(self.work_threads) # 设定 线程数目
		self.lib.randReset() # 设置随机种子
		self.lib.importTrainFiles() # 加载测试集
		self.relTotal = self.lib.getRelationTotal() # 获取 关系数
		self.entTotal = self.lib.getEntityTotal() # 获取 实体书
		self.tripleTotal = self.lib.getTrainTotal() # 获取总样本数

		if self.batch_size == None:
			self.batch_size = self.tripleTotal // self.nbatches # 整数除法
		if self.nbatches == None:
			self.nbatches = self.tripleTotal // self.batch_size
		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel) #batch 缓冲区大小

		self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
		self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
		self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
		self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
		self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

	def sampling(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			0,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h, 
			"batch_t": self.batch_t, 
			"batch_r": self.batch_r, 
			"batch_y": self.batch_y,
			"mode": "normal"
		}

	def sampling_head(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			-1,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h,
			"batch_t": self.batch_t[:self.batch_size],
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "head_batch"
		}

	def sampling_tail(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			1,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h[:self.batch_size],
			"batch_t": self.batch_t,
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "tail_batch"
		}

	def cross_sampling(self):
		self.cross_sampling_flag = 1 - self.cross_sampling_flag 
		# self.cross_sampling_flag = 0 #haha
		if self.cross_sampling_flag == 0:
			return self.sampling_head()
		else:
			return self.sampling_tail()

	"""interfaces to set essential parameters"""

	def set_work_threads(self, work_threads):
		self.work_threads = work_threads

	def set_in_path(self, in_path):
		self.in_path = in_path

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
		self.nbatches = self.tripleTotal // self.batch_size

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_bern_flag(self, bern):
		self.bern = bern

	def set_filter_flag(self, filter):
		self.filter = filter

	"""interfaces to get essential parameters"""

	def get_batch_size(self):
		return self.batch_size

	def get_ent_tot(self):
		return self.entTotal

	def get_rel_tot(self):
		return self.relTotal

	def get_triple_tot(self):
		return self.tripleTotal

	def __iter__(self):
		if self.sampling_mode == "normal":
			return TrainDataSampler(self.nbatches, self.sampling)
		else:
			return TrainDataSampler(self.nbatches, self.cross_sampling)

	def __len__(self):
		return self.nbatches