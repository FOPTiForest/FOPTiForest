import numpy as np
import copy as cp
from tqdm import trange
from . import opt_tree as Ot
from joblib import Parallel, delayed
import copy as cp
import time

class OptForest:
	def __init__(self, num_trees, sampler, lsh_family, threshold, branch, distance, granularity=1):
		self._num_trees = num_trees
		self._sampler = sampler
		self._lsh_family = lsh_family
		self._granularity = granularity
		self._trees = []
		self.threshold = threshold
		self.branch = branch
		self.distance = distance

	
	def display(self):
		for t in self._trees:
			t.display()


	def fit(self, data):
		self.build(data)


	def build(self, data):
		
		
		indices = range(len(data))
		# Uncomment the following code for continuous values
		data = np.c_[indices, data]

		# Important: clean the tree array
		self._trees = []


		start_time = time.time()
		# Sampling data
		self._sampler.fit(data)
		sampled_datas = self._sampler.draw_samples(data)


		end_time = time.time()
		execution_time = end_time - start_time

		print("内鬼的执行时间：", execution_time, "秒")

		
		start_time = time.time()

		# Build LSH instances based on the given data
		#org
		# lsh_instances = []
		# for i in range(self._num_trees):
		# 	transformed_data = data
		# 	if self._sampler._bagging != None:
		# 		transformed_data = self._sampler._bagging_instances[i].get_transformed_data(data)	
		# 	self._lsh_family.fit(transformed_data)
		# 	lsh_instances.append(cp.deepcopy(self._lsh_family))

		#par
		def fit_lsh(i, data, sampler, lsh_family):
			transformed_data = data
			if sampler._bagging is not None:
				transformed_data = sampler._bagging_instances[i].get_transformed_data(data)    
			lsh_family.fit(transformed_data)
			return cp.deepcopy(lsh_family)

		lsh_instances = Parallel(n_jobs=10, batch_size=10)(delayed(fit_lsh)(i, data, self._sampler, self._lsh_family) for i in range(self._num_trees))



		end_time = time.time()
		execution_time = end_time - start_time

		print("build instance data执行时间：", execution_time, "秒")

		start_time = time.time()

		# Build LSH trees
		# org
		# print("@@@@@@@",self._num_trees)
		for i in trange(self._num_trees):
			sampled_data = sampled_datas[i]
			tree = Ot.HierTree(lsh_instances[i], self.threshold, self.branch, self.distance)
			tree.build(sampled_data)
			self._trees.append(tree)

		#par
		def build_tree(i, sampled_datas, lsh_instances, threshold, branch, distance):
			sampled_data = sampled_datas[i]
			tree = Ot.HierTree(lsh_instances[i], threshold, branch, distance)
			tree.build(sampled_data)
			return tree

		self._trees = Parallel(n_jobs=10, batch_size=10)(delayed(build_tree)(i, sampled_datas, lsh_instances, self.threshold, self.branch, self.distance) for i in trange(self._num_trees))


		end_time = time.time()
		execution_time = end_time - start_time

		print("LSHtree执行时间：", execution_time, "秒")
	
	
	
	def decision_function(self, data):
		indices = range(len(data))
		# Uncomment the following code for continuous data
		data = np.c_[indices, data]

		depths=[]
		data_size = len(data)
		# org
		# for i in trange(data_size):
		# 	d_depths = []
		# 	for j in range(self._num_trees):
		# 		transformed_data = data[i]
		# 		if self._sampler._bagging != None:
		# 			transformed_data = self._sampler._bagging_instances[j].get_transformed_data(np.mat(data[i])).A1
		# 		d_depths.append(self._trees[j].predict(self._granularity, transformed_data))
		# 	depths.append(d_depths)

		#par
		def process_batch(batch, data, sampler, trees, granularity):
			batch_depths = []
			for i in batch:
				d_depths = []
				for j in range(len(trees)):
					transformed_data = data[i]
					if sampler._bagging != None:
						transformed_data = sampler._bagging_instances[j].get_transformed_data(np.mat(data[i])).A1
					d_depths.append(trees[j].predict(granularity, transformed_data))
				batch_depths.append(d_depths)
			return batch_depths

		# Split data indices into batches
		batch_size = max(data_size/10+2,3000) # Adjust this value to your needs
		batches = [range(i, min(i + batch_size, data_size)) for i in range(0, data_size, batch_size)]

		# Process each batch in parallel
		depths = []
		for batch_depths in Parallel(n_jobs=10)(delayed(process_batch)(batch, data, self._sampler, self._trees, self._granularity) for batch in batches):
			depths.extend(batch_depths)


		# Arithmatic mean	
		avg_depths=[]
		for i in range(data_size):
			depth_avg = 0.0
			for j in range(self._num_trees):
				depth_avg += depths[i][j]
			depth_avg /= self._num_trees
			avg_depths.append(depth_avg)

		avg_depths = np.array(avg_depths)
		return -1.0*avg_depths


	def get_avg_branch_factor(self):
		sum = 0.0
		for t in self._trees:
			sum += t.get_avg_branch_factor()
		return sum/self._num_trees		
