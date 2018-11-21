import numpy as np
import json
import random
import gzip
import math
import struct
import pandas as pd
from array import array

class Tensorflow_data:
	def __init__(self, data_path, input_train_dir, set_name):
		#get product/user/vocabulary information
		self.product_ids = []
		with gzip.open(data_path + 'product.txt.gz', 'r') as fin:
			for line in fin:
				self.product_ids.append(line.strip())
		self.product_size = len(self.product_ids)
		self.user_ids = []
		with gzip.open(data_path + 'users.txt.gz', 'r') as fin:
			for line in fin:
				self.user_ids.append(line.strip())
		self.user_size = len(self.user_ids)
		self.words = []
		with gzip.open(data_path + 'vocab.txt.gz', 'r') as fin:
			for line in fin:
				self.words.append(line.strip())
		self.vocab_size = len(self.words)

		#get review sets
		self.word_count = 0
		self.vocab_distribute = np.zeros(self.vocab_size) 
		self.review_info = []
		self.review_text = []
		with gzip.open(input_train_dir + set_name + '.txt.gz', 'r') as fin:
			for line in fin:
				arr = line.strip().split('\t')
				self.review_info.append((int(arr[0]), int(arr[1]))) # (user_idx, product_idx)
				self.review_text.append([int(i) for i in arr[2].split(' ')])
				for idx in self.review_text[-1]:
					self.vocab_distribute[idx] += 1
				self.word_count += len(self.review_text[-1])
		self.review_size = len(self.review_info)
		self.vocab_distribute = self.vocab_distribute.tolist() 
		self.sub_sampling_rate = None
		self.review_distribute = np.ones(self.review_size).tolist()
		self.product_distribute = np.ones(self.product_size).tolist()

		# get knowledge
		self.related_product_ids = []
		with gzip.open(data_path + 'related_product.txt.gz', 'r') as fin:
			for line in fin:
				self.related_product_ids.append(line.strip())
		self.related_product_size = len(self.related_product_ids)
		self.brand_ids = []
		with gzip.open(data_path + 'brand.txt.gz', 'r') as fin:
			for line in fin:
				self.brand_ids.append(line.strip())
		self.brand_size = len(self.brand_ids)
		self.category_ids = []
		with gzip.open(data_path + 'category.txt.gz', 'r') as fin:
			for line in fin:
				self.category_ids.append(line.strip())
		self.category_size = len(self.category_ids)
		
		self.entity_vocab = {
			'user' : self.user_ids,
			'word' : self.words,
			'product' : self.product_ids,
			'related_product' : self.related_product_ids,
			'brand' : self.brand_ids,
			'categories' : self.category_ids
		}
		knowledge_file_dict = {
			'also_bought' : data_path + 'also_bought_p_p.txt.gz',
			'also_viewed' : data_path + 'also_viewed_p_p.txt.gz',
			'bought_together' : data_path + 'bought_together_p_p.txt.gz',
			'brand' : data_path + 'brand_p_b.txt.gz',
			'categories' : data_path + 'category_p_c.txt.gz'
		}
		knowledge_vocab = {
			'also_bought' : self.related_product_ids,
			'also_viewed' : self.related_product_ids,
			'bought_together' : self.related_product_ids,
			'brand' : self.brand_ids,
			'categories' : self.category_ids
		}
		self.knowledge = {}
		for name in knowledge_file_dict:
			self.knowledge[name] = {}
			self.knowledge[name]['data'] = []
			self.knowledge[name]['vocab'] = knowledge_vocab[name]
			self.knowledge[name]['distribute'] = np.zeros(len(self.knowledge[name]['vocab']))
			with gzip.open(knowledge_file_dict[name], 'r') as fin:
				for line in fin:
					knowledge = []
					arr = line.strip().split(' ')
					for x in arr:
						if len(x) > 0:
							x = int(x)
							knowledge.append(x)
							self.knowledge[name]['distribute'][x] += 1
					self.knowledge[name]['data'].append(knowledge)
			self.knowledge[name]['distribute'] = self.knowledge[name]['distribute'].tolist()

		print("Data statistic: vocab %d, review %d, user %d, product %d\n" % (self.vocab_size, 
					self.review_size, self.user_size, self.product_size))

	def sub_sampling(self, subsample_threshold):
		self.sub_sampling_rate = np.ones(self.vocab_size)
		if subsample_threshold == 0.0:
			return
		threshold = sum(self.vocab_distribute) * subsample_threshold
		count_sub_sample = 0
		for i in xrange(self.vocab_size):
			#vocab_distribute[i] could be zero if the word does not appear in the training set
			if self.vocab_distribute[i] == 0:
				continue	
			self.sub_sampling_rate[i] = min((np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]),
											1.0)
			count_sub_sample += 1

	def read_image_features(self, data_path):
		self.img_feature_num = 4096
		self.img_features = [None for i in xrange(self.product_size)]
		with open(data_path + 'product_image_feature.b', 'rb') as fin:
			for i in xrange(self.product_size):
				float_array = array('f')
				float_array.fromfile(fin, 4096)
				self.img_features[i] = list(float_array)

	def read_latent_factor(self, data_path):
		user_latent_factor_file_name = data_path + 'user_factors.csv'
		item_latent_factor_file_name = data_path + 'item_factors.csv'
		user = pd.read_csv(user_latent_factor_file_name).iloc[:, 1:]
		item = pd.read_csv(item_latent_factor_file_name).iloc[:, 1:]
		self.user_factors = user.values[:self.user_size]
		self.product_factors = item.values[:self.product_size]
		self.rate_factor_num = len(self.user_factors[0])
		print('Rate factor size ' + str(self.rate_factor_num))
		#return user.values, item.values

	def read_train_product_ids(self, data_path):
		self.user_train_product_set_list = [set() for i in xrange(self.user_size)]
		self.train_review_size = 0
		with gzip.open(data_path + 'train.txt.gz', 'r') as fin:
			for line in fin:
				self.train_review_size += 1
				arr = line.strip().split('\t')
				self.user_train_product_set_list[int(arr[0])].add(int(arr[1]))


	def compute_test_product_ranklist(self, u_idx, original_scores, sorted_product_idxs, rank_cutoff):
		product_rank_list = []
		product_rank_scores = []
		rank = 0
		for product_idx in sorted_product_idxs:
			if product_idx in self.user_train_product_set_list[u_idx] or math.isnan(original_scores[product_idx]):
				continue
			product_rank_list.append(product_idx)
			product_rank_scores.append(original_scores[product_idx])
			rank += 1
			if rank == rank_cutoff:
				break
		return product_rank_list, product_rank_scores


	def output_ranklist(self, user_ranklist_map, user_ranklist_score_map, output_path, similarity_func):
		with open(output_path + 'test.'+similarity_func+'.ranklist', 'w') as rank_fout:
			for u_idx in user_ranklist_map:
				user_id = self.user_ids[u_idx]
				for i in xrange(len(user_ranklist_map[u_idx])):
					product_id = self.product_ids[user_ranklist_map[u_idx][i]]
					rank_fout.write(user_id + ' Q0 ' + product_id + ' ' + str(i+1)
							+ ' ' + str(user_ranklist_score_map[u_idx][i]) + ' MultiViewEmbedding\n')

	def output_embedding(self, embeddings, output_file_name):
		with open(output_file_name,'w') as emb_fout:
			try:
				length = len(embeddings)
				if length < 1:
					return
				dimensions = len(embeddings[0])
				emb_fout.write(str(length) + '\n')
				emb_fout.write(str(dimensions) + '\n')
				for i in xrange(length):
					for j in xrange(dimensions):
						emb_fout.write(str(embeddings[i][j]) + ' ')
					emb_fout.write('\n')
			except:
				if isinstance(embeddings.tolist(), float):
					emb_fout.write(str(embeddings) + '\n')
				else:
					emb_fout.write(' '.join([str(x) for x in embeddings.tolist()]) + '\n')
				#else:
				#	emb_fout.write(str(embeddings) + '\n')

	def print_entity_list(self, relation_name, entity_name, entity_scores, rank_cut, remove_map):
		if entity_name not in self.entity_vocab:
			print('Cannot find entity %s' % entity_name)
		print('%s list: rank, id, name, score' % relation_name)
		sorted_entity_idxs = sorted(range(len(entity_scores)), 
									key=lambda k: entity_scores[k], reverse=True)
		entity_rank_list = []
		entity_rank_scores = []
		rank = 0
		for entity_idx in sorted_entity_idxs:
			if entity_name in remove_map and entity_idx in remove_map[entity_name]:
				continue
			if math.isnan(entity_scores[entity_idx]):
				continue
			entity_rank_list.append(entity_idx)
			entity_rank_scores.append(entity_scores[entity_idx])
			rank += 1
			if rank >= rank_cut:
				break
		# print results
		for i in xrange(len(entity_rank_list)):
			print('%d\t%d\t"%s"\t%.4f' % (i, entity_rank_list[i], 
				self.entity_vocab[entity_name][entity_rank_list[i]], entity_rank_scores[i]))
	
	def get_idx(self, input_str, entity_name):
		if entity_name not in self.entity_vocab:
			print('Cannot find entity %s' % entity_name)
			return None
		try:
			return self.entity_vocab[entity_name].index(input_str)
		except:
			return int(input_str)



