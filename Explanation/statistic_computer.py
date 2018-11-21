import numpy as np
import json
import random
import gzip
import math
import sys, os


def compute_mean_std(count_list):
	return np.mean(count_list), np.std(count_list)

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
		self.query_words = []
		self.query_max_length = 0
		with gzip.open(input_train_dir + 'query.txt.gz', 'r') as fin:
			for line in fin:
				words = [int(i) for i in line.strip().split(' ')]
				if len(words) > self.query_max_length:
					self.query_max_length = len(words)
				self.query_words.append(words)
		#pad
		for i in xrange(len(self.query_words)):
			self.query_words[i] = [-1 for j in xrange(self.query_max_length-len(self.query_words[i]))] + self.query_words[i]


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

		#get product query sets
		self.product_query_idx = []
		with gzip.open(input_train_dir + set_name + '_query_idx.txt.gz', 'r') as fin:
			for line in fin:
				arr = line.strip().split(' ')
				query_idx = []
				for idx in arr:
					if len(idx) < 1:
						continue
					query_idx.append(int(idx))
				self.product_query_idx.append(query_idx)

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

	def print_statistics(self):
		# Number of brands
		print('Number of Brands %d' % len(self.brand_ids))
		# Number of categories
		print('Number of Categories %d' % len(self.category_ids))
		# Words per user
		u_words_count = np.zeros(len(self.user_ids))
		# Words per item
		p_words_count = np.zeros(len(self.product_ids))
		for i in xrange(len(self.review_info)):
			uid, pid = self.review_info[i]
			length = len(self.review_text[i])
			u_words_count[uid] += length
			p_words_count[pid] += length 
		print('Words per users %.2f$\\pm$%.2f' % compute_mean_std(u_words_count))
		print('Words per items %.2f$\\pm$%.2f' % compute_mean_std(p_words_count))

		# Also brought per item
		# Also viewed per item
		# Brought together per item
		# brand per item
		# category per item
		for name in self.knowledge:
			count_list = [ len(l) for l in self.knowledge[name]['data']]
			print(name + ' per items %.2f$\\pm$%.2f' % compute_mean_std(count_list))


def main(argv):
    data_path = argv[0]
    data = Tensorflow_data(data_path, data_path + 'query_split/', 'train')
    data.print_statistics()

if __name__ == "__main__":
    main(sys.argv[1:])



