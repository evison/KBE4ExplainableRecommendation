from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange# pylint: disable=redefined-builtin
from six.moves import zip	 # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

from tensorflow.python.client import timeline

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange# pylint: disable=redefined-builtin
import tensorflow as tf

class MultiViewEmbedding_model(object):
	def __init__(self, data_set, window_size,
				 embed_size, max_gradient_norm, batch_size, learning_rate, L2_lambda, image_weight,
				 net_struct, similarity_func, forward_only=False, negative_sample = 5):
		"""Create the model.
	
		Args:
			vocab_size: the number of words in the corpus.
			dm_feature_len: the length of document model features.
			review_size: the number of reviews in the corpus.
			user_size: the number of users in the corpus.
			product_size: the number of products in the corpus.
			embed_size: the size of each embedding
			window_size: the size of half context window
			vocab_distribute: the distribution for words, used for negative sampling
			review_distribute: the distribution for reviews, used for negative sampling
			product_distribute: the distribution for products, used for negative sampling
			max_gradient_norm: gradients will be clipped to maximally this norm.
			batch_size: the size of the batches used during training;
			the model construction is not independent of batch_size, so it cannot be
			changed after initialization.
			learning_rate: learning rate to start with.
			learning_rate_decay_factor: decay learning rate by this much when needed.
			forward_only: if set, we do not construct the backward pass in the model.
			negative_sample: the number of negative_samples for training
		"""
		self.data_set = data_set
		self.negative_sample = negative_sample
		self.embed_size = embed_size
		self.window_size = window_size
		self.max_gradient_norm = max_gradient_norm
		self.batch_size = batch_size * (self.negative_sample + 1)
		self.init_learning_rate = learning_rate
		self.L2_lambda = L2_lambda
		self.image_weight = image_weight
		self.net_struct = net_struct
		self.similarity_func = similarity_func
		self.global_step = tf.Variable(0, trainable=False)

		self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
		init_width = 0.5 / self.embed_size
		#self.img_feature_num = data_set.img_feature_num
		#self.rate_factor_num = data_set.rate_factor_num
		self.need_image = True if self.image_weight > 0 else False

		def entity(name, vocab):
			#print('%s size %s' % (name,str(len(vocab))))
			return {
				'name' : name,
				'vocab' : vocab,
				'size' : len(vocab),
				'embedding' :tf.Variable( tf.random_uniform(									
							[len(vocab), self.embed_size], -init_width, init_width),				
							name="%s_emb"%name)	
			}
		self.entity_dict = {
			'user' : entity('user', data_set.user_ids),
			'product' : entity('product', data_set.product_ids),
			'word' : entity('word', data_set.words),
			'related_product' : entity('related_product',data_set.related_product_ids),
			'brand' : entity('brand', data_set.brand_ids),
			'categories' : entity('categories', data_set.category_ids),
		}

		self.user_idxs = tf.placeholder(tf.int64, shape=[None], name="user_idxs")
		
		def relation(name, distribute):
			#print('%s size %s' % (name, str(len(distribute))))
			return {
				'distribute' : distribute,
				'idxs' : tf.placeholder(tf.int64, shape=[None], name="%s_idxs"%name),
				'weight' : 	tf.placeholder(tf.float32, shape=[None], name="%s_weight"%name),
				'embedding' : tf.Variable( tf.random_uniform(									
								[self.embed_size], -init_width, init_width),				
								name="%s_emb"%name),
				'bias' : tf.Variable(tf.zeros([len(distribute)]), name="%s_b"%name)	
			}
		self.relation_dict = {
			'product' : relation('purchase', data_set.product_distribute),
			'word' : relation('write', data_set.vocab_distribute),
			#'image' : relation('has_image', data_set.product_distribute),
			'also_bought' : relation('also_bought', data_set.knowledge['also_bought']['distribute']),
			'also_viewed' : relation('also_viewed', data_set.knowledge['also_viewed']['distribute']),
			'bought_together' : relation('bought_together', data_set.knowledge['bought_together']['distribute']),
			'brand' : relation('is_brand', data_set.knowledge['brand']['distribute']),
			'categories' : relation('is_category', data_set.knowledge['categories']['distribute'])
		}

		# Select which relation to use
		self.use_relation_dict = {
			'also_bought' : False,
			'also_viewed' : False,
			'bought_together' : False,
			'brand' : False,
			'categories' : False,
		}
		if 'none' in self.net_struct:
			print('Use no relation')
		else:
			need_relation_list = []
			for key in self.use_relation_dict:
				if key in self.net_struct:
					self.use_relation_dict[key] = True
					need_relation_list.append(key)
			if len(need_relation_list) > 0:
				print('Use relation ' + ' '.join(need_relation_list))
			else:
				print('Use all relation')
				for key in self.use_relation_dict:
					self.use_relation_dict[key] = True


		print('L2 lambda ' + str(self.L2_lambda))

		# Training losses.
		self.loss = self.build_embedding_graph_and_loss()

		# Gradients and SGD update operation for training the model.
		params = tf.trainable_variables()
		if not forward_only:
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			self.gradients = tf.gradients(self.loss, params)
			
			self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
																	 self.max_gradient_norm)
			self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
											 global_step=self.global_step)
			
			#self.updates = opt.apply_gradients(zip(self.gradients, params),
			#								 global_step=self.global_step)
		else:
			#self.product_scores = self.get_product_scores(self.user_idxs)

			# user + purchase -> word
			user_vec = tf.nn.embedding_lookup(self.entity_dict['user']['embedding'], self.user_idxs)
			self.product_scores, up_vec = self.get_relation_scores(0.5, user_vec, 'product', 'product')
			# user + write -> word
			self.uw_scores, uw_vec = self.get_relation_scores(0.5, user_vec, 'word', 'word')
			# user + purchase + write -> word
			self.upw_scores, upw_vec = self.get_relation_scores(0.5, up_vec, 'word', 'word')
			# product + write -> word
			p_vec = tf.nn.embedding_lookup(self.entity_dict['product']['embedding'], self.relation_dict['product']['idxs'])
			self.pw_scores, pw_vec = self.get_relation_scores(0.5, p_vec, 'word', 'word')
			# Compute all information based on user
			self.up_entity_list = [
				('purchase', 'product', self.product_scores), ('write', 'word', self.uw_scores),
			]
			# Compute all information based on product
			self.p_entity_list = [('write', 'word', self.pw_scores)]

			if self.use_relation_dict['also_bought']:
				# user + purchase + also_bought -> product
				self.upab_scores, upab_vec = self.get_relation_scores(0.5, up_vec, 'also_bought', 'related_product')
				self.up_entity_list.append(('also_bought', 'related_product', self.upab_scores))
				# product + also_bought -> product
				self.pab_scores, pab_vec = self.get_relation_scores(0.5, p_vec, 'also_bought', 'related_product')
				self.p_entity_list.append(('also_bought', 'related_product', self.pab_scores))
			
			if self.use_relation_dict['also_viewed']:
				# user + purchase + also_viewed -> product
				self.upav_scores, upav_vec = self.get_relation_scores(0.5, up_vec, 'also_viewed', 'related_product')
				self.up_entity_list.append(('also_viewed', 'related_product', self.upav_scores))
				# product + also_viewed -> product
				self.pav_scores, pav_vec = self.get_relation_scores(0.5, p_vec, 'also_viewed', 'related_product')
				self.p_entity_list.append(('also_viewed', 'related_product', self.pav_scores))
			
			if self.use_relation_dict['bought_together']:
				# user + purchase + bought_together -> product
				self.upbt_scores, upbt_vec = self.get_relation_scores(0.5, up_vec, 'bought_together', 'related_product')
				self.up_entity_list.append(('bought_together', 'related_product', self.upbt_scores))
				# product + bought_together -> product
				self.pbt_scores, pbt_vec = self.get_relation_scores(0.5, p_vec, 'bought_together', 'related_product')
				self.p_entity_list.append(('bought_together', 'related_product', self.pbt_scores))
			
			if self.use_relation_dict['brand']:
				# user + purchase + is_brand -> brand
				self.upib_scores, upib_vec = self.get_relation_scores(0.5, up_vec, 'brand', 'brand')
				self.up_entity_list.append(('brand', 'brand', self.upib_scores))
				# product + is_brand -> brand
				self.pib_scores, pib_vec = self.get_relation_scores(0.5, p_vec, 'brand', 'brand')
				self.p_entity_list.append(('brand', 'brand', self.pib_scores))
			
			if self.use_relation_dict['categories']:
				# user + purchase + is_category -> category
				self.upic_scores, upic_vec = self.get_relation_scores(0.5, up_vec, 'categories', 'categories')
				self.up_entity_list.append(('categories', 'categories', self.upic_scores))
				# product + is_category -> category
				self.pic_scores, pic_vec = self.get_relation_scores(0.5, p_vec, 'categories', 'categories')
				self.p_entity_list.append(('categories', 'categories', self.pic_scores))
			
	
		self.saver = tf.train.Saver(tf.global_variables())

	def get_relation_scores(self, add_weight, head_vec, relation_name, tail_name, tail_idxs = None, scope = None):
		with variable_scope.variable_scope(scope or "embedding_graph"):										
			# get relation embedding [None, embed_size]
			relation_vec = self.relation_dict[relation_name]['embedding']
			relation_bias = self.relation_dict[relation_name]['bias']

			# get candidate product embedding [None, embed_size]										
			tail_vec = None
			tail_bias = None
			if tail_idxs != None:										
				tail_vec = tf.nn.embedding_lookup(self.entity_dict[tail_name]['embedding'], tail_idxs)									
				tail_bias = tf.nn.embedding_lookup(relation_bias, tail_idxs)
			else:										
				tail_vec = self.entity_dict[tail_name]['embedding']
				tail_bias = relation_bias								
									
			#example_vec = (1.0 - add_weight) * head_vec + add_weight * relation_vec
			example_vec = head_vec + relation_vec
			
			if self.similarity_func == 'product':										
				return tf.matmul(example_vec, tail_vec, transpose_b=True), example_vec
			elif self.similarity_func == 'bias_product':
				return tf.matmul(example_vec, tail_vec, transpose_b=True) + tail_bias, example_vec
			else:										
				norm_vec = example_vec / tf.sqrt(tf.reduce_sum(tf.square(example_vec), 1, keep_dims=True))
				tail_vec = tail_vec / tf.sqrt(tf.reduce_sum(tf.square(tail_vec), 1, keep_dims=True))									
				return tf.matmul(norm_vec, tail_vec, transpose_b=True), example_vec
				

	def build_embedding_graph_and_loss(self, scope = None):
		with variable_scope.variable_scope(scope or "embedding_graph"):
			loss = None
			regularization_terms = []
			batch_size = array_ops.shape(self.user_idxs)[0]#get batch_size	

			# user + purcahse -> product
			up_loss, up_embs = relation_nce_loss(self, 0.5, self.user_idxs, 'user', 'product', 'product')
			regularization_terms.extend(up_embs)
			#up_loss = tf.Print(up_loss, [up_loss], 'this is up', summarize=5)
			loss = up_loss

			# user + write -> word
			uw_loss, uw_embs = relation_nce_loss(self, 0.5, self.user_idxs, 'user', 'word', 'word')
			regularization_terms.extend(uw_embs)
			#uw_loss = tf.Print(uw_loss, [uw_loss], 'this is uw', summarize=5)
			loss += uw_loss

			# product + write -> word
			pw_loss, pw_embs = relation_nce_loss(self, 0.5, self.relation_dict['product']['idxs'], 'product', 'word', 'word')
			regularization_terms.extend(pw_embs)
			#pw_loss = tf.Print(pw_loss, [pw_loss], 'this is pw', summarize=5)
			loss += pw_loss

			# product + also_bought -> product
			if self.use_relation_dict['also_bought']:
				pab_loss, pab_embs = relation_nce_loss(self, 0.5, self.relation_dict['product']['idxs'], 'product', 'also_bought', 'related_product')
				regularization_terms.extend(pab_embs)
				#pab_loss = tf.Print(pab_loss, [pab_loss], 'this is pab', summarize=5)
				loss += pab_loss

			# product + also_viewed -> product
			if self.use_relation_dict['also_viewed']:
				pav_loss, pav_embs = relation_nce_loss(self, 0.5, self.relation_dict['product']['idxs'], 'product', 'also_viewed', 'related_product')
				regularization_terms.extend(pav_embs)
				#pav_loss = tf.Print(pav_loss, [pav_loss], 'this is pav', summarize=5)
				loss += pav_loss

			# product + bought_together -> product
			if self.use_relation_dict['bought_together']:
				pbt_loss, pbt_embs = relation_nce_loss(self, 0.5, self.relation_dict['product']['idxs'], 'product', 'bought_together', 'related_product')
				regularization_terms.extend(pbt_embs)
				#pbt_loss = tf.Print(pbt_loss, [pbt_loss], 'this is pbt', summarize=5)
				loss += pbt_loss

			# product + is_brand -> brand
			if self.use_relation_dict['brand']:
				pib_loss, pib_embs = relation_nce_loss(self, 0.5, self.relation_dict['product']['idxs'], 'product', 'brand', 'brand')
				regularization_terms.extend(pib_embs)
				#pib_loss = tf.Print(pib_loss, [pib_loss], 'this is pib', summarize=5)
				loss += pib_loss

			# product + is_category -> categories
			if self.use_relation_dict['categories']:
				pic_loss, pic_embs = relation_nce_loss(self, 0.5, self.relation_dict['product']['idxs'], 'product', 'categories', 'categories')
				regularization_terms.extend(pic_embs)
				#pic_loss = tf.Print(pic_loss, [pic_loss], 'this is pic', summarize=5)
				loss += pic_loss

			# product + is_image -> images
			'''
			if self.need_image:
				self.img_product_features =	tf.constant(self.data_set.img_features, shape=[self.entity_dict['product']['size'], self.img_feature_num],
										name="img_product_features")
				pii_loss, pii_embs = self.image_nce_loss(0.5, self.relation_dict['product']['idxs'], 'product')
				regularization_terms.extend(pii_embs)
				#pii_loss = tf.Print(pii_loss, [pii_loss], 'this is pii', summarize=5)
				loss += pii_loss
			'''

			# L2 regularization
			if self.L2_lambda > 0:
				l2_loss = tf.nn.l2_loss(regularization_terms[0])
				for i in xrange(1,len(regularization_terms)):
					l2_loss += tf.nn.l2_loss(regularization_terms[i])
				loss += self.L2_lambda * l2_loss
			
			return loss / math_ops.cast(batch_size, dtypes.float32)

	#get product embeddings
	def decode_img(self, input_data, reuse, scope=None):
		#reuse = None if index < 1 else True
		with variable_scope.variable_scope(scope or 'img_decode',
										 reuse=reuse):
			output_data = input_data
			output_sizes = [int((self.img_feature_num + self.embed_size)/2), self.embed_size]
			#output_sizes = [self.embed_size]
			current_size = output_data.get_shape().as_list()[1]
			for i in xrange(len(output_sizes)):
				expand_W = variable_scope.get_variable("expand_W_%d" % i, [current_size, output_sizes[i]])
				expand_b = variable_scope.get_variable("expand_b_%d" % i, [output_sizes[i]])
				output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
				output_data = tf.nn.elu(output_data)
				current_size = output_sizes[i]
				#print(expand_W.name)
			return output_data

	def image_nce_loss(self, add_weight, example_idxs, head_entity_name):  #, neg_product_idxs, product_img_features, neg_img_features):
		batch_size = array_ops.shape(example_idxs)[0]#get batch_size
		relation_vec = self.relation_dict['image']['embedding']										
		example_emb = self.entity_dict[head_entity_name]['embedding']
		label_idxs = self.relation_dict['image']['idxs']
		label_bias = self.relation_dict['image']['bias']
		label_size = len(self.relation_dict['image']['distribute'])
		label_distribution = self.relation_dict['image']['distribute']

		# Negative sampling.
		labels_matrix = tf.reshape(tf.cast(label_idxs,dtype=tf.int64),[batch_size, 1])
		sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
				true_classes=labels_matrix,
				num_true=1,
				num_sampled=self.negative_sample,
				unique=False,
				range_max=label_size,
				distortion=0.75,
				unigrams=label_distribution))

		#get example embeddings [batch_size, embed_size]
		example_vec = tf.nn.embedding_lookup(example_emb, example_idxs) * (1-add_weight) + relation_vec * add_weight							
		# get label embeddings and bias [batch_size, embed_size], [batch_size, 1]
		true_image_features = tf.nn.embedding_lookup(self.img_product_features, label_idxs)
		true_w = self.decode_img(true_image_features, None)
		true_b = tf.nn.embedding_lookup(label_bias, label_idxs)	
											
		#get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]										
		sampled_image_features = tf.nn.embedding_lookup(self.img_product_features, sampled_ids)
		sampled_w = self.decode_img(sampled_image_features, True)									
		sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)			

		# True logits: [batch_size, 1]										
		true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b	

		# Sampled logits: [batch_size, num_sampled]										
		# We replicate sampled noise lables for all examples in the batch										
		# using the matmul.										
		sampled_b_vec = tf.reshape(sampled_b, [self.negative_sample])										
		sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec										
												
		return tf.reduce_sum(self.relation_dict['image']['weight'] * nce_loss(true_logits, sampled_logits)), [example_vec, true_w, sampled_w]	

	
	def step(self, session, input_feed, forward_only, test_mode = 'product_scores'):
		"""Run a step of the model feeding the given inputs.
	
		Args:
			session: tensorflow session to use.
			learning_rate: the learning rate of current step
			user_idxs: A numpy [1] float vector.
			product_idxs: A numpy [1] float vector.
			review_idxs: A numpy [1] float vector.
			word_idxs: A numpy [None] float vector.
			context_idxs: list of numpy [None] float vectors.
			product_img_features: image features for the product
			neg_product_idxs: negative sample indexes for image training 
			neg_img_features: negative samples' image feature
			forward_only: whether to do the update step or only forward.
	
		Returns:
			A triple consisting of gradient norm (or None if we did not do backward),
			average perplexity, and the outputs.
	
		Raises:
			ValueError: if length of encoder_inputs, decoder_inputs, or
			target_weights disagrees with bucket size for the specified bucket_id.
		"""
	
		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates,	# Update Op that does SGD.
						 #self.norm,	# Gradient norm.
						 self.loss]	# Loss for this batch.
		else:
			if test_mode == 'output_embedding':
				self.embed_output_keys = []
				output_feed = []
				for key in self.entity_dict:
					self.embed_output_keys.append(key)
					output_feed.append(self.entity_dict[key]['embedding'])
				for key in self.relation_dict:
					self.embed_output_keys.append(key + '_embed')
					output_feed.append(self.relation_dict[key]['embedding'])
				for key in self.relation_dict:
					self.embed_output_keys.append(key + '_bias')
					output_feed.append(self.relation_dict[key]['bias'])
			elif 'explain' in test_mode:
				if test_mode == 'explain_user_query':
					entity_list = self.up_entity_list
				elif test_mode == 'explain_product':
					entity_list = self.p_entity_list
				output_feed = [scores for _, _, scores in entity_list]
			else:
				output_feed = [self.product_scores] #negative instance output
	
		outputs = session.run(output_feed, input_feed)

		if not forward_only:
			return outputs[1], None	# loss, no outputs, Gradient norm.
		else:
			if test_mode == 'output_embedding':
				return outputs, self.embed_output_keys
			elif 'explain' in test_mode:
				return [(entity_list[i][0], entity_list[i][1], outputs[i]) for i in xrange(len(entity_list))], None
			else:
				return outputs[0], None	# product scores to input user

	def setup_data_set(self, data_set, words_to_train):
		self.data_set = data_set
		self.words_to_train = words_to_train
		self.finished_word_num = 0
		if self.net_struct == 'hdc':
			self.need_context = True

	def intialize_epoch(self, training_seq):
		self.train_seq = training_seq
		self.review_size = len(self.train_seq)
		self.cur_review_i = 0
		self.cur_word_i = 0
		self.tested_user = set()

	def get_train_batch(self):
		user_idxs, product_idxs, word_idxs = [],[],[]
		knowledge_idxs_dict = {
			'also_bought' : [],
			'also_viewed' : [],
			'bought_together' : [],
			'brand' : [],
			'categories' : []
		}
		knowledge_weight_dict = {
			'also_bought' : [],
			'also_viewed' : [],
			'bought_together' : [],
			'brand' : [],
			'categories' : []
		}
		#product_img_features, neg_product_idxs, neg_img_features = [],[],[] #add image
		learning_rate = self.init_learning_rate * max(0.0001, 
									1.0 - self.finished_word_num / self.words_to_train)
		review_idx = self.train_seq[self.cur_review_i]
		user_idx = self.data_set.review_info[review_idx][0]
		product_idx = self.data_set.review_info[review_idx][1]
		text_list = self.data_set.review_text[review_idx]
		text_length = len(text_list)
		# add knowledge
		product_knowledge = {key : self.data_set.knowledge[key]['data'][product_idx] for key in knowledge_idxs_dict}

		while len(word_idxs) < self.batch_size:
			#print('review %d word %d word_idx %d' % (review_idx, self.cur_word_i, text_list[self.cur_word_i]))
			#if sample this word
			if random.random() < self.data_set.sub_sampling_rate[text_list[self.cur_word_i]]:
				user_idxs.append(user_idx)
				product_idxs.append(product_idx)
				word_idxs.append(text_list[self.cur_word_i])
				# Add knowledge
				for key in product_knowledge:
					if len(product_knowledge[key]) < 1:
						knowledge_idxs_dict[key].append(-1)
						knowledge_weight_dict[key].append(0.0)
					else:
						knowledge_idxs_dict[key].append(random.choice(product_knowledge[key]))
						knowledge_weight_dict[key].append(1.0)

			#move to the next
			self.cur_word_i += 1
			self.finished_word_num += 1
			if self.cur_word_i == text_length:
				self.cur_review_i += 1
				if self.cur_review_i == self.review_size:
					break
				self.cur_word_i = 0
				review_idx = self.train_seq[self.cur_review_i]
				user_idx = self.data_set.review_info[review_idx][0]
				product_idx = self.data_set.review_info[review_idx][1]
				text_list = self.data_set.review_text[review_idx]
				text_length = len(text_list)
				# add knowledge
				product_knowledge = {key : self.data_set.knowledge[key]['data'][product_idx] for key in knowledge_idxs_dict}

		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		input_feed = {}
		input_feed[self.learning_rate.name] = learning_rate
		input_feed[self.user_idxs.name] = user_idxs
		input_feed[self.relation_dict['product']['idxs'].name] = product_idxs
		input_feed[self.relation_dict['product']['weight'].name] = [1.0 for _ in xrange(len(product_idxs))]
		#input_feed[self.relation_dict['image']['idxs'].name] = product_idxs
		#input_feed[self.relation_dict['image']['weight'].name] = [1.0 for _ in xrange(len(product_idxs))]
		input_feed[self.relation_dict['word']['idxs'].name] = word_idxs
		input_feed[self.relation_dict['word']['weight'].name] = [1.0 for _ in xrange(len(word_idxs))]
		for key in knowledge_idxs_dict:
			input_feed[self.relation_dict[key]['idxs'].name] = knowledge_idxs_dict[key]
			input_feed[self.relation_dict[key]['weight'].name] = knowledge_weight_dict[key]

		has_next = False if self.cur_review_i == self.review_size else True
		return input_feed, has_next

	def get_test_batch(self):
		user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
		knowledge_idxs_dict = {
			'also_bought' : [],
			'also_viewed' : [],
			'bought_together' : [],
			'brand' : [],
			'categories' : []
		}
		knowledge_weight_dict = {
			'also_bought' : [],
			'also_viewed' : [],
			'bought_together' : [],
			'brand' : [],
			'categories' : []
		}
		learning_rate = self.init_learning_rate * max(0.0001, 
									1.0 - self.finished_word_num / self.words_to_train)
		review_idx = self.train_seq[self.cur_review_i]
		user_idx = self.data_set.review_info[review_idx][0]

		# add image
		#product_img_features = self.data_set.img_features

		while len(user_idxs) < self.batch_size:
			if user_idx not in self.tested_user:
				product_idx = self.data_set.review_info[review_idx][1]
				text_list = self.data_set.review_text[review_idx]
				user_idxs.append(user_idx)
				product_idxs.append(product_idx)
				review_idxs.append(review_idx)
				word_idxs.append(text_list[0])
				# Add knowledge
				for key in knowledge_idxs_dict:
					knowledge_idxs_dict[key].append(-1)
					knowledge_weight_dict[key].append(0.0)
				self.tested_user.add(user_idx)
			
			#move to the next review
			self.cur_review_i += 1
			if self.cur_review_i == self.review_size:
				break
			review_idx = self.train_seq[self.cur_review_i]
			user_idx = self.data_set.review_info[review_idx][0]

		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		input_feed = {}
		input_feed[self.learning_rate.name] = learning_rate
		input_feed[self.user_idxs.name] = user_idxs
		input_feed[self.relation_dict['product']['idxs'].name] = product_idxs
		input_feed[self.relation_dict['product']['weight'].name] = [1.0 for _ in xrange(len(product_idxs))]
		#input_feed[self.relation_dict['image']['idxs'].name] = product_idxs
		#input_feed[self.relation_dict['image']['weight'].name] = [1.0 for _ in xrange(len(product_idxs))]
		input_feed[self.relation_dict['word']['idxs'].name] = word_idxs
		input_feed[self.relation_dict['word']['weight'].name] = [1.0 for _ in xrange(len(word_idxs))]
		for key in knowledge_idxs_dict:
			input_feed[self.relation_dict[key]['idxs'].name] = knowledge_idxs_dict[key]
			input_feed[self.relation_dict[key]['weight'].name] = knowledge_weight_dict[key]
		

		has_next = False if self.cur_review_i == self.review_size else True
		return input_feed, has_next

def relation_nce_loss(model, add_weight, example_idxs, head_entity_name, relation_name, tail_entity_name):
	relation_vec = model.relation_dict[relation_name]['embedding']
	example_emb = model.entity_dict[head_entity_name]['embedding']
	label_idxs = model.relation_dict[relation_name]['idxs']
	label_emb = model.entity_dict[tail_entity_name]['embedding']
	label_bias = model.relation_dict[relation_name]['bias']
	label_size = model.entity_dict[tail_entity_name]['size']
	label_distribution = model.relation_dict[relation_name]['distribute']
	loss, embs = pair_search_loss(model, add_weight, relation_vec, example_idxs, example_emb, label_idxs, label_emb, label_bias, label_size, label_distribution)
	#print(loss.get_shape())
	#print(model.relation_dict[relation_name]['weight'].get_shape())
	return tf.reduce_sum(model.relation_dict[relation_name]['weight'] * loss), embs									

def pair_search_loss(model, add_weight, relation_vec, example_idxs, example_emb, label_idxs, label_emb, label_bias, label_size, label_distribution):
	batch_size = array_ops.shape(example_idxs)[0]#get batch_size										
	# Nodes to compute the nce loss w/ candidate sampling.										
	labels_matrix = tf.reshape(tf.cast(label_idxs,dtype=tf.int64),[batch_size, 1])										
											
	# Negative sampling.										
	sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(										
			true_classes=labels_matrix,								
			num_true=1,								
			num_sampled=model.negative_sample,								
			unique=False,								
			range_max=label_size,								
			distortion=0.75,								
			unigrams=label_distribution))								
											
	#get example embeddings [batch_size, embed_size]
	#example_vec = tf.nn.embedding_lookup(example_emb, example_idxs) * (1-add_weight) + relation_vec * add_weight							
	example_vec = tf.nn.embedding_lookup(example_emb, example_idxs) + relation_vec
											
	#get label embeddings and bias [batch_size, embed_size], [batch_size, 1]										
	true_w = tf.nn.embedding_lookup(label_emb, label_idxs)										
	true_b = tf.nn.embedding_lookup(label_bias, label_idxs)										
											
	#get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]										
	sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)										
	sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)										
											
	# True logits: [batch_size, 1]										
	true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b										
											
	# Sampled logits: [batch_size, num_sampled]										
	# We replicate sampled noise lables for all examples in the batch										
	# using the matmul.										
	sampled_b_vec = tf.reshape(sampled_b, [model.negative_sample])										
	sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec										
											
	return nce_loss(true_logits, sampled_logits), [example_vec, true_w, sampled_w]						
											
def nce_loss(true_logits, sampled_logits):											
	"Build the graph for the NCE loss."										
											
	# cross-entropy(logits, labels)										
	true_xent = tf.nn.sigmoid_cross_entropy_with_logits(										
			logits=true_logits, labels=tf.ones_like(true_logits))								
	sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(										
			logits=sampled_logits, labels=tf.zeros_like(sampled_logits))								
	#print(true_xent.get_shape())
	#print(sampled_xent.get_shape())
	# NCE-loss is the sum of the true and noise (sampled words)										
	# contributions, averaged over the batch.										
	nce_loss_tensor = (true_xent + tf.reduce_sum(sampled_xent, 1)) 										
	return nce_loss_tensor	
