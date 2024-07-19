import argparse
import copy
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from time import time
from tensorflow.python.ops.nn_ops import leaky_relu
from training.load_data import load_EOD_data, load_relation_data
from training.evaluator import evaluate

class ReRaLSTM:
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 emb_fname, parameters, steps=1, epochs=50, batch_size=None, 
                 flat=False, gpu=False, in_pro=False, output_dir = 'SAVED_MODEL/model-ep1'):
        # Set seeds for reproducibility
        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        # Fix attributes (paths, market/ticker/relation name)
        self.model_path = output_dir
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        self.steps = steps
        self.sess = None
        
        # Load data
        self.load_data()
        
        # Load relation data
        self.load_relational_data()
        print('relation encoding shape:', self.rel_encoding.shape)
        print('relation mask shape:', self.rel_mask.shape)
        
        # Load embedding
        self.embedding = np.load(
            os.path.join(self.data_path, '..', 'pretrain', emb_fname))
        print('embedding shape:', self.embedding.shape)
        
        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        self.batch_size = batch_size if batch_size else len(self.tickers)
        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5
        self.gpu = gpu

    def load_data(self):
        self.tickers = np.genfromtxt(os.path.join(self.data_path, '..', self.tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        print('#tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(self.data_path, self.market_name, self.tickers, self.steps)

    def load_relational_data(self):
        rname_tail = {'sector_industry': '_industry_relation.npy',
                      'wikidata': '_wiki_relation.npy'}
        rel_encoding_wiki, rel_mask_wiki = load_relation_data(
                                             os.path.join('data', 'relation', 'wikidata',
                                             self.market_name + rname_tail['wikidata'])
                                             )
        rel_encoding_indus, rel_mask_indus = load_relation_data(
                        os.path.join('data', 'relation', 'sector_industry',
                         self.market_name + rname_tail['sector_industry'])
        )
        # Concatenate relation encodings along the last dimension (axis=2)
        self.rel_encoding = np.concatenate((rel_encoding_wiki, rel_encoding_indus), axis=2)

        # Combine masks using logical OR operation and then adjust the values
        combined_mask_flags = np.logical_or(rel_mask_wiki == -1e9, rel_mask_indus == -1e9)
        self.rel_mask = np.where(combined_mask_flags, -1e9, 0)
        print('relation encoding shape:', self.rel_encoding.shape)
        print('relation mask shape:', self.rel_mask.shape)
    
    def build_model(self):
        with tf.device('/gpu:0' if self.gpu else '/cpu:0'):
            tf.reset_default_graph()
            seed = 123456789
            random.seed(seed)
            np.random.seed(seed)
            tf.set_random_seed(seed)
            
            ground_truth = tf.placeholder(tf.float32, [self.batch_size, 1])
            mask = tf.placeholder(tf.float32, [self.batch_size, 1])
            feature = tf.placeholder(tf.float32, [self.batch_size, self.parameters['unit']])
            base_price = tf.placeholder(tf.float32, [self.batch_size, 1])
            all_one = tf.ones([self.batch_size, 1], dtype=tf.float32)

            relation = tf.constant(self.rel_encoding, dtype=tf.float32)
            rel_mask = tf.constant(self.rel_mask, dtype=tf.float32)

            rel_weight = tf.layers.dense(relation, units=1, activation=leaky_relu)

            if self.inner_prod:
                print('inner product weight')
                inner_weight = tf.matmul(feature, feature, transpose_b=True)
                weight = tf.multiply(inner_weight, rel_weight[:, :, -1])
            else:
                print('sum weight')
                head_weight = tf.layers.dense(feature, units=1, activation=leaky_relu)
                tail_weight = tf.layers.dense(feature, units=1, activation=leaky_relu)
                weight = tf.add(tf.add(tf.matmul(head_weight, all_one, transpose_b=True),
                                       tf.matmul(all_one, tail_weight, transpose_b=True)), 
                                rel_weight[:, :, -1])
            weight_masked = tf.nn.softmax(tf.add(rel_mask, weight), axis=0)
            outputs_proped = tf.matmul(weight_masked, feature)
            if self.flat:
                print('one more hidden layer')
                outputs_concated = tf.layers.dense(
                    tf.concat([feature, outputs_proped], axis=1),
                    units=self.parameters['unit'], activation=leaky_relu,
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
            else:
                outputs_concated = tf.concat([feature, outputs_proped], axis=1)

            prediction = tf.layers.dense(
                outputs_concated, units=1, activation=leaky_relu, name='reg_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            return_ratio = tf.div(tf.subtract(prediction, base_price), base_price)
            reg_loss = tf.losses.mean_squared_error(
                ground_truth, return_ratio, weights=mask
            )
            pre_pw_dif = tf.subtract(
                tf.matmul(return_ratio, all_one, transpose_b=True),
                tf.matmul(all_one, return_ratio, transpose_b=True)
            )
            gt_pw_dif = tf.subtract(
                tf.matmul(all_one, ground_truth, transpose_b=True),
                tf.matmul(ground_truth, all_one, transpose_b=True)
            )
            mask_pw = tf.matmul(mask, mask, transpose_b=True)
            rank_loss = tf.reduce_mean(
                tf.nn.relu(
                    tf.multiply(
                        tf.multiply(pre_pw_dif, gt_pw_dif),
                        mask_pw
                    )
                )
            )
            loss = reg_loss + tf.cast(self.parameters['alpha'], tf.float32) * rank_loss
            optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters['lr']).minimize(loss)

            # Store the necessary components in the class for later use
            self.feature = feature
            self.mask = mask
            self.ground_truth = ground_truth
            self.base_price = base_price
            self.return_ratio = return_ratio
            self.loss = loss
            self.reg_loss = reg_loss
            self.rank_loss = rank_loss
            self.optimizer = optimizer

            print('Model built successfully.')

    def train(self):
        if self.sess == None:
            sess = tf.Session()
        else:
            sess = self.sess
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        best_valid_pred = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
        best_valid_gt = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
        best_valid_mask = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
        best_test_pred = np.zeros([len(self.tickers), self.trade_dates - self.parameters['seq'] - self.test_index - self.steps + 1], dtype=float)
        best_test_gt = np.zeros([len(self.tickers), self.trade_dates - self.parameters['seq'] - self.test_index - self.steps + 1], dtype=float)
        best_test_mask = np.zeros([len(self.tickers), self.trade_dates - self.parameters['seq'] - self.test_index - self.steps + 1], dtype=float)
        best_valid_perf = {'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0}
        best_test_perf = {'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0}
        best_valid_loss = np.inf
    
        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            for j in range(self.valid_index - self.parameters['seq'] - self.steps + 1):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(batch_offsets[j])
                feed_dict = {
                    self.feature: emb_batch,
                    self.mask: mask_batch,
                    self.ground_truth: gt_batch,
                    self.base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, batch_out = sess.run(
                    (self.loss, self.reg_loss, self.rank_loss, self.optimizer),
                    feed_dict
                )
                tra_loss += cur_loss
                tra_reg_loss += cur_reg_loss
                tra_rank_loss += cur_rank_loss
            print('Train Loss:', tra_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_reg_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_rank_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1))
    
            # test on validation set
            cur_valid_pred = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
            cur_valid_gt = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
            cur_valid_mask = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
            val_loss = 0.0
            val_reg_loss = 0.0
            val_rank_loss = 0.0
            for cur_offset in range(self.valid_index - self.parameters['seq'] - self.steps + 1,
                                    self.test_index - self.parameters['seq'] - self.steps + 1):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)
                feed_dict = {
                    self.feature: emb_batch,
                    self.mask: mask_batch,
                    self.ground_truth: gt_batch,
                    self.base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr, = sess.run(
                    (self.loss, self.reg_loss, self.rank_loss, self.return_ratio), 
                    feed_dict
                )
                val_loss += cur_loss
                val_reg_loss += cur_reg_loss
                val_rank_loss += cur_rank_loss
                cur_valid_pred[:, cur_offset - (self.valid_index - self.parameters['seq'] - self.steps + 1)] = copy.copy(cur_rr[:, 0])
                cur_valid_gt[:, cur_offset - (self.valid_index - self.parameters['seq'] - self.steps + 1)] = copy.copy(gt_batch[:, 0])
                cur_valid_mask[:, cur_offset - (self.valid_index - self.parameters['seq'] - self.steps + 1)] = copy.copy(mask_batch[:, 0])
            print('\nValid Loss:', val_loss / (self.test_index - self.valid_index),
                  val_reg_loss / (self.test_index - self.valid_index),
                  val_rank_loss / (self.test_index - self.valid_index))
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
            print('\n Valid performance:', cur_valid_perf)
    
            # test on testing set
            cur_test_pred = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
            cur_test_gt = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
            cur_test_mask = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
            test_loss = 0.0
            test_reg_loss = 0.0
            test_rank_loss = 0.0
            for cur_offset in range(self.test_index - self.parameters['seq'] - self.steps + 1,
                                    self.trade_dates - self.parameters['seq'] - self.steps + 1):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)
                feed_dict = {
                    self.feature: emb_batch,
                    self.mask: mask_batch,
                    self.ground_truth: gt_batch,
                    self.base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = sess.run(
                    (self.loss, self.reg_loss, self.rank_loss, self.return_ratio), 
                    feed_dict
                )
                test_loss += cur_loss
                test_reg_loss += cur_reg_loss
                test_rank_loss += cur_rank_loss
                cur_test_pred[:, cur_offset - (self.test_index - self.parameters['seq'] - self.steps + 1)] = copy.copy(cur_rr[:, 0])
                cur_test_gt[:, cur_offset - (self.test_index - self.parameters['seq'] - self.steps + 1)] = copy.copy(gt_batch[:, 0])
                cur_test_mask[:, cur_offset - (self.test_index - self.parameters['seq'] - self.steps + 1)] = copy.copy(mask_batch[:, 0])
            print('\nTest Loss:', test_loss / (self.trade_dates - self.test_index),
                  test_reg_loss / (self.trade_dates - self.test_index),
                  test_rank_loss / (self.trade_dates - self.test_index))
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            print('\t Test performance:', cur_test_perf)
    
            if val_loss / (self.test_index - self.valid_index) < best_valid_loss:
                best_valid_loss = val_loss / (self.test_index - self.valid_index)
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_gt = copy.copy(cur_valid_gt)
                best_valid_pred = copy.copy(cur_valid_pred)
                best_valid_mask = copy.copy(cur_valid_mask)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_gt = copy.copy(cur_test_gt)
                best_test_pred = copy.copy(cur_test_pred)
                best_test_mask = copy.copy(cur_test_mask)
                print('Better valid loss:', best_valid_loss)
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
        
        print('\nBest Valid performance:', best_valid_perf)
        print('\nBest Test performance:', best_test_perf)
        saver.save(sess, self.model_path)
        sess.close()
        tf.reset_default_graph()
        return best_valid_pred, best_valid_gt, best_valid_mask, best_test_pred, best_test_gt, best_test_mask

    def save_model(self, sess, model_path = 'output_model'):
        if not os.exists(model_path):
            os.mkdir(model_path)
        saver = tf.train.Saver()
        saver.save(sess, model_path)
        print(f'Model saved to {model_path}')

    def load_model(self, filepath):
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, filepath)
        print(f'Model restored from {filepath}')
        self.sess = sess

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.embedding[:, offset, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )
               
    
    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True

