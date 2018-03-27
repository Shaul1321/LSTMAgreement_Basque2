import dynet as dy
import numpy as np
import random
import time
import gc
from collections import Counter

import dynet_config
dynet_config.set_gpu()
#from googletrans import Translator

NUM_LAYERS = 1
EMBEDDING_SIZE = 150
ATTENTION_HIDDENSIZE = 20
LSTM_HIDDENSIZE = 256
DROPOUT_RATE = 0.0

ABSO = {"<NR_HK>": "absolutive: pl3", "<NR_HU>": "absolutive: sg3", "<NR_HI>": "absolutive; ??",
"<NR_GU>": "absolutive :1pl", "<NR_NI>":"absolutive: 1sg", "<NR_ZU>": "absolutive: 2sg", "<NR_ZK>": "absolutive: 2pl", "None": "absolutive: None"}

ERG = {"<NK_HU>": "ergative: sg3", "<NK_HK>": "ergative: pl3", "<NK_HI>": "ergative: ??",
"<NK_GU>": "ergative: 1pl", "<NK_NI>": "ergative: 1sg", "<NK_ZU>": "ergative: 2sg", "<NK_ZK>": "ergative: 2pl", "None": "ergative: None"}

DAT = {"<NI_HU>": "dative: sg3", "<NI_HK>": "dative: pl3", "<NI_HI>": "dative: ??",
"<NI_GU>": "dative: 1pl", "<NI_NI>": "dative: 1sg", "<NI_ZU>": "dative: 2sg", "<NI_ZK>": "dative: 2pl", "None": "dative: None"}


def write_errors(good, bad, total):
 f = open("error_rates3__task2.txt", "w")
 items = sorted(total.items(), key = lambda (key, val): -val)
 for (key, val) in items:
	good_count = good[key] 
	bad_count =  bad[key]
	f.write(str(key).replace(" ", "")+"\t"+str(good_count)+"\t"+str(bad_count)+"\n")
 f.close()


class RNN(object):

	def __init__(self, in_size, hid_size, out_size, dataGenerator, I2A, I2E, I2D, I2W, model, encoder):

		self.in_size = in_size
		self.hid_size = hid_size
		self.out_size = out_size

		self.I2A = I2A
		self.I2E = I2E
		self.I2D = I2D
		self.I2W = I2W

		self.generator = dataGenerator
		self.model = model
		self.encoder = encoder
		self.attention_fwd, self.attention_bwd = None, None
		self.create_model()

		self.subj_dev = []
		self.obj_dev = []
		self.subj_train = []
		self.obj_train = []
		

	def create_model(self):

                """add parameters to the model, that consists of a biLSTM layer(s),
		and 3 softmax layers (for dative (d), ergative (e) and absolutive (a) agreement prediction).

		W_attention1, W_attention2 - attention matrices
		W_ha, W_he, W_hd - hidden-output matrices for absolutive, ergative and dative.
		W_hh, W_hh2 - hidden-hidden matrices
		fwdLSTM, bwdLSTM - builders for bidirectional lstm layers
		"""

		self.W_attention1 = self.model.add_parameters((ATTENTION_HIDDENSIZE, LSTM_HIDDENSIZE))
		self.b_attention1 = self.model.add_parameters((ATTENTION_HIDDENSIZE, 1))
		self.W_attention2 = self.model.add_parameters((1, ATTENTION_HIDDENSIZE))

		hid = 64

		self.W_ho = self.model.add_parameters((self.out_size, hid))
		self.W_hh = self.model.add_parameters((hid, LSTM_HIDDENSIZE))
		self.b_hh = self.model.add_parameters((hid, 1))
		self.W_hh2 = self.model.add_parameters((hid, hid))
		self.W_hh3 = self.model.add_parameters((hid, hid))
		self.W_hh4 = self.model.add_parameters((hid, hid))

		self.LSTM = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, LSTM_HIDDENSIZE, self.model)

		#self.trainer = dy.AdagradTrainer(self.model)
		self.trainer = dy.AdamTrainer(self.model)
		#self.trainer.learning_rate = 0.05
        

	def _attend(self, states, training=True):

                """computes attention weights over the lsmt states
		- encoded_sent is the encoding of the input sentence
		- states is a list of lsmt states (one for each encoded word)
		
		returns: weights, an array of attention weights

		"""

		#assert len(encoded_sent)==len(states)

		W_attention1 = dy.parameter(self.W_attention1)
		b_attention1 = dy.parameter(self.b_attention1)
		W_attention2 = dy.parameter(self.W_attention2)

		# pass the concatenation of the words & the states through a hidden layer, then softmax

		#h = [dy.rectify(W_attention1*dy.concatenate([w,s])) for w,s in zip(encoded_sent,states)]
		drop = 0.#DROPOUT_RATE if not training else 0.
		h = [dy.dropout(dy.rectify(W_attention1*s + b_attention1), drop) for s in states]
		#if training: h = dy.dropout(h, DROPOUT_RATE)
		#weights = dy.concatenate([W_attention2*h_elem if h.index(h_elem)!=len(h)-1 else dy.scalarInput(0.) for h_elem in h]) to force the network not attend the <verb> token

		weights = dy.concatenate([W_attention2*h_elem for h_elem in h])
		weights =  dy.softmax(weights)
	

		#assert len(weights.npvalue())==len(encoded_sent)

		return weights


        def _predict(self, sentence, output, lemmas, training=True, dropout_rate = 0.1):	

                """predict the agreement of the subject, object and indirect object.

		- sentence - a list of sentence words (as strings)
		
		returns: a_pred, e_pred, d_pred - absolutive, ergative and dative agreements

		"""

		"""
		if training:
			self.fwdLSTM.set_dropout(DROPOUT_RATE)
			self.bwdLSTM.set_dropout(DROPOUT_RATE)
		else:
			self.fwdLSTM.disable_dropout()
			self.bwdLSTM.disable_dropout()	
		"""
		#prepare parameters

		W_ho = dy.parameter(self.W_ho)

		W_hh = dy.parameter(self.W_hh)
		b_hh = dy.parameter(self.b_hh)
		W_hh2 = dy.parameter(self.W_hh2)
		W_hh3 = dy.parameter(self.W_hh3)
		W_hh4 = dy.parameter(self.W_hh4)

		s = self.LSTM.initial_state()

		# encode sentence & pass through biLstm

		encoded = [self.encoder.encode(w,o,l) for (w,o,l) in zip(sentence,output,lemmas)]

		#if training: encoded = [dy.dropout(e, DROPOUT_RATE) for e in encoded]

		output_states = s.transduce(encoded)

		# attend over bilstm states

		weights = self._attend(output_states, training=training)
		self.attention = weights
		weighted_states = dy.esum([o*w for o,w in zip(output_states, weights)])

		h = dy.rectify(W_hh * weighted_states + b_hh)
		if training: h = dy.dropout(h, DROPOUT_RATE)
		h = dy.rectify(W_hh2 * h)
		if training: h = dy.dropout(h, DROPOUT_RATE)
		#h = dy.rectify(W_hh3 * h)
		#h = dy.rectify(W_hh4 * h)

		# predict absolutive, ergative and dative agreements.

		pred = dy.softmax(W_ho * h)

		return pred

	def encode(self, sentence):

		"""encode the sentence words with the encoder"""

		return [self.encoder.encode(w) for w in sentence]

        def train(self, epochs=30):

	  n = self.generator.get_train_size()
	  print "size of training set: ", n
          print "training..."

	  iteration = 0
	  good, bad = 1., 1.
	  losses = []

	  for i, batch in enumerate(self.generator.generate(is_training=True)):

		dy.renew_cg()

		if i%500 == 0:
			print "{}".format(i)
			print "training accuracy: {}".format(good/(good+bad))
			gc.collect()
	
		for j, training_example in enumerate(batch):


			iteration+=1

			#stopping criteria
	
                        if iteration > epochs*n: 
				#print "Calcualting accuracy on test set. This may take some time"
				#self.test(Mode.TEST)
				return

			# report progress. 

			if iteration%n == 0:

				iteration = 0
				print "EPOCH {} / {}".format(iteration/n, epochs)


			# prepare input & predict

			(sent, output, lemmas), (gold, (a,e,d)), data_sample = training_example

			#print sent
			#print "============"			
			#for (w,o) in zip(sent, output):
			#	print w, o
			#print "============"

			pred = self._predict(sent,output,lemmas, training=True)

			# collect loss & errors

			loss = -dy.log(pred[gold]) 
			pred_val = np.argmax(pred.npvalue())

			if pred_val==gold:
				good+=1
			else:
				bad+=1


			losses.append(loss)

		# backprop

		loss_sum = dy.esum(losses)
		#loss_sum.forward()
		loss_sum.backward()
		self.trainer.update()
		losses = []

		# check dev set accuracy

		if i%(2500) == 0 and i>0:

			print "iteration {} / {}".format(iteration, n)
			print "Calculating accuracy on dev set."
			self.test()
			print "train accuracy: {}.".format(good/(good+bad))

			good, bad = 1., 1.
			if i >=65000: return


        def test(self, train_set=False):
	   good_preds, bad_preds = Counter(), Counter()
	   labels = Counter()
	   good, bad = 0., 0.
	   true_labels = Counter()
	   n = self.generator.get_dev_size()
	   iteration = 0
	   for i, batch in enumerate(self.generator.generate(is_training=False)):
		dy.renew_cg()

		for j, training_example in enumerate(batch):

			iteration+=1
			(sent, output, lemmas), (gold, (a,e,d)), data_sample = training_example
			a,e,d = ABSO[self.I2A[a]], ERG[self.I2E[e]], DAT[self.I2D[d]]
			labels[(a,e,d)]+=1
			y_hat = self._predict(sent,output,lemmas, training=False)
			#loss = -dy.log(a_hat[a_true]) + -dy.log(e_hat[e_true])# + -dy.log(d_hat[d_true])

			pred = np.argmax(y_hat.npvalue())
			if iteration%5000==0: 
				print iteration
				print "gold: ", gold
				att = self.attention.npvalue()
				att = [round(a,3) for a in att]
				attention_and_words = [(w,a) for w, a in zip(sent, att)]
				most_attended = sorted(attention_and_words, key = lambda pair: -pair[1])		
				print "attention:"
				print attention_and_words
				print "====="
				print most_attended
				print "============================="

			if pred==gold:
				good+=1
				good_preds[(a,e,d)]+=1
			else:
				bad+=1
				bad_preds[(a,e,d)]+=1
		
			#if d_pred==d_true:
			#	good_d+=1
			#else:
			#	bad_d+=1


			if iteration > n:

	   			print "accuracy: {}".format(good/(good+bad))
				write_errors(good_preds, bad_preds, labels)
				return

	




