import dynet as dy
import numpy as np
import random
import time
import gc
from collections import Counter

#import dynet_config
#dynet_config.set_gpu()
#from googletrans import Translator

NUM_LAYERS = 1
EMBEDDING_SIZE = 150
ATTENTION_HIDDENSIZE = 32
LSTM_HIDDENSIZE = 150
DROPOUT_RATE = 0.0

I2SUFFIX = {0: "ak", 1: "ek", 2: "a", 3: "", 4: "ei", 5: "ari"}
SUFFIX2I = {"ak": 0, "ek": 1, "a":2, "": 3, "ei": 4, "ari": 5}

class RNN(object):

	def __init__(self, in_size, hid_size, (a_out, e_out, d_out), dataGenerator, I2A, I2E, I2D, I2W, model, encoder, embedding_collector, states_collector):

		self.in_size = in_size
		self.hid_size = hid_size
		self.a_out = a_out
		self.e_out = e_out
		self.d_out = d_out

		self.I2A = I2A
		self.I2E = I2E
		self.I2D = I2D
		self.I2W = I2W

		self.generator = dataGenerator
		self.model = model
		self.encoder = encoder
		self.attention_fwd, self.attention_bwd = None, None
		self.attention = []
		self.create_model()
		self.embedding_collector = embedding_collector
		self.states_collector = states_collector
		
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

		self.W_attention1 = self.model.add_parameters((ATTENTION_HIDDENSIZE, LSTM_HIDDENSIZE+LSTM_HIDDENSIZE))
		self.b_attention1 = self.model.add_parameters((ATTENTION_HIDDENSIZE, 1))
		self.W_attention2 = self.model.add_parameters((1, ATTENTION_HIDDENSIZE))
		self.W_attention_simple = self.model.add_parameters((1, 2*EMBEDDING_SIZE))

		hid = 128

		self.W =  self.model.add_parameters((6,  hid))
		self.W_hh = self.model.add_parameters((hid,  LSTM_HIDDENSIZE))

		self.b =   self.model.add_parameters((1,  1))
		self.b_hh = self.model.add_parameters((hid, 1))
		self.biLSTM_fwd = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, LSTM_HIDDENSIZE, self.model)
		self.biLSTM_bwd = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, LSTM_HIDDENSIZE, self.model)
		self.LSTM = dy.LSTMBuilder(NUM_LAYERS, LSTM_HIDDENSIZE, LSTM_HIDDENSIZE, self.model)

		#self.trainer = dy.AdagradTrainer(self.model)
		self.trainer = dy.AdamTrainer(self.model)
		#self.trainer = dy.SimpleSGDTrainer(self.model)
		self.trainer.set_clip_threshold(1.0)
        

	def _attend(self, word_repr, states, training=True):

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

		h = [dy.rectify(W_attention1*dy.concatenate([word_repr, s])) for s in states]
		drop = 0.#DROPOUT_RATE if not training else 0.
		#h = [dy.dropout(dy.rectify(W_attention1*s + b_attention1), drop) for s in states]
		#if training: h = dy.dropout(h, DROPOUT_RATE)
		#weights = dy.concatenate([W_attention2*h_elem if h.index(h_elem)!=len(h)-1 else dy.scalarInput(0.) for h_elem in h]) to force the network not attend the <verb> token

		weights = dy.concatenate([W_attention2*h_elem for h_elem in h])
		weights =  dy.softmax(weights)
	
		self.attention.append(weights.npvalue())
		#assert len(weights.npvalue())==len(encoded_sent)
		weighted_states = dy.esum([o*w for o,w in zip(states, weights)]) + word_repr

		return weighted_states


	def _attend2(self, word_repr, states, training=True):

		w = dy.parameter(self.W_attention_simple)
		scores = dy.concatenate([w*dy.concatenate([s, word_repr])  for s in states])
		weights = dy.softmax(scores)
		self.attention.append(weights)
		weighted_states = dy.esum([o*w for o,w in zip(states, weights)])
		return weighted_states
		


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

		W = dy.parameter(self.W)
		W_hh = dy.parameter(self.W_hh)
		b = dy.parameter(self.b)
		b_hh = dy.parameter(self.b_hh)
		s_fwd = self.biLSTM_fwd.initial_state()
		s_bwd = self.biLSTM_bwd.initial_state()
		s = self.LSTM.initial_state()

		# encode sentence & pass through biLstm

		encoded = [self.encoder.encode(w,o,l) for (w,o,l) in zip(sentence,output,lemmas)]

		#if training: encoded = [dy.dropout(e, DROPOUT_RATE) for e in encoded]
		
		output_states_fwd = s_fwd.transduce(encoded)
		output_states_bwd = s_bwd.transduce(encoded[::-1])
		bilstm_repr = [f+b for (f,b) in zip(output_states_fwd,output_states_bwd[::-1])]
		#output_states = s.transduce(bilstm_repr)
		self.output_states = [o.npvalue() for o in bilstm_repr]
		self.attention = []
		
		#attended_words = [self._attend2(w, encoded) for w in encoded]
		
		hs = [dy.tanh(W_hh*(o+w)) for o,w in zip(bilstm_repr, encoded)]

		self.output_states = [o.npvalue() for o in hs]
		pred = [dy.softmax(W*h) for h in hs]
		

		return pred


        def train(self, epochs=30):
	  start = time.time()
	  n = self.generator.get_train_size()
	  print "size of training set: ", n
          print "training..."

	  iteration = 0
	  good_a, bad_a, good_e, bad_e, good_d, bad_d = 1., 1., 1., 1., 1., 1.
	  losses = []

	  for i, batch in enumerate(self.generator.generate(is_training=True)):

		dy.renew_cg()
		if i%500 == 0:
			print "{}".format(i)
			print time.time() - start
			#gc.collect()

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

			(sent,orig_sent, output, lemmas), suffixes, data_sample = training_example
		
			#print "============"			
			#for (w,o,l) in zip(sent, output,lemmas):
			#	print w, o, l
			#print "============"

			suffixes_probs = self._predict(sent,output,lemmas, training=True)
			
			# collect loss & errors

			suffixes_encoded = [SUFFIX2I[s] for s in suffixes]
			loss = dy.scalarInput(0.)

			for w, pred_vec, true_suffix in zip(sent, suffixes_probs, suffixes_encoded):
				#if true_suffix!=2 and np.random.random()<.001:
					#print w, true_suffix, np.argmax(pred_vec.value())
				#pred = np.argmax(pred_vec)
				#if (not (true_suffix==3 and pred_vec==3)) or np.random.random()<.2:
					loss -= dy.log(pred_vec[true_suffix])
				#pass
	

			suffixes_pred = [I2SUFFIX[np.argmax(probs)] for probs in suffixes_probs]

			losses.append(loss)

		# backprop

		loss_sum = dy.esum(losses)
		#loss_sum.forward()
		loss_sum.backward()
		try:
			self.trainer.update()
		except:
			print "Bad gradient."
			self.trainer.restart()
		losses = []

		# check dev set accuracy

		if i%(2000) == 0:# and i > 0:
			self.embedding_collector.collect()
			print "evaluating accuracy on the dev set."
			if i >= 0: self.test()
			


        def test(self, train_set=False):

	   n = self.generator.get_dev_size()
	   iteration = 0

           f2 = open("attention2.txt", "w")
	   ak_good, ak_bad = 1., 1.
	   ek_good, ek_bad = 1., 1.
	   ari_good, ari_bad = 1., 1.
	   ei_good, ei_bad = 1., 1.
	   total_good, total_bad = 1., 1.
	   a_good, a_bad = 1., 1.
	   ak_total, ak_pred_total = 1., 1.
	   ek_total, ek_pred_total = 1., 1.,
	   a_total, a_pred_total = 1., 1.
	   ari_total, ari_pred_total = 1., 1.
	   ei_total, ei_pred_total = 1., 1.
	   anysuffix_total, anysuffix_pred_total = 1., 1.
	   anysuffix_good = 1.
	   attends = []

	   for i, batch in enumerate(self.generator.generate(is_training=False)):
		dy.renew_cg()

		for j, training_example in enumerate(batch):

			iteration+=1
			(sent,orig_sent, output, lemmas), suffixes, data_sample = training_example

			suffixes_probs = self._predict(sent,output,lemmas, training=True)


			suffixes_pred = [I2SUFFIX[np.argmax(probs.value())] for probs in suffixes_probs]

			for s_true, s_pred in zip(suffixes,suffixes_pred):

				if s_true == "ak":
					ak_total+=1
					if s_pred == "ak":
						ak_good+=1.
						
				if s_true == "ei":
					ei_total+=1
					if s_pred == "ei":
						ei_good+=1.

				if s_true == "ari":
					ari_total+=1
					if s_pred == "ari":
						ari_good+=1.	

				elif s_true == "ek":
					ek_total+=1
					if s_pred == "ek":
						ek_good += 1.
	
				elif s_true == "a":
					a_total+=1
					if s_pred == "a":
						a_good +=1.
				if s_true != "":
					anysuffix_total += 1
					if s_pred != "":
						anysuffix_good += 1

				if s_pred == "ak":
					ak_pred_total += 1
				elif s_pred == "ek":
					ek_pred_total += 1
				elif s_pred == "a":
					a_pred_total += 1

				elif s_pred == "ari":
					ari_pred_total += 1
				elif s_pred == "ei":
					ei_pred_total += 1
					
				if s_pred != "":
					anysuffix_pred_total += 1
				
			
			assert len(suffixes) == len(suffixes_pred)
			self.states_collector.collect(data_sample, orig_sent, suffixes, suffixes_pred, self.output_states)
			

			if i % 2000 == 0:
			   for aa in attends:
				for ii, a in enumerate(aa):
					sep = "\t" if ii != len(self.attention) - 1 else "\n"
					#f2.write(" ".join([str(s) for s in a])+sep)

			   attends = []


			if iteration%200 == 0:
				print "{}/{}".format(iteration, n)
				print "sentence seen:", " ".join(sent)
				print "orig sent: ", " ".join(orig_sent)
			 	sent_pred = [w+"-"+s+" ("+t+")" for (w,s,t) in zip(sent, suffixes_pred, suffixes)]
				print " ".join(sent_pred)
				print

				#for a,word in zip(self.attention, sent):
			       	 #att = a
				 #att = [round(a,3) for a in att]
				 #attention_and_words = [(w,a) for w, a in zip(sent, att)]
				 #most_attended = sorted(attention_and_words, key = lambda pair: -pair[1])
				
				 #print "attention for word {}:".format(word)
				 # print attention_and_words
				 #print "====="
				 #print most_attended
				 #print "========="
				print "-----------------------"

			if iteration > n:

				ak_recall = ak_good/(ak_total)
				ek_recall = ek_good/(ek_total)
				a_recall = a_good/(a_total)
				ei_recall = ei_good/ei_total
				ari_recall = ari_good/ari_total
				anysuffix_recall = anysuffix_good/anysuffix_total
				ak_precision = ak_good/(ak_pred_total)
				ek_precision = ek_good/(ek_pred_total)
				ari_precision = ari_good/ari_pred_total
				ei_precision = ei_good/ei_pred_total
				anysuffix_precision = anysuffix_good/anysuffix_pred_total
				a_precision = a_good/(a_pred_total)
				ak_f1 = 2 * (ak_recall * ak_precision)/(ak_recall + ak_precision)
				ek_f1 = 2 * (ek_recall * ek_precision)/(ek_recall + ek_precision)
				a_f1 = 2 * (a_recall * a_precision)/(a_recall + a_precision)
				ari_f1 = 2 * (ari_recall * ari_precision)/(ari_recall + ari_precision)
				ei_f1 = 2 * (ei_recall * ei_precision)/(ei_recall + ei_precision)
				anysuffix_f1 =  2 * (anysuffix_recall * anysuffix_precision)/(anysuffix_recall + anysuffix_precision)
				print "ak recall: {}; ek recall: {}".format(ak_recall, ek_recall)
				print "ak precision: {}; ek precision: {}".format(ak_precision, ek_precision)
				print "a precision: {}; a recall: {}".format(a_precision, a_recall)
				print "ari precision: {}; ari recall: {}".format(ari_precision, ari_recall)
				print "ei precision: {}; ei recall: {}".format(ei_precision, ei_recall)
				print "anysuffix precision: {}; anysuffix recall: {}".format(anysuffix_precision, anysuffix_recall)
				print "ak f1: {}; ek f1: {}; a f1: {}; ari f1: {}; ei f1: {};  anysuffix f1: {}".format(ak_f1, ek_f1, a_f1, ari_f1, ei_f1, anysuffix_f1)
				#print "NONE 10%"
						
				self.states_collector.end()
				f2.close()
				#print "ATTENTION."
				return

	




