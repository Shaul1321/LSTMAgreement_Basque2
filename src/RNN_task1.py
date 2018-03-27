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
ATTENTION_HIDDENSIZE = 32
LSTM_HIDDENSIZE = 128
DROPOUT_RATE = 0.0

ABSO = {"<NR_HK>": "absolutive: pl3", "<NR_HU>": "absolutive: sg3", "<NR_HI>": "absolutive; ??",
"<NR_GU>": "absolutive :1pl", "<NR_NI>":"absolutive: 1sg", "<NR_ZU>": "absolutive: 2sg", "<NR_ZK>": "absolutive: 2pl", "None": "absolutive: None"}

ERG = {"<NK_HU>": "ergative: sg3", "<NK_HK>": "ergative: pl3", "<NK_HI>": "ergative: ??",
"<NK_GU>": "ergative: 1pl", "<NK_NI>": "ergative: 1sg", "<NK_ZU>": "ergative: 2sg", "<NK_ZK>": "ergative: 2pl", "None": "ergative: None"}

DAT = {"<NI_HU>": "dative: sg3", "<NI_HK>": "dative: pl3", "<NI_HI>": "dative: ??",
"<NI_GU>": "dative: 1pl", "<NI_NI>": "dative: 1sg", "<NI_ZU>": "dative: 2sg", "<NI_ZK>": "dative: 2pl", "None": "dative: None"}


def write_errors(good, bad, good_abs, bad_abs, good_erg, bad_erg, bad_erg2, bad_abs2,error_count, total):
 
 f = open("error_rates7_count.txt", "w")
 items = sorted(error_count.items(), key = lambda (key, val): -val)
 for (key, val) in items:
	f.write(str(key).replace(" ", "")+"\t"+str(val)+"\n")
 f.close()
 

 f = open("error_rates6.txt", "w")
 items = sorted(total.items(), key = lambda (key, val): -val)
 for (key, val) in items:
	good_count = good[key] 
	bad_count =  bad[key]
	f.write(str(key).replace(" ", "")+"\t"+str(good_count)+"\t"+str(bad_count)+"\n")
 f.close()

 f = open("error_rates_abs6.txt", "w")
 keys = list(set(good_abs.keys() + bad_abs.keys()))
 keys = sorted(keys, key = lambda k: -(good_abs[k]+bad_abs[k]))
 for k in keys:
	good_count = good_abs[k] 
	bad_count =  bad_abs[k] 
	f.write(str(k).replace(" ", "")+"\t"+str(good_count)+"\t"+str(bad_count)+"\n")

 f.close()


 f = open("error_rates_erg6.txt", "w")
 keys = list(set(good_erg.keys() + bad_erg.keys()))
 keys = sorted(keys, key = lambda k: -(good_erg[k]+bad_erg[k]))
 for k in keys:
	good_count = good_erg[k] 
	bad_count =  bad_erg[k] 
	f.write(str(k).replace(" ", "")+"\t"+str(good_count)+"\t"+str(bad_count)+"\n")

 f.close()


 f = open("error_rates_abs6.2.txt", "w")
 keys = list(set(good_abs.keys() + bad_abs2.keys()))
 keys = sorted(keys, key = lambda k: -(good_abs[k]+bad_abs2[k]))
 for k in keys:
	good_count = good_abs[k] 
	bad_count =  bad_abs2[k] 
	f.write(str(k).replace(" ", "")+"\t"+str(good_count)+"\t"+str(bad_count)+"\n")

 f.close()


 f = open("error_rates_erg6.2.txt", "w")
 keys = list(set(good_erg.keys() + bad_erg2.keys()))
 keys = sorted(keys, key = lambda k: -(good_erg[k]+bad_erg2[k]))
 for k in keys:
	good_count = good_erg[k] 
	bad_count =  bad_erg2[k] 
	f.write(str(k).replace(" ", "")+"\t"+str(good_count)+"\t"+str(bad_count)+"\n")

 f.close()



class RNN(object):

	def __init__(self, in_size, hid_size, (a_out, e_out, d_out), dataGenerator, I2A, I2E, I2D, I2W, model, encoder):

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

		self.W_attention1 = self.model.add_parameters((ATTENTION_HIDDENSIZE, EMBEDDING_SIZE+LSTM_HIDDENSIZE))
		self.b_attention1 = self.model.add_parameters((ATTENTION_HIDDENSIZE, 1))
		self.W_attention2 = self.model.add_parameters((1, ATTENTION_HIDDENSIZE))
		self.W_attention_simple = self.model.add_parameters((1, EMBEDDING_SIZE+LSTM_HIDDENSIZE))

		hid = 128

		self.W_ha = self.model.add_parameters((self.a_out, hid))
		self.W_he = self.model.add_parameters((self.e_out, hid))
		self.W_hd = self.model.add_parameters((self.d_out, hid))
		self.W_hh = self.model.add_parameters((hid, LSTM_HIDDENSIZE))
		self.b_hh = self.model.add_parameters((hid, 1))
		self.W_hh2 = self.model.add_parameters((hid, hid))
		self.W_hh3 = self.model.add_parameters((hid, hid))
		self.W_hh4 = self.model.add_parameters((hid, hid))

		self.LSTM_first = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, LSTM_HIDDENSIZE, self.model)
		self.LSTM_second = dy.LSTMBuilder(NUM_LAYERS, LSTM_HIDDENSIZE+EMBEDDING_SIZE, LSTM_HIDDENSIZE, self.model)

		#self.trainer = dy.AdagradTrainer(self.model)
		self.trainer = dy.AdamTrainer(self.model)
		#self.trainer.learning_rate = 0.05
        

	def _attend(self, encoded_sent, states, training=True):

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

		h = [dy.rectify(W_attention1*dy.concatenate([w,s])) for w,s in zip(encoded_sent,states)]
		drop = 0.#DROPOUT_RATE if not training else 0.
		#h = [dy.dropout(dy.rectify(W_attention1*s + b_attention1), drop) for s in states]
		#if training: h = dy.dropout(h, DROPOUT_RATE)
		#weights = dy.concatenate([W_attention2*h_elem if h.index(h_elem)!=len(h)-1 else dy.scalarInput(0.) for h_elem in h]) to force the network not attend the <verb> token

		weights = dy.concatenate([W_attention2*h_elem for h_elem in h])
		weights =  dy.softmax(weights)
	

		#assert len(weights.npvalue())==len(encoded_sent)

		return weights


	def _attend2(self, encoded_sent, states, training=True):

		w = dy.parameter(self.W_attention_simple)
		scores = dy.concatenate([w*dy.concatenate([e,s]) for e,s in zip(encoded_sent, states)])
		weights = dy.softmax(scores)
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

		W_ha = dy.parameter(self.W_ha)
		W_he = dy.parameter(self.W_he)
		W_hd = dy.parameter(self.W_hd)

		W_hh = dy.parameter(self.W_hh)
		b_hh = dy.parameter(self.b_hh)
		W_hh2 = dy.parameter(self.W_hh2)
		W_hh3 = dy.parameter(self.W_hh3)
		W_hh4 = dy.parameter(self.W_hh4)

		s = self.LSTM_first.initial_state()

		# encode sentence & pass through biLstm

		encoded = [self.encoder.encode(w,o,l) for (w,o,l) in zip(sentence,output,lemmas)]

		#if training: encoded = [dy.dropout(e, DROPOUT_RATE) for e in encoded]

		output_state = s.transduce(encoded)[-1]
		#encoded = encoded[sentence.index("<end>")+1:]
		encoded_with_statevector = [dy.concatenate([e,output_state]) for e in encoded]
		ss = self.LSTM_second.initial_state()
		output_states = ss.transduce(encoded_with_statevector)

		# attend over bilstm states

		weights = self._attend2(encoded, output_states, training=training)
		self.attention = weights
		weighted_states = dy.esum([o*w for o,w in zip(output_states, weights)])

		h = dy.rectify(W_hh * weighted_states + b_hh)
		if training: h = dy.dropout(h, DROPOUT_RATE)
		h = dy.rectify(W_hh2 * h)
		if training: h = dy.dropout(h, DROPOUT_RATE)
		#h = dy.rectify(W_hh3 * h)
		#h = dy.rectify(W_hh4 * h)

		# predict absolutive, ergative and dative agreements.

		a_pred = dy.softmax(W_ha * h)
		e_pred = dy.softmax(W_he * h)
		d_pred = dy.softmax(W_hd * h)

		return (a_pred, e_pred, d_pred)

	def encode(self, sentence):

		"""encode the sentence words with the encoder"""

		return [self.encoder.encode(w) for w in sentence]

        def train(self, epochs=30):

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

			(sent, output, lemmas), (e_true,a_true,d_true), data_sample = training_example
			#print "============"			
			#for (w,o,l) in zip(sent, output,lemmas):
			#	print w, o, l
			#print "============"

			a_pred, e_pred, d_pred = self._predict(sent,output,lemmas, training=True)

			# collect loss & errors

			loss = -dy.log(a_pred[a_true]) + -dy.log(e_pred[e_true]) + -dy.log(d_pred[d_true])
			#loss = -dy.log(e_pred[e_true])
			#print sent
			#print a_pred
			#print a_pred.value()
			#print e_pred.value()
			#print d_pred.value()
			#print "-----------"
			a_pred, e_pred, d_pred = np.argmax(a_pred.npvalue()),  np.argmax(e_pred.npvalue()),  np.argmax(d_pred.npvalue())

			if a_pred==a_true:
				good_a+=1
			else:
				bad_a+=1
			if e_pred==e_true:
				good_e+=1
			else:
				bad_e+=1

			losses.append(loss)

		# backprop

		loss_sum = dy.esum(losses)
		loss_sum.forward()
		loss_sum.backward()
		self.trainer.update()
		losses = []

		# check dev set accuracy

		if i%(2500) == 0:#and i>0:

			print "iteration {} / {}".format(iteration, n)
			print "Calculating accuracy on dev set."
			self.test()
			print "train accuracy: a: {}; e: {}.".format((good_a/(good_a+bad_a)), good_e/(good_e+bad_e))
			self.obj_train.append(good_a/(good_a+bad_a))
			self.subj_train.append(good_e/(good_e+bad_e))
			good_a, bad_a, good_e, bad_e = 1., 1., 1., 1.
			if i >=30000: return


        def test(self, train_set=False):

	   good_e,bad_e, good_a, bad_a, good_d, bad_d = 0., 0., 0, 0., 0., 0.
	   true_labels = Counter()
	   good_preds, bad_preds = Counter(), Counter()
	   good_abs, bad_abs = Counter(), Counter()
	   good_erg, bad_erg = Counter(), Counter()
	   error_count = Counter()
	   dative_total, dative_correct = 0.1, 0.1
	   bad_abs2 = Counter()
	   bad_erg2 = Counter()

	   labels = Counter()
	   n = self.generator.get_dev_size()
	   iteration = 0
	   for i, batch in enumerate(self.generator.generate(is_training=False)):
		dy.renew_cg()

		for j, training_example in enumerate(batch):

			iteration+=1
			(sent, output, lemmas), (e_true, a_true, d_true), data_sample = training_example

			a_hat, e_hat, d_hat = self._predict(sent,output,lemmas, training=False)
			#loss = -dy.log(a_hat[a_true]) + -dy.log(e_hat[e_true])# + -dy.log(d_hat[d_true])

			a_pred, e_pred, d_pred = np.argmax(a_hat.npvalue()),  np.argmax(e_hat.npvalue()),  np.argmax(d_hat.npvalue())
	
			a,e,d = ABSO[self.I2A[a_true]], ERG[self.I2E[e_true]], DAT[self.I2D[d_true]]
			labels[(a,e,d)]+=1

			if iteration%2000==0:
				print "{}/{}".format(iteration, n)
				#print "predicted: {}, {}".format(ABSO[self.I2A[a_pred]], ERG[self.I2E[e_pred]])#, self.D2I[d_pred]
				print "predicted: {}, {}, {}".format(ABSO[self.I2A[a_pred]], ERG[self.I2E[e_pred]], DAT[self.I2D[d_pred]])
				print "true: {}, {}, {}".format(ABSO[self.I2A[a_true]], ERG[self.I2E[e_true]], DAT[self.I2D[d_true]])
				print "success:", a_true==a_pred and e_true==e_pred and d_true==d_pred

				verb_index = sent.index("<verb>")
				print "verb index:", verb_index
				print "verb output:", data_sample['verb_output']
				print "orig sentence:", data_sample['orig_sentence']
				print "sentence as string: ", " ".join(sent)
				
				att = self.attention.npvalue()
				att = [round(a,3) for a in att]
				attention_and_words = [(w,a) for w, a in zip(sent, att)]
				most_attended = sorted(attention_and_words, key = lambda pair: -pair[1])
				
				print "attention:"
				print attention_and_words
				print "====="
				print most_attended
				print "============================="

			if a_pred==a_true:
				good_a+=1
				good_abs[a]+=1
			else:
				bad_a+=1
				bad_abs[a]+=1
				bad_abs2[ABSO[self.I2A[a_pred]]]+=1

			if e_pred==e_true:
				good_e+=1
				good_erg[e]+=1
			else:
				bad_e+=1
				bad_erg[e]+=1
				bad_erg2[ERG[self.I2E[e_pred]]]+=1

			if d!="dative: None":
				dative_total+=1
				if d_pred==d_true:
					dative_correct+=1

			if a_pred==a_true and e_pred==e_true:
				good_preds[(a,e,d)]+=1
			else:
				bad_preds[(a,e,d)]+=1
				error_count[(a,e,d),(ABSO[self.I2A[a_pred]], ERG[self.I2E[e_pred]],DAT[self.I2D[d_true]])]+=1

			true_labels[ABSO[self.I2A[a_true]]]+=1
			true_labels[ERG[self.I2E[e_true]]]+=1

			if not (d_pred==d_true and DAT[self.I2D[d_true]]=="None"):
				if d_pred==d_true:
					good_d+=1					
				else:
					bad_d+=1
	

			
			#if d_pred==d_true:
			#	good_d+=1
			#else:
			#	bad_d+=1


			if iteration > n:

	   			print "accuracy: e: {}; a: {}; d: {}; total: {}; true_labels: {}".format(good_e/(good_e+bad_e), good_a/(good_a+bad_a),good_d/(good_d+bad_d), (good_a+good_e+good_d)/(good_a+good_e+bad_a+bad_e+good_d+bad_d), true_labels)
				self.obj_dev.append(good_a/(good_a+bad_a))
				self.subj_dev.append(good_e/(good_e+bad_e))
				print "recording error counts. present"
				print "good preds: {}; bad preds: {}".format
				write_errors(good_preds, bad_preds, good_abs, bad_abs, good_erg, bad_erg, bad_erg2, bad_abs2, error_count)
				print "dative recall: {}".format(dative_correct/dative_total)
				print "dative total: {}".format(dative_total)
				return

	




