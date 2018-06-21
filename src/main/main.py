part1 = False

import dynet as dy
from utils2 import *
#from RNN import *
if part1: 
	from RNN_attention7 import * 
	from p1_generator2 import *
else: 
	from RNN_attention8 import *
	from p3_generator import *
from embedding_collector import *
from states_collector import *
import matplotlib.pyplot as plt
import numpy as np
import Encoder

dyparams = dy.DynetParams()
dyparams.set_autobatch(True)

def plot(model):


    l = len(model.subj_dev)
    iterations = np.arange(1500, 1500*(l+1), 1500)
    #iterations/=100

    labels = ["ergative, dev", "absolutive, dev", "ergative, train", "absolutive, train"]

    fig, ax = plt.subplots()

    ax.plot(iterations, model.subj_dev, label = labels[0])
    ax.plot(iterations, model.obj_dev, label = labels[1])
    ax.plot(iterations, model.subj_train, label = labels[2])
    ax.plot(iterations, model.obj_train, label = labels[3])

    legend = ax.legend(shadow=True) 
    plt.grid(True)
    plt.xlabel('number of mini-batches')
    plt.ylabel('accuracy')
    plt.yticks(np.arange(0, 1, 0.05))
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
	
	if not part1: 
		dg = P3Generator(SENTENCES, W2I, I2W, D2I, A2I, E2I, prepared_dev = DEV_SENTENCES)
		embedding_filename = "EMBEDDINGS.txt" 
	else:
		dg = P1Generator(SENTENCES, W2I, I2W, D2I, A2I, E2I, prepared_dev = DEV_SENTENCES)
		embedding_filename = "EMBEDDINGS1.TXT"
		
	in_size, a_out_size, d_out_size, e_out_size = len(W2I), len(A2I), len(D2I), len(E2I)
	model = dy.Model()

	#encoder = ComplexEncoder(in_size, model, W2I, C2I)
	#encoder = LSTMEncoder(len(C2I), model, C2I)
	#encoder = EmbeddingEncoder(in_size, model, W2I)
	#encoder = Encoder.SubwordEncoder(in_size, model, W2I, SUFFIX2I, PREFIX2I, OUTPUT2IND)
	encoder = Encoder.CompleteSubwordEncoder(model, W2I, NGRAM2I, OUTPUT2IND, LEMMA2I, SUFFIX2I)
	embedding_collector = Collector(encoder, "VOC_WITH_LEMMAS.txt", embedding_filename)
	states_collector = StatesCollector("preds2.txt")
	if not part1: 
		rnn = RNN(in_size, 64, (a_out_size, e_out_size, d_out_size), dg,  I2A, I2E, I2D, I2W, model, encoder, embedding_collector, states_collector)
	else:
		rnn = RNN(in_size, 64, (a_out_size, e_out_size, d_out_size), dg,  I2A, I2E, I2D, I2W, model, encoder, embedding_collector)
	
	#rnn = RNN(in_size, 64, 2, dg,  I2A, I2E, I2D, I2W, model, encoder)

	rnn.train()
	plot(rnn)

