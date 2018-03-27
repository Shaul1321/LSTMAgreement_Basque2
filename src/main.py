import dynet as dy
from utils import *
print "done importing util."

from p11_generator import *
#from RNN import *
from RNN_task1 import *
print "done importing rnn."
import matplotlib.pyplot as plt
import numpy as np
print "done importing numpy etc."
import Encoder
print "done importing encoder."

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

	dg = P1Generator(SENTENCES, W2I, I2W, D2I, A2I, E2I)
	in_size, a_out_size, d_out_size, e_out_size = len(W2I), len(A2I), len(D2I), len(E2I)
	model = dy.Model()

	#encoder = ComplexEncoder(in_size, model, W2I, C2I)
	#encoder = LSTMEncoder(len(C2I), model, C2I)
	#encoder = EmbeddingEncoder(in_size, model, W2I)
	#encoder = Encoder.SubwordEncoder(in_size, model, W2I, SUFFIX2I, PREFIX2I, OUTPUT2IND)
	encoder = Encoder.CompleteSubwordEncoder(model, W2I, NGRAM2I, OUTPUT2IND, LEMMA2I)

	rnn = RNN(in_size, 64, (a_out_size, e_out_size, d_out_size), dg,  I2A, I2E, I2D, I2W, model, encoder)
	#rnn = RNN(in_size, 64, 2, dg,  I2A, I2E, I2D, I2W, model, encoder)

	rnn.train()
	plot(rnn)

