from linear_classifier import *
from utils import *
import dynet as dy

if __name__ == '__main__':

	model = Net(128, 128, 1)
	print "train size: {}; dev size: {}".format(len(TRAIN), len(DEV))
	model.train(TRAIN, DEV)
	print dy.parameter(model.W_direct).npvalue()
