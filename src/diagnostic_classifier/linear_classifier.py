import dynet as dy
import numpy as np
import random


class Net(object):

	def __init__(self, in_size, hid_size, out_size):

		self.in_size = in_size
		self.hid_size = hid_size
		self.out_size = out_size

		self.model = dy.Model()
		self.W_ih = self.model.add_parameters((hid_size, in_size))
		self.b = self.model.add_parameters((hid_size, 1))
		self.W_ho = self.model.add_parameters((out_size, hid_size))

		self.W_direct = self.model.add_parameters((out_size, in_size))
		self.b_direct = self.model.add_parameters((out_size, 1))

		self.trainer = dy.AdamTrainer(self.model)

	def forward(self, x):

		v = dy.vecInput(len(x))
		v.set(x)

		multi = False

		if multi:
			W_ih = dy.parameter(self.W_ih)
			b = dy.parameter(self.b)
			W_ho = dy.parameter(self.W_ho)

			h = dy.rectify(W_ih * v + b)
			o = dy.logistic(W_ho * h)

		else:
			W, b = dy.parameter(self.W_direct), dy.parameter(self.b_direct)
			o = dy.logistic(W * v + b)
		#print o.value()
		return o

	def train(self, train, dev, epochs = 200, batch_size = 64):

		dy.renew_cg()

		for I in range(epochs):
	
			loss = dy.scalarInput(0.)
			losses = []
			np.random.shuffle(train)

			if I > 0: 
				print "epoch {}. calculating accuracy...".format(I)
				self.test(dev)
			
			for i, example in enumerate(train):

				x, y, p, g = example
				y_hat = self.forward(x)
				y_val = dy.scalarInput(y)
				#loss = -dy.log(y_hat[y])
				loss = dy.squared_distance(y_val, y_hat)
				losses.append(loss)

				if i % batch_size == 0:

					loss_sum = dy.esum(losses)
					loss_sum.backward()
					self.trainer.update()
					losses = []
					dy.renew_cg()
				


	def test(self, dev):

		good, bad = 0.1, 0.1
		good_wrong, bad_wrong = 0.1, 0.1
		good_right, bad_right = 0.1, 0.1

		for i, example in enumerate(dev):

			dy.renew_cg()

			#if i % 500 == 0:
				#print "{}/{}".format(i, len(dev))

			x, y, p, g = example
			y_hat = self.forward(x)
			y_pred = 1 if y_hat.value()[0] > .5 else 0

			if y_pred == y:

				good += 1
			else:
				bad += 1

			if p != g:

				if y_pred == y:
					good_wrong +=1
				else:
					bad_wrong += 1

			else:
				
				if y_pred == y:
					good_right +=1
				else:
					bad_right += 1

		print "acc total: {}; acc wrong: {}; acc right: {}".format(good / (good + bad), good_wrong/(good_wrong + bad_wrong), good_right/(good_right + bad_right))
		print "total right: {}; total wrong: {}".format(good_right+bad_right, good_wrong+bad_wrong)
