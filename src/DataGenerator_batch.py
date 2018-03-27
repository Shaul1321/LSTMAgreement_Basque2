import random
import re

"""
an abstract class for creating training example, based on the training corpus.
derived class override create_example method according to the requierments. 
"""

bad_words = ["familiak", "pertsonak", "pertsona", "inaktiboetatik", "inaktiboak", "erretiraturik", "etxetan", "etxebizitza","etxek", "udalerri", "metropolitanoan", "diagrama", "etxeak", "apartamentuak", "eskola", "kilometrotako", "kilometroko", "biztanleriak", "komertzialetatik", "publikoetatik","probintzian", "ospitale", "konderrian", "kokatua", "kilometro", "osasunekipamendu", "zentsuaren", "biztanle", "publikoak", "komertzioetatik", "administratiboki"] #words associated with bot-generated articles.


class DataGenerator(object):

 def __init__(self, data_dicts, W2I, I2W, D2I, A2I, E2I):
   print "data generator constructed."
   self.data_dicts = data_dicts
   self.W2I = W2I
   self.I2W = I2W

   self.E2I = E2I
   self.A2I = A2I
   self.D2I = D2I

   n = len(self.data_dicts)
   random.shuffle(self.data_dicts)
  
   self.train = self.data_dicts[int(0.0*n):int(0.90*n)]
   self.test  = self.data_dicts[int(0.9*n):int(1*n)]

   #self.train = self.data_dicts[:int(0.35*n)]
   #self.test  = self.data_dicts[int(0.75*n):int(0.751*n)]


   print "len of dev set before: ", len(self.test)
   self._clear_dev_set() #try to remove bot-generated articles from the dev set.
   print "len of dev set after: ", len(self.test)
   print "train set size: {}; total size: {}".format(len(self.train), len(self.data_dicts) ) 

 def _clear_dev_set(self):
	deleted = set()
	for i, data_dict in enumerate(self.test[:]):
		if i%50 == 0:
			print "{}/{}".format(i, len(self.test))

		for w in bad_words:
			if w in data_dict['orig_sentence']:
				deleted.add(i)# data_dict
				break
	self.test = [self.test[i] for i in range(len(self.test)) if i not in deleted]

 def get_train_size(self):
	return len(self.train)

 def get_dev_size(self):
	return len(self.test)

 def create_example(self, data_sample):
 	raise NotImplementedError

 def _get_analyser_output(self, output_string):

	output_as_list = filter(None, output_string.split("^"))
	sent = [o.split("/")[0] for o in output_as_list]

	analyser_output = []
	lemmas = []

	for o in output_as_list:
		word, output  = o.split("/", 1)
		word_analysis =  re.findall("<.*?>", output)
		lemma = output.split("<")[0]
		lemmas.append(lemma)
		analyser_output.append(word_analysis)

	return sent, analyser_output, lemmas


 def generate(self, is_training):

   """
 	a template method for generating a training example. the abstract method create_example
 	is implemented in the drived class, according to the requierments in each part.

	is_training - a boolean flag indicating training/prediction mode.
   """

   i = 0

   while True: 
      i+=1
      if i%3000 == 0: 
	#random.shuffle(self.train)
	#random.shuffle(self.test)
	pass

      batch_size = 16
      batch = []

      for k in range(batch_size):
      	if is_training:
      		i = random.choice(range(1, len(self.train)-1))
		source = self.train
      	else:
		i = random.choice(range(1, len(self.test)-1))
		source = self.test

     	data, prev_data, next_data = source[i], source[i-1], source[i+1]

     	x_encoded, y_encoded = self.create_example(data, prev_data, next_data)
	batch.append((x_encoded, y_encoded, data))

      yield batch

