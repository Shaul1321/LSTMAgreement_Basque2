from DataGenerator_batch import *
import random
import re

class P3Generator(DataGenerator):

 def __init__(self, data_dicts, W2I, I2W, D2I, A2I, E2I, prepared_dev = None):
     DataGenerator.__init__(self, data_dicts, W2I, I2W, D2I, A2I, E2I, prepared_dev)

 def _remove_suffix(self, sent, out, lemmas):

	suffixes = []
	sent_without_suffixes = []

	for w,o,lemma in zip(sent, out, lemmas):
		#lemma = lemma if not lemma.endswith("$") else lemma[:-1]
		lemma = lemma.strip()
		known_word = lemma != "*" + w +"$"
		#print w,o,lemma
		
		if "<vbsint>" in o or "<vblex>" in o or len(w)<=2:

			suffixes.append("")
			sent_without_suffixes.append(w)
		else:

			if w.endswith("ak") or w.endswith("ek") or w.endswith("ei") and len(w) > 3:
				suffixes.append(w[-2:]) 
				sent_without_suffixes.append(w[:-2]+"a")

			elif w.endswith("a") and (not lemma.endswith("a") and known_word):
				suffixes.append("a")
				sent_without_suffixes.append(w[:-1]+"a")
				
			elif (w.endswith("ari") and not lemma.endswith("ari") and len(w) > 3):
				suffixes.append("ari")
				sent_without_suffixes.append(w[:-3]+"a")
			else:
				suffixes.append("")
				sent_without_suffixes.append(w)

	return suffixes, sent_without_suffixes
		
	

 def create_example(self, data_sample, prev_data, next_data, shuffle=False):
 

	verb_index = int(data_sample['verb_index'])

	sent, analyser_output, lemmas = self._get_analyser_output(data_sample['output'])
	prev_sent, prev_analyser_output, prev_lemmas = self._get_analyser_output(prev_data['output'])

	#for i, (w,o) in enumerate(zip(sent, analyser_output)):
	#	if "<vbsint>" in o:
	#		sent[i] = "<verb>"
	analyser_output = ["<begin>"] + ["None"]*len(analyser_output) + ["<end>"]

			
	#analyser_output[verb_index+1]=["<verb>"]
	lemmas = ["<begin>"] + lemmas + ["<end>"]
	
	#print sent
	
	sent = ["<begin>"] + sent + ["<end>"]
	
	orig_sent = sent[:]
	for i, o in enumerate(orig_sent):
		orig_sent[i] = orig_sent[i].replace(" ", "-")

	out = ["<begin>"] + filter(None, data_sample['output'].split("^")) + ["<end>"]

	suffixes, sent = self._remove_suffix(sent, out, lemmas)
		
	assert len(suffixes) == len(sent) == len(analyser_output)

	if shuffle:
		
		zipped = zip(sent, analyser_output, lemmas, suffixes)
		random.shuffle(zipped)
		(sent, analyser_output, lemmas, suffixes) = zip(*zipped)
		
	return (sent,orig_sent, analyser_output, lemmas), (suffixes) 


