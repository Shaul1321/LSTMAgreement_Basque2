from DataGenerator_batch import *
import random
import numpy as np

class P2Generator(DataGenerator):

 def __init__(self, data_dicts, W2I, I2W, D2I, A2I, E2I):
     DataGenerator.__init__(self, data_dicts, W2I, I2W, D2I, A2I, E2I)
     self._create_probs()

 def _create_probs(self):

	verb_forms = []

	with open("VERB_FORMS.txt") as f:
		lines = f.readlines()
	lines = [line.strip() for line in lines]
	lines = lines[:100]
	for line in lines:
		verb_form, lemma, freq = line.split("\t")
		freq = int(freq)
		verb_forms.append((verb_form, lemma, freq))

	verbs, freqs = [v for (v,l, f) in verb_forms], [f for (v,l, f) in verb_forms]
	probs = [f/(1.*sum(freqs)) for f in freqs]

	verb_forms_ukan = filter(lambda (v,l,f): l=="ukan", verb_forms)
	verb_forms_izan  = filter(lambda (v,l,f): l=="izan", verb_forms)
	verbs_ukan, freqs_ukan = [v for (v,l, f) in verb_forms_ukan], [f for (v,l, f) in verb_forms_ukan]
	verbs_izan, freqs_izan = [v for (v,l, f) in verb_forms_izan], [f for (v,l, f) in verb_forms_izan]
	probs_ukan = [f/(1.*sum(freqs_ukan)) for f in freqs_ukan]
	probs_izan = [f/(1.*sum(freqs_izan)) for f in freqs_izan]

	self.verbs = verbs
	self.probs = probs
	self.probs_ukan = verbs_ukan
	self.verbs_ukan = freqs_ukan
	self.probs_izan = probs_izan
	self.verbs_izan = verbs_izan


 def _sample_random_verb(self, current_verb):

	verb = current_verb
	i = 0
	while current_verb==verb:
		i+=1
		ind = np.random.choice(len(self.verbs), 1, p=self.probs)[0]
		verb = self.verbs[ind]

	return verb
 
 def create_example(self, data_sample, prev_data, next_data):
        """
        creates a training example - the sentence without the verb, alongisde its agreements

 	returns:
        	sent_without_verb, (ergative_encoded, absolutive_encoded, dative_encoded) 

    	sent_without_verb: the sentence, as a list of string words, with the vebr ommited
	(ergative_encoded, absolutive_encoded, dative_encoded) - encoding of the coorresponding 	agreements.
	
        """


	verb_index = int(data_sample['verb_index'])
	verb_output = data_sample['verb_output']


	dative = [d for d in self.D2I if d in verb_output]
	absolutive = [a for a in self.A2I if a in verb_output]
	ergative = [e for e in self.E2I if e in verb_output]	
	
	#assert len(dative)<=1
	#assert len(absolutive)<=1
	#assert len(ergative)<=1

	ergative_encoded = self.E2I["None"] if len(ergative) == 0 else self.E2I[ergative[0]] 
	absolutive_encoded = self.A2I["None"] if len(absolutive)==0 else self.A2I[absolutive[0]]
	dative_encoded = self.D2I["None"] if len(dative) == 0 else self.D2I[dative[0]]

	sent, analyser_output, lemmas = self._get_analyser_output(data_sample['output'])
	prev_sent, prev_analyser_output, prev_lemmas = self._get_analyser_output(prev_data['output'])

		
	#next = next_data['orig_sentence'].split(" ")
	verb = sent[verb_index]
	lemma = verb_output.split("/")[0]

	analyser_output[verb_index] = ["<verb>"]#+absolutive

	negative_sample = False

	if np.random.random()<.5:

		negative_sample = True
		sent[verb_index] = self._sample_random_verb(verb)


	sent =  ["<begin>"]+ sent + ["<end>"]
	analyser_output = ["<begin>"] + analyser_output + ["<end>"]
	#sent =  ["<begin>"] + prev_sent +["<end>"] +  ["<begin>"]+ sent + ["<end>"]
	#analyser_output = ["<begin>"] +prev_analyser_output+["<end>"]+ ["<begin>"] + analyser_output + ["<end>"]
	assert len(analyser_output) == len(sent)

	#return sent_without_verb, (ergative, absolutive, dative)
	return (sent, analyser_output, ["<unk>"]*len(sent)), (0 if negative_sample else 1,  (absolutive_encoded, ergative_encoded,dative_encoded))



