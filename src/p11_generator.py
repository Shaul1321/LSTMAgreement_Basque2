from DataGenerator_batch import *
import random
import re

class P1Generator(DataGenerator):

 def __init__(self, data_dicts, W2I, I2W, D2I, A2I, E2I):
     DataGenerator.__init__(self, data_dicts, W2I, I2W, D2I, A2I, E2I)

 """
 def _get_analyser_output(self, output_string):
	output_as_list = filter(None, output_string.split("^"))
	sent = [o.split("/")[0] for o in output_as_list]
	analyser_output = []

	for o in output_as_list:
		word, output  = o.split("/", 1)
		word_analysis =  re.findall("<.*?>", output)
		analyser_output.append(word_analysis)

	return sent, analyser_output
 """

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
	#sent = data_sample['orig_sentence'].split(" ")


	sent, analyser_output, lemmas = self._get_analyser_output(data_sample['output'])
	prev_sent, prev_analyser_output, prev_lemmas = self._get_analyser_output(prev_data['output'])

		
	#next = next_data['orig_sentence'].split(" ")
	verb = sent[verb_index]

	sent_without_verb = sent[:verb_index]+["<verb>"]+sent[verb_index+1:]
	
	if len(ergative) == 0: ergative = ["None"]
	if len(absolutive) == 0: absolutive = ["None"]
	analyser_output[verb_index] = ["<verb>"]#+absolutive


	sent_without_verb = ["<begin>"] + sent_without_verb + ["<end>"]
	#sent_without_verb =  ["<begin>"] + prev_sent +["<end>"] +  ["<begin>"]+ sent_without_verb + ["<end>"]

	#analyser_output = ["<begin>"] +prev_analyser_output+["<end>"]+ ["<begin>"] + analyser_output + ["<end>"]
	analyser_output = ["<begin>"] + analyser_output + ["<end>"]
	lemmas = ["<begin>"] + lemmas + ["<end>"]
	#lemmas = ["<begin>"] +prev_lemmas+["<end>"]+["<begin>"]+ lemmas + ["<end>"]

	assert len(analyser_output) == len(sent_without_verb)
	assert len(analyser_output) == len(lemmas)

	#return sent_without_verb, (ergative, absolutive, dative)
	#print (ergative_encoded, absolutive_encoded, dative_encoded) 
	#print sent
	sent_without_verb = [w[:-2] if (w[-2:]=="ak" and len(w)>2) else w for w in sent_without_verb]
	sent_without_verb = [w[:-2] if (w[-2:]=="ek" and len(w)>2) else w for w in sent_without_verb]

	return (sent_without_verb, analyser_output, lemmas), (ergative_encoded, absolutive_encoded, dative_encoded) 


