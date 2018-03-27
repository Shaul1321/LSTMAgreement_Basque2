import csv
from collections import Counter
import re

"""this code is responsible for the creation the dataset csv file, based on all_outputs.txt
(the output of the morphological analyser for all sentences) and the text file that contains
all the training sentences."""

duplicate_counter = 0.
ambiguous = set()
labels = Counter()

def read_data(lower=False, remove_suffixes=False):

	with open("all_outputs.txt", "r") as f:
		outputs = f.readlines()
	outputs = [line.strip() for line in outputs]

	with open("all_sentences.txt", "r") as f:
		sentences = f.readlines() 

	sentences = [line.strip() for line in sentences]
	
	if lower:
		outputs = [o.lower() for o in outputs]
		sentences = [s.lower() for s in sentences]

	if remove_suffixes:
		sentences = [s.replace("ak ", " ") for s in sentences]
		outputs = [o.replace("ak ", " ") for o in outputs]

	#sentences = sentences[:len(outputs)]

	assert len(sentences)==len(outputs)
	
	return zip(sentences, outputs)


def create_csv(fname, sents_and_outputs, condition, duplicate_handler, unambiguous = False, one_verb=False):

	errors = 0.
	verbs_count = 0.
	
	with open(fname, 'wb') as f:
		
		wr = csv.writer(f, quoting=csv.QUOTE_ALL)

		print len(sents_and_outputs)
		sents_and_outputs = filter(lambda (sent,out): "<vbsint>" in out, sents_and_outputs)
		print len(sents_and_outputs)
		sents_and_outputs = filter(lambda (sent,out): "NONE" not in out, sents_and_outputs)
		print len(sents_and_outputs)
			

		if unambiguous:
			sents_and_outputs = filter(lambda (sent,out): "ak " not in sent, sents_and_outputs)
		print len(sents_and_outputs)

		if one_verb:
			sents_and_outputs = filter(lambda (sent,out): out.count("<vbsint>")==1, sents_and_outputs)
		print len(sents_and_outputs)

		for ind, (sentence, output) in enumerate(sents_and_outputs):

			#if ind%1000 == 0: print "{}/{}".format(ind, len(sents_and_outputs))

			output_splitted = output.split("^")[1:]
			output_splitted = [o.split("$")[0] for o in output_splitted]
			sentence_splitted = sentence.split(" ")

			verbs_count += output.count("<vbsint>")

			for i, w_output in enumerate(output_splitted): 

				if "<vbsint>" in w_output and condition(w_output):

					ls = [sentence, output] #, str(i), w_output]
					word = w_output.split("/")[0]
					oooo = duplicate_handler(w_output)
					try:
						#index = sentence_splitted.index(word)
						index = i
						#assert sentence_splitted[index]==word
						ls+=[str(index), duplicate_handler(w_output)]
						wr.writerow(ls)
						
						global labels #collect label statistics
						word_analysis =  re.findall("<.*?>", ls[-1])
						labels.update(word_analysis)	

					except Exception as e:
						print sentence_splitted
						print "--------------"
						print [o.split("/")[0] for o in output_splitted]
						print oooo
						print e
						print "==============================="
						errors+=1
					
					break
	l = len(sents_and_outputs)
	print "number of sentences: {}".format(l)
	print "average number of verbs per sentence: {}".format(verbs_count/l)
	print "error rate: {}".format(errors/l)
	print "duplicate rate: {}".format(duplicate_counter/l)
	print "there are {} ambiguous verbs: {}".format(len(ambiguous), ambiguous)




def duplicate_handler(output):

	"""
	handles cases where the output of the morphologiacl analyser contains more than one meaning.
	output: a string, the output of the analyser for a given word.

	if one meaning is one of the two auxilary verb, choose it (as it is assumed to be more
	common). sort out familarity forms not common in Wikiepdia dataset.
	
	"""
	if output.count("/")==1: return output	
	#print output
	meanings = output.split("/")
	word = meanings[0]

	global ambiguous
	ambiguous.add(word)

	global duplicate_counter
	duplicate_counter+=1

	out = meanings[1]
	contains_auxiliary = "ukan" in output or "izan" in output or "egon" in output

	found = False
	for meaning in meanings[1:]:
		if (("ukan" in meaning or "izan" in meaning or "egon" in meaning) and not "_HI" in meaning and not "<NO>" in meaning) or (not contains_auxiliary and not "<NO>" in meaning):
			out = meaning
			found = True
			break


	if not found:

		pass

	return word+"/"+out
		

if __name__ == '__main__':
	sents_and_outputs = read_data()

	all_verbs =  lambda out: True
	present_verbs = lambda out: "<pri>" in out
 	past_verbs  = lambda out: "<pii>" in out or "<pp>" in out

	create_csv("data.csv", sents_and_outputs, condition=present_verbs, duplicate_handler=duplicate_handler)
	print labels









