import csv
from collections import Counter
import re

"""this code is responsible for create the dataset csv file, based on all_outputs.txt
(the output of the morphological analyser for all sentences) and the text file that contains
all the training sentences."""

duplicate_counter = 0.
ambiguous = set()
labels = Counter()

def count_verbs(output):
	output_splitted = output.split("^")
	count = 0

	for o in output_splitted:
		
		if "<vbsint>" in o:
			count += 1
	return count

def read_data(lower=False, remove_suffixes=False):

	with open("all_outputs.txt", "r") as f:
		outputs = f.readlines()
	outputs = [line.strip() for line in outputs]

	with open("raw_sentneces.txt", "r") as f:
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


#def remove_suffixes (

def create_csv(fname, sents_and_outputs, condition, duplicate_handler, unambiguous = False, one_verb=False, filter_lexical=False, filter_no_ek=False):

	errors = 0.
	verbs_count = 0.
	cases_count = 0.
	both = Counter()
	contains_ek = 0.

	with open(fname, 'wb') as f:
		
		wr = csv.writer(f, quoting=csv.QUOTE_ALL)

		print len(sents_and_outputs)
		sents_and_outputs = filter(lambda (sent,out): "<vbsint>" in out, sents_and_outputs)
		print len(sents_and_outputs)
		sents_and_outputs = filter(lambda (sent,out): "NONE" not in out, sents_and_outputs)
		print len(sents_and_outputs)
		
		if filter_no_ek:

			sents_and_outputs = filter(lambda (sent,out): "ek " in sent, sents_and_outputs)
			#print "number of sentences with ek: ", len(sents_and_outputs)
			for i in range(100):
				print sents_and_outputs[i][0]
				print

		if filter_lexical:
			sents_and_outputs = filter(lambda (sent,out): "<vblex>" not in out, sents_and_outputs)
			print len(sents_and_outputs)
			

		if one_verb:
			sents_and_outputs = filter(lambda (sent,out): count_verbs(out) == 1, sents_and_outputs)
		"""
		without_cases = filter(lambda (sent,out): "ak " not in sent and "ek " not in sent and "a " not in sent, sents_and_outputs)
		without_ekak = filter(lambda (sent,out): "ak " not in sent and "ek " not in sent, sents_and_outputs)
		print "number of sentences without case marks: {}; percentage: {}".format(len(without_cases), (1.*len(without_cases)/len(sents_and_outputs)))
		c=0.
		for (s,o) in sents_and_outputs:
		 #if o.count("<vbsint>")==1:
		
			sp = s.split(" ")
			op = o.split("^")
			if (sp[-2].endswith("ak") or sp[-2].endswith("ek") or sp[-2].endswith("a")) and ("<n>" in op[-2] or "<det>" in op[-2]):
			
				c+=1
				print s
				print op[-2]
				print "------------------"
				#print s.index("ek ")

		print c/len(sents_and_outputs)
		"""
		if unambiguous:
			sents_and_outputs = filter(lambda (sent,out): "ak " not in sent, sents_and_outputs)
		print len(sents_and_outputs)


		print len(sents_and_outputs)

		for ind, (sentence, output) in enumerate(sents_and_outputs):



			#if ind%1000 == 0: print "{}/{}".format(ind, len(sents_and_outputs))

			output_splitted = output.split("^")[1:]
			output_splitted = [o.split("$")[0] for o in output_splitted]
			sentence_splitted = sentence.split(" ")



			if "ek " in sentence: contains_ek+=1
			if "ek " in sentence and "ak " in sentence:
				ak_ind, ek_ind = -1, -1
				for jj, w in enumerate(sentence_splitted):
					if w.endswith("ek") and ek_ind==-1:
						ek_ind = jj
					elif w.endswith("ak") and ak_ind == -1:
						ak_ind = jj

				both[ak_ind-ek_ind]+=1





			verbs_count += output.count("<vbsint>")

			for i, w_output in enumerate(output_splitted): 

				if "<vbsint>" in w_output:
					#if len(sentence)<70 and sentence.count("ak ")>=2: print sentence
					ls = [sentence, output] #, str(i), w_output]
					word = w_output.split("/")[0]
					output_pruned = duplicate_handler(w_output)
					if not condition(output_pruned): break

					try:

						index = i
						#assert sentence_splitted[index]==word
						ls+=[str(index), output_pruned]			
						wr.writerow(ls)
						
						global labels #collect label statistics
						word_analysis =  re.findall("<.*?>", ls[-1])
						labels.update(word_analysis)
	

					except Exception as e:
						print sentence_splitted
						print "--------------"
						print [o.split("/")[0] for o in output_splitted]
						print zeuden
						print e
						print "==============================="
						errors+=1
					
					break
	l = len(sents_and_outputs)
	print both
	print contains_ek/(1.*l)
	print sum(both.values())
	print sum(both.values())/(1.*l)
	print "number of sentences: {}".format(l)
	print "average number of verbs per sentence: {}".format(verbs_count/l)
	print "error rate: {}".format(errors/l)
	print "duplicate rate: {}".format(duplicate_counter/l)
	#print "there are {} ambiguous verbs: {}".format(len(ambiguous), ambiguous)




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

	create_csv("dataset.csv", sents_and_outputs, condition=all_verbs, duplicate_handler=duplicate_handler)
	print labels









