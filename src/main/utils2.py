import csv
import re

voc_size = 100000 #100000
ngrams_size = 100000 #75000
lemmas_size = 30000 #20000
suffixes_size = 100000 #100000
prefixes_size = 100000 #10000

def read_words(filename):
	
	words = []
	with open(filename, "r") as f:
		lines = f.readlines()

	for line in lines:

		line_splitted = line.strip().split("\t")
		if len(line_splitted) == 1: line_splitted = [" "]+line_splitted
		w, freq = line_splitted
		words.append(w)

	return words

def generate_index_mapping(words, dict_size=50000):
	if "<unk>" not in words: words = ["<unk>"] + words
	tokens = words[:dict_size]
	
	W2I = {w:i for i,w in enumerate(tokens)}
	I2W = {i:w for i,w in enumerate(tokens)}
	return W2I, I2W

def generate_label_index_mapping():

	dative = ["<NI_HU>", "<NI_HK>","<NI_HI>", "<NI_GU>", "<NI_NI>", "<NI_ZU>", "<NI_ZK>", "None"]
	absolutive = ["<NR_HU>", "<NR_HK>","<NR_HI>", "<NR_GU>", "<NR_NI>", "<NR_ZU>", "<NR_ZK>", "None"]
	ergative = ["<NK_HU>", "<NK_HK>", "<NK_HI>", "<NK_GU>", "<NK_NI>", "<NK_ZU>", "<NK_ZK>", "None"]
	#dative = [d.lower() for d in dative]
	#absolutive = [a.lower() for a in absolutive]
	#ergative = [e.lower() for e in ergative]

	D2I = {agreement:index for index, agreement in enumerate(sorted(dative))}
	A2I = {agreement:index for index, agreement in enumerate(sorted(absolutive))}
	E2I = {agreement:index for index, agreement in enumerate(sorted(ergative))}

	I2D = {index:agreement for index, agreement in enumerate(sorted(dative))}
	I2A = {index:agreement for index, agreement in enumerate(sorted(absolutive))}
	I2E = {index:agreement for index, agreement in enumerate(sorted(ergative))}

	return D2I, A2I, E2I, I2D, I2A, I2E

def create_dataset(filename):

	labels = ["orig_sentence", "output", "verb_index", "verb_output"]
	dataset = []
	analyser_outputs = set()

	with open(filename, "r") as f:

		reader = csv.reader(f)

		for row in reader:

			orig_sentence, output, verb_output, verb_index = row
			if output[0]!="^": continue #sentences that start with number such as 3D...
			sentence_words = orig_sentence.split(" ")

			sent_dictionary = {}
			for i, val in enumerate(row):
				sent_dictionary[labels[i]] = val

			dataset.append(sent_dictionary)

	return dataset


def parse_analyser_output(data):

	as_list = filter(None, data.split("^"))
	sent = [o.split("/")[0] for o in as_list]
	sent_data = []
	anlayser_tokens = set()

	for o in as_list:
		#print w, w.split("/")[1]
		word, output  = o.split("/", 1)
		lemma = output.split("<")[0]
		word_analysis =  re.findall("<.*?>", output)
		anlayser_tokens.update(word_analysis)
		sent_data.append((word, lemma, word_analysis))

	return anlayser_tokens, sent_data

def get_lemmas(fname):
	lemmas = set()

	with open(fname, "r") as f:
		lines = f.readlines()
	
	for line in lines:
		if line[0]!="^": continue
		as_list = filter(None, line.strip().split("^"))

		for o in as_list:

			word, output  = o.split("/", 1)
			lemma = output.split("<")[0]
			lemmas.add(lemma)

	return lemmas



def read_tags_list():
	with open("wiki.tag", "r") as f:
		lines = f.readlines()

	tags = []
	for line in lines:
		splitted = line.strip().split("\t")
		tags.append(splitted[1])
	return tags

words = read_words("VOC.txt")
ngrams = read_words("NGRAMS.txt")
suffixes = read_words("SUFFIXES.txt")
prefixes = read_words("PREFIXES.txt")	
W2I, I2W = generate_index_mapping(words, dict_size=voc_size)
SUFFIX2I, I2SUFFIX = generate_index_mapping(suffixes, dict_size=suffixes_size)
PREFIX2I, I2PREFIX = generate_index_mapping(prefixes, dict_size=prefixes_size)
NGRAM2I, I2NGRAM = generate_index_mapping(ngrams, dict_size=ngrams_size)
print "loaded voc and ngrams."
#SENTENCES = create_dataset("raw_sentences_present_all_mostupdated_capital_NO_DUPLICATES2.csv")
SENTENCES = create_dataset("dataset.csv")
DEV_SENTENCES = create_dataset("treebank/train.csv")
print "created dataset."
analyser_outputs = read_tags_list()
D2I, A2I, E2I, I2D, I2A, I2E = generate_label_index_mapping()

OUTPUT2IND, IND2OUTPUT = generate_index_mapping(analyser_outputs, dict_size = len(analyser_outputs))
lemmas = read_words("LEMMAS.txt")
print "finished reading lemmas."
LEMMA2I, I2LEMMA = generate_index_mapping(lemmas, dict_size = lemmas_size)


del words
del ngrams
del lemmas
del I2SUFFIX
del I2PREFIX
del I2NGRAM
print "all finished."
		
