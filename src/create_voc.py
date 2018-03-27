from collections import Counter
import csv

"""this helper script contains methods to create VOC.txt and "NGRAMS.txt - a list of
all words and ngrams in the text, respectively."""

def get_all_ngrams(words, n=5):

	"""
	extract all ngrams from a list of words.
	n: the maximum length of an n-gram.
	return: a list of tuples (ngram, freq), sorted according to the frequency.
	"""

	fc = Counter()

	for w in words:
 
 		ngrams = []
 		for k in range(1, min(n+1, len(w)+1)):
   			ngrams = ngrams + find_ngrams(w, k)
 		fc.update(ngrams)
	
	items = fc.items()
	items = sorted(items, key = lambda (ngram, freq): -freq)
	return items

def get_all_suffixes_and_prefixes(words, n=5):

	suffixes, prefixes = Counter(), Counter()
	for w in words:
	  for i in range(1,n+1):
		suf, pre = w[-i:], w[:i]
		suffixes[suf]+=1
		prefixes[pre]+=1

	suf_items, pre_items = suffixes.items(), prefixes.items()
	suf_items, pre_items = sorted(suf_items, key = lambda (suf, freq): -freq), sorted(pre_items, key = lambda (pre, freq): -freq)

	return suf_items, pre_items
	

def find_ngrams(w, n):
  return ["".join(seq) for seq in zip(*[w[i:] for i in range(n)])]

def read_words():
	"""
	extreact all words from the csv file that contains the data sentences and morophological 
	analysis output.
	return: a list of tuples (word, frequency), sorted according to the frequency (highest first).
	"""

	words_counter = Counter()
	lemmas_counter = Counter()


	with open("data.csv", "r") as f:
		reader = csv.reader(f)

		for row in reader:

			orig_sentence, output, verb_output, verb_index = row
			sentence_words = filter(None, orig_sentence.split(" "))
			sentence_outputs = filter(None, output.split("^"))

			lemmas = [output.split("<")[0].split('/')[1].strip() if "/" in output else output.strip() for output in sentence_outputs]
			#if sentence_words!=orig_sentence.split(" "):
			#	print orig_sentence, verb_index
			words_counter.update(sentence_words)
			lemmas_counter.update(lemmas)

	items = words_counter.items()
	sorted_word_items = sorted(items, key = lambda (w,f): -f)

	items = lemmas_counter.items()
	sorted_lemmas_items = sorted(items, key = lambda (w,f): -f)

	return sorted_word_items, sorted_lemmas_items

def read_words2(): #reads from the morpho. analyser output
	"""
	extreact all words from the csv file that contains the data sentences and morophological 
	analysis output.
	return: a list of tuples (word, frequency), sorted according to the frequency (highest first).
	"""

	words_counter = Counter()
	lemmas_counter = Counter()


	with open("data.csv", "r") as f:
		reader = csv.reader(f)

		for row in reader:

			orig_sentence, output, verb_index, verb_output  = row
			sentence_words = filter(None, orig_sentence.split(" "))
			sentence_outputs = filter(None, output.split("^"))
			sentence_outputs = [o.replace(" ", "-") for o in sentence_outputs]
			output_splitted = [o.split("$")[0] for o in sentence_outputs]
			output_words = [o.split("/")[0] for o in output_splitted]
			lemmas = [output.split("<")[0].split('/')[1].strip() if "/" in output else output.strip() for output in sentence_outputs]
			#if sentence_words!=orig_sentence.split(" "):
			#	print orig_sentence, verb_index
			words_counter.update(output_words)
			lemmas_counter.update(lemmas)

	items = words_counter.items()
	sorted_word_items = sorted(items, key = lambda (w,f): -f)

	items = lemmas_counter.items()
	sorted_lemmas_items = sorted(items, key = lambda (w,f): -f)

	return sorted_word_items, sorted_lemmas_items
	
def read_verb_forms(fname): 

	verb_forms = Counter()
	with open(fname, "r") as f:
		reader = csv.reader(f)

		for row in reader:

			orig_sentence, output, verb_index, verb_output  = row
			verb_form =  verb_output.split("/")[0]
			lemma = verb_output.split("/")[1].split("<")[0]
			verb_forms[verb_form+"\t"+lemma]+=1

	items = verb_forms.items()
	items = sorted(items, key = lambda (w,f): -f)
	return items
	
def write_to_file(container, file_name, write_special_tokens = True):

	"""
	container: a list of strings
	write the items into a file (each in a seperate line.)
	"""
	sep="\t"

	with open(file_name, "w") as f:
		if write_special_tokens:
			f.write("<unk>"+sep+str(1)+"\n")
			f.write("<begin>"+sep+str(1)+"\n")
			f.write("<end>"+sep+str(1)+"\n")
			f.write("<verb>"+sep+str(1)+"\n")
		for (w,freq) in container:
			if w!=" ":
				f.write(w+sep+str(freq)+"\n")
	

words_and_freqs, lemmas_and_freqs = read_words2()
words = [w for (w,f) in words_and_freqs]
ngrams_and_freqs = get_all_ngrams(words)
suffs_and_freqs, prefs_and_freqs = get_all_suffixes_and_prefixes(words)
verb_forms_and_freqs = read_verb_forms("data.csv")

write_to_file(words_and_freqs, "VOC.txt")
write_to_file(ngrams_and_freqs, "NGRAMS.txt")
write_to_file(lemmas_and_freqs, "LEMMAS.txt")
write_to_file(suffs_and_freqs, "SUFFIXES.txt")
write_to_file(prefs_and_freqs, "PREFIXES.txt")
write_to_file(verb_forms_and_freqs, "VERB_FORMS.txt", write_special_tokens=False)
