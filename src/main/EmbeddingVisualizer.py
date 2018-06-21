
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random


def visaulize(embedding_filename, only_verbs = False, only_nouns = False):
	
		"""
		plot the TSNE projection of the embedded vectors of the verbs in the training set.

		embedding: the trained embedding

		word2key: word, key dictionary

		verbs: a set of tuples (present_tense_verb, the verb_pos)

		"""

		with open(embedding_filename, "r") as f:
			lines = f.readlines()
			
		# collect the embedding of different verbs
	
		embeddings, words, labels = [], [], []
		 
		for line in lines[:3000]:
			
			word, lemma, output, vector_string = line.strip().split("\t")
			if only_verbs: 
				if (not "<vblex>" in output) and not "<vblex>" in output: continue
				
			if only_nouns:
				if not "<n>" in output: continue
				
			print output
			vec = [float(v) for v in vector_string.split(" ")]
			embeddings.append(vec)
			words.append(word)
			is_verb_sint = "<vbsint>" in output 
			is_verb_lex = "<vblex>" in output

			is_noun = "<n>" in output
			is_adj = "<adj>" in output
			is_other = not is_verb_sint and not is_verb_lex and not is_noun and not is_adj
			labels.append("auxilary verb" if is_verb_sint else "lexical verb" if is_verb_lex else "noun" if is_noun else "adjective" if is_adj else "other/unknown")
			
		embeddings = np.array(embeddings)


		# calculate TSNE projection & plot
		print "calculating projection..."
		
		proj = TSNE(n_components=2).fit_transform(embeddings)

		fig, ax = plt.subplots()
		
		xs, ys = proj[:,0], proj[:,1]
		xs, ys = list(xs), list(ys)
		colors_dict = {l: (random.random(), random.random(), random.random()) for l in set(labels)}
		labels_set = list(set(labels))
		jump = 256./len(labels_set)
		#color_dict = {labels_set[i]:int(i * jump) for i in range(len(labels_set))}
		colors_dict = {"auxilary verb": "red", "lexical verb": "brown", "noun": "blue", "adjective": "green", "other/unknown": "yellow"} 
		colors = [colors_dict[l] for l in labels]
		
		plots = []
		label_colors = []
		
		for l in set(labels):
			xs_l = [xs[i] for i in range(len(xs)) if labels[i] == l]
			ys_l = [ys[i] for i in range(len(xs)) if labels[i] == l]

			#correct_l = [correct[i] for i in range(len(xs)) if labels[i] == l]
			#markers = ["s" if correct=="1" else "o" for correct in correct_l]
			print colors_dict[l]
			print l
			q = ax.scatter(xs_l, ys_l, alpha=0.4, c = colors_dict[l], label = l)
			plots.append(q)
			label_colors.append(l)
		
		for i, w in enumerate(words):
                    if i%34 == 0:
    			ax.annotate(w, (xs[i],ys[i]), size = 11, weight="bold")
		plt.title("t-SNE projection of a sample of the learned embeddings")
		plt.legend(plots, label_colors, scatterpoints=1, title="Part Of Speech")
		plt.show()


if __name__ == '__main__':

		visaulize("EMBEDDINGS1.TXT")

