import numpy as np

def all_verb_indices(output):
	output_splitted = output.split("^")
	indices = []
	for i, o in enumerate(output_splitted):
		if "<vbsint>" in o and not "<dem>" in o:
			indices.append(i)

	return indices


def load_data(fname):

	with open(fname, "r") as f:

		lines = f.readlines()

	verb = "da"

	data = []

	for line in lines:

		orig_sent, gold, pred, output, states = line.strip().split("\t")
		states_splitted = states.split("*")
		sent_splitted = orig_sent.split(" ")
		gold_splitted, pred_splitted = gold.split(" "), pred.split(" ")	

		indices = all_verb_indices(output)
		ind1, ind2 = indices
		da_index = ind1 if sent_splitted[ind1] == verb else ind2
		other_index = ind1 if ind1!=da_index else ind2
		data.append((sent_splitted, gold_splitted, pred_splitted, da_index, other_index, states_splitted))

	return data

def prepare_dataset(sents):

	data = []
	y_1, y_0 = 0., 0.
	for example in sents:

		sent_splitted, gold_splitted, pred_splitted, da_index, other_index, states = example
		#states = [s.split(" ") for s in states]
		#states = [[float(val) for val in s] for s in states]
		print sent_splitted[1]
		assert len(pred_splitted) == len(gold_splitted) == len(states) == len(sent_splitted)

		for i, (p,g,s) in enumerate(zip(pred_splitted, gold_splitted, states)):

			if g != "none":
				word = sent_splitted[i]
				dis_da = abs(da_index - i)
				dis_other = abs(other_index - i)
				closer_to_da = dis_da <= dis_other
				related_to_da = p == "a"
				closest_is_right = (closer_to_da and related_to_da) or (not related_to_da and not closer_to_da)
				y = "1" if closest_is_right else "0"
				if y == "1": y_1+=1
				else:	y_0+=1
				data.append((s, word, y, p, g))
	print y_1, y_0, y_1/y_0, y_1/(y_1+y_0)
	np.random.shuffle(data)
	
	# data = filter(lambda (s,y,p,g): p == g, data)
	return data

def write_to_file(data):
	f = open("data.txt", "w")
	for example in data:
		s, w, y, p , g = example
		f.write(s+"\t"+w+"\t"+y+"\t"+p+"\t"+g+"\n")
	f.close()



raw_data = load_data("preds_da.txt")
dataset = prepare_dataset(raw_data)
write_to_file(dataset)
	
