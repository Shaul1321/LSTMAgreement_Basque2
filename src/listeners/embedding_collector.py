
class Collector(object):

	def __init__(self, encoder, voc_file, embedding_filename):
		self.encoder = encoder
		self.voc_file = voc_file
		self.embedding_filename = embedding_filename
		
	def collect(self, size = 15000):
	
		print "collecting embedding..."
		
		with open(self.voc_file, "r") as f:
			lines = f.readlines()
			
		vecs = []
		
		for i, line in enumerate(lines[:size]):
			#print i, len(lines)
			#if i % 500 == 0:
				#print "collecting embedding, line {}/{}".format(i, size)
			word, lemma, output, freq = line.strip().split("\t")
			vec = self.encoder.encode(word, "None", lemma).value()
			vecs.append((word,lemma,output, vec))
		
		f = open(self.embedding_filename, "w")
		#print "len: ", len(vecs)
		for (w,lemma,output,v) in vecs:
			as_string = " ".join([str(round(val,5)) for val in v])
			f.write(w+"\t"+lemma+"\t"+output+"\t"+as_string+"\n")
		f.close()
				
