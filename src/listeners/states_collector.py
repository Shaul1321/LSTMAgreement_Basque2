
class StatesCollector(object):

	def __init__(self, fname):
		self.fname = fname
		self.states_record = []
		
		
	def end(self):

		f = open(self.fname, "w")
		for record in self.states_record:
		
			for i, element in enumerate(record):
				
				sep = "\n" if i == len(record) - 1 else "\t"
				f.write(element+sep)
		
		f.close()
		self.states_record = []
			
		
	def collect(self, data_sample, sent_seen, gold, pred, states, max_size = 10000):
		"""
		collect bilstm states over words and record them in a file.
		"""
		if len(self.states_record) > max_size: return
		
		sent_str = " ".join(sent_seen)
		gold_str = " ".join([s if s!="" else "none" for s in gold])
		pred_str = " ".join([s if s!="" else "none" for s in pred])
		output = data_sample['output']
		
		vec = ""
		
		for index, o in enumerate(states):
		
			o_str = [str(round(v, 5)) for v in o]
			o_str = " ".join(o_str)
			sep = "" if index == len(states) -1 else "*"
			vec += o_str+sep
		
		self.states_record.append((sent_str, gold_str, pred_str, output, vec))
		
