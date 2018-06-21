def load_data(fname):

	with open(fname, "r") as f:
		lines = f.readlines()

	data = []

	for line in lines:

		s, w, y, p, g = line.strip().split("\t")
		y = int(y)
		s = [float(value) for value in s.split(" ")]
		data.append((s,y,p,g))

	return data

DATA = load_data("data.txt")
l = int(len(DATA) * 0.8)
TRAIN, DEV = DATA[:l], DATA[l:]

