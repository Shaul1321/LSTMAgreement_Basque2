import json
import re

data = []
with open('all.json') as f:
    for line in f:
        data.append(json.loads(line))

l = len(data)

data = data[:int(0.99*l)]
f = open("all_sentences_raw.txt", "w")
s = ""
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789?.,:-()[] \n"
strings = []
for i, d in enumerate(data):
	if i%1000==0: print "{}/{}".format(i, len(data))

	text = d['text']
	title, content = text.split("\n", 1)
        
	content = re.sub(r"([\w/'+$\s]+|[^\w/'+$\s])\s*", r"\1 ", content).strip()

	for c in content:
		if c not in chars:
			content = content.replace(c, "")
	#s += content
	strings.append(content)

s = "".join(strings)
s = s.replace("\n", "")
s = s.replace("-", " ")
s = s.encode('utf-8')
s = " ".join(filter(None, s.split(" ")))
#s = s.lower()


s = s.split(".")
for i in range(100):
	print s[i]+"."
	print s[i][0]==" "
	print "it's", s[i][0]


text = ""
for i, sent in enumerate(s):
	if i%1000 == 0:
		print "{}/{}".format(i, len(s))
	to_write = sent
	if len(sent) <=30: continue

	first = sent[0]
	if first == " ": to_write = sent[1:]
	text+=to_write+".\n"
	#f.write(to_write+".\n")
f.write(text)
f.close()

print "===================="

txt = "".join(s)
voc = list(set(txt.split(" ")))

for i in range(2):
	print voc[i].strip()

print len(voc)

