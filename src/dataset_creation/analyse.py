import subprocess
import csv

with open("all_sentences_raw.txt", "r") as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]

i=0
with open("all_outputs.txt", 'w') as all_outputs:

      #wr = csv.writer(f, quoting=csv.QUOTE_ALL)

      for ind, sentence in enumerate(lines):

	  
	  if ind%500==0: print ind

	  query = 'echo "' + sentence +  '" | lt-proc apertium-eus/eus.automorf.bin'
	  try:
		  output = subprocess.check_output(query, shell=True).strip()
	  except: 
		print sentence
		output="NONE"
	   
	  all_outputs.write(output+"\n")



	  

