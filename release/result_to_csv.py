import csv

with open('results.csv', 'ab') as csvfile:
	with open('results_gpu.txt', 'rb') as fi:
		for line in fi:
			if line[9] == "o":
				csvfile.write(line[12:])
		csvfile.flush()
	with open('results_cpu.txt', 'rb') as fi:
		for line in fi:
			if line[9] == "o":
				csvfile.write(line[12:])
		csvfile.flush()


	
	

