trades=(1, 5, 10, 50, 100)
gens=(1, 6, 12, 36, 72, 144)
for b in "${gens[@]}"
do
	for a in "${trades[@]}"
	do
		./main 0 $a $b 0 | tee -a results_cpu.txt
	done
done
