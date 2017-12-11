trades=(100)
gens=(1, 6, 12, 36, 72, 144, 1024, 2048, 4096, 8192)
blocksize=(128, 256, 512, 1024)
for a in "${trades[@]}"
do
	for b in "${gens[@]}"
	do
		for c in "${blocksize[@]}"
		do
			./main 1 $a $b $c | tee -a results_gpu.txt
		done
	done
done
