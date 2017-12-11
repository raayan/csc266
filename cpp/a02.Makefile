all: 2dconv 2dconv_gpu 2dconv_gpu_um

2dconv: 2dconv.cpp ppma_io.a
	gcc -O3 $^ -o $@ -lm -lrt

# modify as necessary
2dconv_threads: 2dconv_threads.cpp
	gcc -O3 $^ -o $@ -lm -lrt

# modify as necessary
2dconv_vec1: 2dconv_threads.cpp
	gcc -O3 -mavx2 $^ -o $@ -lm -lrt

# modify as necessary
2dconv_vec2: 2dconv_vec2.cpp
	gcc -O3 -mavx2 $^ -o $@ -lm -lrt

2dconv_gpu: 2dconv_gpu.cu ppma_io.a
	nvcc -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -lineinfo --compiler-options -Wall -o $@ $+ -lcudadevrt

2dconv_gpu_um: 2dconv_gpu_um.cu ppma_io.a
	nvcc -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -lineinfo --compiler-options -Wall -o $@ $+ -lcudadevrt

ppma_io.a: ppma_io.o 
	ar rs $@ $<

clean:
	rm -f ppma_io.a ppma_io.o 2dconv 2dconv_threads 2dconv_vec1 2dconv_vec2

