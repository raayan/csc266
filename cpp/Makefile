all: main 

test: test.o
	g++ test.o -o test -lcurl -ljsoncpp

main: main.cu
	nvcc -w -std=c++11 -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -lineinfo --compiler-options -Wall -o $@ $+ -lcudadevrt -lcurl -lpthread

test.o: test.cpp
	g++ -c test.cpp

clean:
	rm *o test
