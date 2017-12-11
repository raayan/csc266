all: test 

test: test.o
	g++ test.o -o test -lcurl -ljsoncpp

test.o: test.cpp
	g++ -c test.cpp

clean:
	rm *o test
