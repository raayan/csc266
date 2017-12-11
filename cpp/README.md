# Raayan Pillai CSC 266 Project 
## Market Simulation on GPU

### Prerequisites
* C++11
* CUDA
* NVCC
* libcurl
* libjson
* make



### Running 
```
make
./main <gpu> <trades> <generations> <blocksize> <symbol>
```

* gpu - 0 for cpu, not 0 for gpu.
* trades - # of trades executed (reasonably 0-100)
* generations - # of futures to be simultanesouly modelled
* blocksize - blocksize for CUDA
* symbol - symbol from bitfinex to represent currencies exchange https://api.bitfinex.com/v1/symbols

### Third Party Tools
[https://github.com/nlohmann/json](JSON for Modern C++) was used to parse the JSON data from the server.
