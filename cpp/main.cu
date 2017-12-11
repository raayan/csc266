#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <random>
#include <vector>
#include <ctime>
#include <math.h>
#include <pthread.h>
// CUDA imports
#include <curand.h>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "helper_string.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
// Library imports
#include <curl/curl.h>
#include "json.hpp"
#include "Timer.h"

using namespace std;
using json = nlohmann::json;

/////////////////////////// Classes: ///////////////////////////
class Managed {
public:
	void *operator new(size_t len) {
		void *ptr;
		cudaMallocManaged(&ptr, len);
//		cudaDeviceSynchronize();
		return ptr;
	}

	void operator delete(void *ptr) {
//		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};

class Order : public Managed {
public:
	double price;
	double quantity;
	
	__host__ __device__ Order() {
		price = 0;
		quantity = 0;
	}
	
	__host__ __device__ Order(double p, double q) {
		price = p;
		quantity = q;
	}

	bool operator < (const Order& ord) const {
		return (price < ord.price);
	}

	bool operator > (const Order& ord) const {
		return (price > ord.price);
	}
};

class MarketBook : public Managed {
public:
	int size = 100;
	Order asks[100];
	Order bids[100];

	MarketBook() {
		cudaMallocManaged((void**)&asks, size*sizeof(Order));
		cudaMallocManaged((void**)&bids, size*sizeof(Order));
	}
	__device__ MarketBook(int a) {
	}
	
	__host__ __device__ double calculate_mean() {
		double total_quantity = 0;
		double total = 0;
		for(int i = 0; i != size; i++) {
			total_quantity += asks[i].quantity;
			total += asks[i].quantity*asks[i].price;
		}
		for(int i = 0; i != size; i++) {
			total_quantity += bids[i].quantity;
			total += bids[i].quantity*bids[i].price;
		}
		return total/total_quantity;
	}


	__host__ __device__ double calculate_total() {
		double total_quantity = 0;
		for(int i = 0; i != size; i++) {
			total_quantity += asks[i].quantity;
		}
		for(int i = 0; i != size; i++) {
			total_quantity += bids[i].quantity;
		}
		return total_quantity;
	}
	
	__host__ __device__ double calculate_stddev() {
		double total = calculate_total();
		double sum = 0;
		double probability = 0;

		for(int i = 0; i != size; i++) {
			probability = asks[i].quantity/total;
			sum += asks[i].price*asks[i].price*probability;
		}
		for(int i = 0; i != size; i++) {
			probability = bids[i].quantity/total;
			sum += bids[i].price*bids[i].price*probability;
		}
		double mean = calculate_mean();
		return pow((sum - mean*mean), 0.5);
	}
	
	__host__ __device__ int find_ask(double price) {
		for(int i=0; i < size-1; i++) {
			if(asks[i].price <= price) {
				if(asks[i+1].price > price) {
					return i;
				} 
			}
		}
		return -1;
	}

	__host__ __device__ int find_bid(double price) {
		for(int i=0; i < size-1; i++) {
			if(bids[i].price >= price) {
				if(bids[i+1].price < price) {
					return i;
				} 
			}
		}
		return -1;
	}
	
	__host__ __device__ int index_ask(double price) {
		for(int i=0; i < size-1; i++) {
			if(asks[i].price == price) return i;
		}
		return -1;
	}
	__host__ __device__ int index_bid(double price) {
		for(int i=0; i < size-1; i++) {
			if(bids[i].price == price) return i;
		}
		return -1;
	}	

	__host__ __device__ bool remove_ask(double price) {
		int pos = find_ask(price);
		if(pos == -1) return false;
		for(int i=pos; i < size-1; i++) {
			asks[i].price = asks[i+1].price;
			asks[i].quantity = asks[i+1].quantity;
		}
		return true;	
	}

	__host__ __device__ bool remove_bid(double price) {
		int pos = find_bid(price);
		if(pos == -1) return false;
		for(int i=pos; i < size-1; i++) {
			bids[i].price = bids[i+1].price;
			bids[i].quantity = bids[i+1].quantity;
		}
		return true;	
	}

	__host__ __device__ bool insert_ask(Order ask) {
		int i = 0; 
		for(int j=0; j<size-1; j++) {
			if(asks[j].price != 0 && asks[j+1].price ==0) {
				i = j;
			}
		}
		while ((i > 0) && (ask.price < asks[i-1].price)) {   
			asks[i].price = asks[i-1].price;
			asks[i].quantity = asks[i-1].quantity;
			i = i - 1;
		}
		asks[i].price = ask.price;
		asks[i].quantity = ask.quantity;
		return true;
	}

	__host__ __device__ bool insert_bid(Order bid) {
		int i = 0; 
		for(int j=0; j<size-1; j++) {
			if(bids[j].price != 0 && bids[j+1].price ==0) {
				i = j;
			}
		}
		while ((i > 0) && (bid.price < bids[i-1].price)) {   
			bids[i].price = bids[i-1].price;
			bids[i].quantity = bids[i-1].quantity;
			i = i - 1;
		}
		bids[i].price = bid.price;
		bids[i].quantity = bid.quantity;
		return true;
	}



	__host__ __device__ bool insert_order(Order active_order, bool ask) {
		double remaining = 0;
		int pos = 0;
		if(ask) {
			pos = find_bid(active_order.price);
			if(pos != -1) {
				if(active_order.quantity > bids[pos].quantity) {
					active_order.quantity -= bids[pos].quantity;
					remove_bid(bids[pos].price);
					return insert_order(active_order, ask);
				}else{
					bids[pos].quantity -= active_order.quantity;
					return true;
				}
			} else {
				pos = index_ask(active_order.price);
				if(pos != -1) {
					asks[pos].quantity += active_order.quantity;
				} else {
					insert_ask(active_order);
				}
				return false;
			}
		} else {
			pos = find_ask(active_order.price);
			if(pos != -1) {
				if(active_order.quantity > asks[pos].quantity) {
					active_order.quantity -= asks[pos].quantity;
					remove_ask(asks[pos].price);
					return insert_order(active_order, ask);
				}else{
					asks[pos].quantity -= active_order.quantity;
					return true;
				}
			} else {
				pos = index_bid(active_order.price);
				if(pos != -1) {
					bids[pos].quantity += active_order.quantity;
				} else {
					insert_bid(active_order);
				} 
				return false;
			}
		}
	}
	
/*
	__host__ __device__ bool sort_books() {
		sort(asks.begin(), asks.end());
		sort(bids.begin(), bids.end());
		return true;
	}
*/
	__device__ void copy_in(MarketBook book) {
		for(int i=0; i<size; i++) {
			asks[i].price = book.asks[i].price;
			asks[i].quantity = book.asks[i].quantity;
			bids[i].price = book.bids[i].price;
			bids[i].quantity = book.bids[i].quantity;
		}
	}
};


/////////////////////////// CUDA stuff: ///////////////////////////
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void setup_kernel(curandState *state) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, id, 0, &state[id]);
}


__global__ void simulate_market(MarketBook in_book, int generations, curandState *state, double *means, double *devs) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	MarketBook book = MarketBook(0);
	book.copy_in(in_book);
	curand_init(1234, id, 0, &state[id]);
	curandState localState = state[id];

	double gen_mean, gen_stddev, number;
	for(int i=0; i<generations; i++) {
		gen_mean = book.calculate_mean();
		gen_stddev = book.calculate_stddev();
		number = curand_log_normal_double(&localState, gen_mean, gen_stddev)-1;
		if(curand_uniform(&localState) > 0.5) {
			book.insert_order(Order(number, 1), false);
		} else {
			book.insert_order(Order(number, 1), true);
		}
	}
	means[id] = gen_mean;
	devs[id] = gen_stddev;
}


/////////////////////////// CPU stuff: ///////////////////////////
struct thread_data
{
	double mean;
	double stddev;
	int g;
	MarketBook book;
};

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);
	return size * nmemb;
}

std::string get_time() {
	time_t t = time(0);
	char cstr[128];
	strftime(cstr, sizeof(cstr), "%Y-%m-%d %H:%M:%S", localtime(&t) );
	return cstr;
}

std::string get_time(int ss) {
	time_t t = time(0)+ ss;
	char cstr[128];
	strftime(cstr, sizeof(cstr), "%Y-%m-%d %H:%M:%S", localtime(&t));
	return cstr;
}

int get_ts() {
	time_t t = time(0);
	return t;
}

json get_response(string URL) {
	CURL *curl;
	CURLcode res;
	string readBuffer;
	json data;
	curl = curl_easy_init();
	if(curl) {
		curl_easy_setopt(curl, CURLOPT_URL, URL.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		res = curl_easy_perform(curl);
		curl_easy_cleanup(curl);
		data = json::parse(readBuffer.c_str());
	}
	return data;
}

MarketBook get_market_book(json data) {
	json obj;
	double amount, price;
	MarketBook book;
	for(int i=0; i<data["bids"].size(); i++){
		obj = data["bids"][i];
		amount = atof(obj["amount"].get<string>().c_str());
		price = atof(obj["price"].get<string>().c_str());
		book.bids[i] = Order(price, amount);
	}
	for(int i=0; i<data["asks"].size(); i++){
		obj = data["asks"][i];
		amount = atof(obj["amount"].get<string>().c_str());
		price = atof(obj["price"].get<string>().c_str());
		book.asks[i] = Order(price, amount);
	}
	return book;
}

void *simulate_market_cpu(void *threadarg) {
	thread_data *my_data;
	my_data = (thread_data *) threadarg;
	MarketBook book = my_data->book;
	int g = my_data->g;
	default_random_engine generator;
	double gen_mean, gen_stddev, number;
	for(int i=0; i<g; i++) {
		gen_mean = book.calculate_mean();
		gen_stddev = book.calculate_stddev();
		normal_distribution<double> distribution(gen_mean, gen_stddev);
		number = distribution(generator);
		if(rand()%2) {
			book.insert_order(Order(number, 1), false);
		} else {
			book.insert_order(Order(number, 1), true);
		}
	}
	my_data->mean = book.calculate_mean();
	my_data->stddev = book.calculate_stddev();
}


int main(int argc, char *argv[]) {
	printf("[OUTPUT][o]:b_ts, b_mean, b_std, p_ts, p_mean, p_std, a_ts, a_mean, a_std, rte(ms), gpu, trades, generations, blocksize\n");

	if(argc != 6) {
		fprintf(stderr, "Usage: %s gpu trades generations blocksize symbol\n", argv[0]);
		fprintf(stderr, "gpu=0 for cpu\n");
		exit(1);
	}


// Get the current order book
	string baseURL = "https://api.bitfinex.com/v1";
	string type = "stats";
	string symbol = argv[5];
	string URL = baseURL + "/" + type + "/" + symbol;
	json data = get_response(URL);

	double volume24 = atof(data[0]["volume"].get<string>().c_str());
	double volume1sec = volume24*1.0/(24*60*60);

	type = "book";
	URL = baseURL + "/" + type + "/" + symbol;
	data = get_response(URL);

	int gpu = atoi(argv[1]);
	int trades = atoi(argv[2]);
	int generations = atoi(argv[3]);
	int blocksize = atoi(argv[4]);
	MarketBook book = get_market_book(data);

	
	printf("[%s][r] mean: %f, std-dev: %f\n", get_time().c_str(),  book.calculate_mean(), book.calculate_stddev());

	if(gpu == 0) {
	// CPU
		double b_mean = book.calculate_mean();
		double b_stddev = book.calculate_stddev();
		int b_ts = get_ts();
		ggc::Timer t("generations");

		pthread_t threads[generations];
		thread_data thread_data_array[generations];
		pthread_attr_t th_attr;
		pthread_attr_init(&th_attr);
		pthread_attr_setdetachstate(&th_attr, PTHREAD_CREATE_JOINABLE);

		t.start();
		for(int i=0; i<generations; i++) {
			thread_data_array[i].g = generations;
			thread_data_array[i].book = book;
			pthread_create(&threads[i], &th_attr, simulate_market_cpu, (void *) &thread_data_array[i]);
		}
		for(int i=0; i<generations; i++) {
			pthread_join(threads[i], NULL);
		}
		t.stop();

		double avg_stddev = 0;
		double avg_mean = 0;
		for(int i=0; i<generations; i++) {
			avg_stddev += thread_data_array[i].stddev;
			avg_mean += thread_data_array[i].mean;
		}
		avg_stddev = avg_stddev/generations;
		avg_mean = avg_mean/generations;
		int p_ts = get_ts();
		printf("[%s][p] avg-mean: %f, avg-std-dev: %f\n", get_time().c_str(), avg_mean, avg_stddev);
		printf("[%s][i] runtime: %llu ms\n", get_time().c_str(), t.duration()/1000000);
		printf("[%s][i] price-expected-at: %s (%fs)\n", get_time().c_str(), get_time(trades*1.0/volume1sec).c_str(), trades*1.0/volume1sec);

		printf("[%s][i] sleeping for %fs\n", get_time().c_str(), trades*1.0/volume1sec);
		sleep(trades*1.0/volume1sec);
		MarketBook new_book = get_market_book(get_response(URL));
		int a_ts = get_ts();
		printf("[%s][r] mean: %f, std-dev: %f\n", get_time().c_str(), new_book.calculate_mean(), new_book.calculate_stddev());
		printf("[OUTPUT][o]:%d, %f, %f, %d, %f, %f, %d, %f, %f, %f, %d, %d, %d, %d\n", b_ts, b_mean, b_stddev, p_ts, avg_mean, avg_stddev, a_ts, new_book.calculate_mean(), new_book.calculate_stddev(), t.duration()/1000000.0, gpu, trades, generations, blocksize);
		

	} else {
		double b_mean = book.calculate_mean();
		double b_stddev = book.calculate_stddev();
		int b_ts = get_ts();
	// GPU
	// Setup
	// Create GPU timers
		cudaEvent_t start, stop;
		float total;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

	// Get GPU
		int devID;
		devID = gpuGetMaxGflopsDeviceId();
		checkCudaErrors(cudaSetDevice(devID));
		
		curandState *devStates;
		curandGenerator_t gen;
		cudaMalloc((void **)&devStates, 64*64*sizeof(curandState));
		cudaMallocManaged((void **)&book, sizeof(book));
		
		double * means;
		double * devs;
		cudaMallocManaged(&means, generations*sizeof(double));
		cudaMallocManaged(&devs, generations*sizeof(double));


		setup_kernel<<<ceil(1.0*generations/blocksize), blocksize>>>(devStates);

		cudaEventRecord(start);

		simulate_market<<<ceil(1.0*generations/blocksize), blocksize>>>(book, trades, devStates, means, devs);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&total, start, stop);

//		cudaDeviceSynchronize();
		
		
		double avg_stddev = 0;
		double avg_mean = 0;
		for(int i=0; i<generations; i++) {
//			printf("[%d] mean: %f, std-dev: %f\n", i, means[i], devs[i]);
			avg_stddev += devs[i];
			avg_mean += means[i];
		}
		int p_ts = get_ts();
		avg_stddev = avg_stddev/generations;
		avg_mean = avg_mean/generations;
		
		printf("[%s][p] avg-mean: %f, avg-std-dev: %f\n", get_time().c_str(), avg_mean, avg_stddev);
		printf("[%s][i] runtime: %f ms\n", get_time().c_str(), total);
		printf("[%s][i] price-expected-at: %s (%fs)\n", get_time().c_str(), get_time(trades*1.0/volume1sec).c_str(), trades*1.0/volume1sec);

		printf("[%s][i] sleeping for %fs\n", get_time().c_str(), trades*1.0/volume1sec);

		sleep(trades*1.0/volume1sec);
		int a_ts = get_ts();
		MarketBook new_book = get_market_book(get_response(URL));
		printf("[%s][r] mean: %f, std-dev: %f\n", get_time().c_str(), new_book.calculate_mean(), new_book.calculate_stddev());
	
		printf("[OUTPUT][o]:%d, %f, %f, %d, %f, %f, %d, %f, %f, %f, %d, %d, %d, %d\n", b_ts, b_mean, b_stddev, p_ts, avg_mean, avg_stddev, a_ts, new_book.calculate_mean(), new_book.calculate_stddev(), total, gpu, trades, generations, blocksize);
		
	
	// CUDA cleanup
		gpuErrchk(cudaEventDestroy(start));
		gpuErrchk(cudaEventDestroy(stop));
	}

	return 0;
}
