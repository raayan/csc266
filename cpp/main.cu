#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <random>
#include <vector>
#include <math.h>
// CUDA imports
#include "helper_cuda.h"
#include "helper_string.h"
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
		cudaDeviceSynchronize();
		return ptr;
	}

	void operator delete(void *ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};

class Order : public Managed {
public:
	double price;
	double quantity;
	
	Order(double p, double q) {
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
	double mean;
	double stdd;
	vector<Order> asks;
	vector<Order> bids;

	MarketBook() {

	}
	
	double calculate_mean() {
		double total_quantity = 0;
		double total = 0;
		for(std::vector<Order>::size_type i = 0; i != asks.size(); i++) {
			total_quantity += asks[i].quantity;
			total += asks[i].quantity*asks[i].price;
		}
		for(std::vector<Order>::size_type i = 0; i != bids.size(); i++) {
			total_quantity += bids[i].quantity;
			total += bids[i].quantity*bids[i].price;
		}
		return total/total_quantity;
	}

	double calculate_total() {
		double total = 0;
		for(std::vector<Order>::size_type i = 0; i != asks.size(); i++) {
			total += asks[i].quantity;
		}
		for(std::vector<Order>::size_type i = 0; i != bids.size(); i++) {
			total += bids[i].quantity;
		}
		return total;
	}
	
	double calculate_stddev() {
		double total = calculate_total();
		double sum = 0;
		double probability = 0;
		for(std::vector<Order>::size_type i = 0; i != asks.size(); i++) {
			probability = asks[i].quantity/total;
			sum += asks[i].price*asks[i].price*probability;
		}
		for(std::vector<Order>::size_type i = 0; i != bids.size(); i++) {
			probability = bids[i].quantity/total;
			sum += bids[i].price*bids[i].price*probability;
		}
		double mean = calculate_mean();
		return pow((sum - mean*mean), 0.5);
	}
	
	// ask=0 for bid ask=1 for ask
	bool insert_order(Order active_order, bool ask) {
		double remaining = 0;
		int pos = 0;
		if(ask) {
			pos = (lower_bound(bids.begin(), bids.end(), active_order) - bids.begin());
			if(pos < bids.size()) {
				if(bids[pos].price >= active_order.price && active_order.quantity > bids[pos].quantity) {
					active_order.quantity -= bids[pos].quantity;
					bids.erase(bids.begin()+pos);
					return insert_order(active_order, ask);
				}else{
					bids[pos].quantity -= active_order.quantity;
					return true;
				}
			} else {
				pos = (lower_bound(asks.begin(), asks.end(), active_order) - asks.begin());
				if(asks[pos].price == active_order.price) {
					asks[pos].quantity += active_order.quantity;
				} else {
					asks.insert(asks.begin()+pos, active_order);
				}
				return false;
			}
		} else {
			pos = (lower_bound(asks.begin(), asks.end(), active_order) - asks.begin());
			if(asks[pos].price <= active_order.price && pos < asks.size()) {
				if(active_order.quantity > asks[pos].quantity) {
					active_order.quantity -= asks[pos].quantity;
					asks.erase(asks.begin()+pos);
					return insert_order(active_order, ask);
				}else{
					asks[pos].quantity -= active_order.quantity;
					return true;
				}
			} else {
				pos = (lower_bound(bids.begin(), bids.end(), active_order) - bids.begin());
				if(bids[pos].price == active_order.price) {
					bids[pos].quantity += active_order.quantity;
				} else {
					bids.insert(bids.begin()+pos, active_order);
				} 
				return false;
			}
		}
	}
};

std::ostream& operator<< (std::ostream & out, Order const& data) {
	out << "Order(";
	out << "price=";
	out << data.price;
	out << ", ";
	out << "quantity=";
	out << data.quantity;
	out << ")";
	return out ;
}
std::ostream& operator<< (std::ostream & out, MarketBook const& data) {
	out << "MarketBook(";
	out << "asks=[\n";
	for(std::vector<Order>::size_type i = 0; i != data.asks.size(); i++) {
		out << data.asks[i];
		if(i < data.asks.size()-1) out << "\n";
	}
	out << "]\n";
	out << "bids=[\n";
	for(std::vector<Order>::size_type i = 0; i != data.bids.size(); i++) {
		out << data.bids[i];
		if(i < data.bids.size()-1) out << "\n";
	}
	out << "])";
	return out ;
}

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

__device__ int convolve() {
	return 1;
}

__global__ void convolve_gpu(int blocks, int blocksize) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_count = blocks*blocksize;
}


/////////////////////////// CPU stuff: ///////////////////////////
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);
	return size * nmemb;
}


int main(int argc, char *argv[]) {
	if(argc != 3) {
		fprintf(stderr, "Usage: %s gpu generations\n", argv[0]);
		fprintf(stderr, "gpu=0 for cpu\n");
		exit(1);
	}


// Get the current order book
	string baseURL = "https://api.bitfinex.com/v1";
	string type = "book";
	string symbol = "ltcbtc";
	string URL = baseURL + "/" + type + "/" + symbol;
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

	json obj;
	double amount, price;
	MarketBook book;

	for(int i=0; i<data["bids"].size(); i++){
		obj = data["bids"][i];
		amount = atof(obj["amount"].get<string>().c_str());
		price = atof(obj["price"].get<string>().c_str());
		book.bids.push_back(Order(price, amount));
	}
	for(int i=0; i<data["asks"].size(); i++){
		obj = data["asks"][i];
		amount = atof(obj["amount"].get<string>().c_str());
		price = atof(obj["price"].get<string>().c_str());
		book.asks.push_back(Order(price, amount));
	}

	cout << book.calculate_mean() << endl;
	cout << book.calculate_total() << endl;
	cout << book.calculate_stddev() << endl;
	sort(book.asks.begin(), book.asks.end());
	sort(book.bids.begin(), book.bids.end());
	int generations = atoi(argv[2]);

	if(atoi(argv[1]) == 0) {
	// CPU
		default_random_engine generator;
		
		for(int i=0; i<10; i++) {
			double gen_mean = book.calculate_mean();
			double gen_stddev = book.calculate_stddev();
			normal_distribution<double> distribution(gen_mean, gen_stddev);
			double number = distribution(generator);
			cout << number << endl;
		}


	} else {
	// GPU
	// Setup
		ggc::Timer t("generations");
	// Create GPU timers
		cudaEvent_t start, stop;
		float total;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

	// Get GPU
		int devID;
		devID = gpuGetMaxGflopsDeviceId();
		checkCudaErrors(cudaSetDevice(devID));

	/*
	// COPY TO GPU
	// create GPU ppm
		cudaMalloc((void **) &d_img_r, size * sizeof(*d_img_r));

		// Copy data from host to device
		cudaMemcpy(d_img_r, img.r, size * sizeof(*d_img_r), cudaMemcpyHostToDevice);

		// Bind pointers
		cudaMemcpy(&(d_img->r), &d_img_r, sizeof(d_img->r), cudaMemcpyHostToDevice);

		ppm * d_out;
		int *d_out_r, *d_out_g, *d_out_b;

		gpuErrchk(cudaMalloc((void**)&d_out, sizeof(ppm)));
		gpuErrchk(cudaMemcpy(d_out, &out, sizeof(ppm), cudaMemcpyHostToDevice));

		cudaMalloc((void **) &d_out_r, size * sizeof(*d_out_r));
		cudaMalloc((void **) &d_out_g, size * sizeof(*d_out_g));
		cudaMalloc((void **) &d_out_b, size * sizeof(*d_out_b));

		// Copy data from host to device
		cudaMemcpy(d_out_r, out.r, size * sizeof(*d_out_r), cudaMemcpyHostToDevice);
		cudaMemcpy(d_out_g, out.g, size * sizeof(*d_out_g), cudaMemcpyHostToDevice);
		cudaMemcpy(d_out_b, out.b, size * sizeof(*d_out_b), cudaMemcpyHostToDevice);

		// Bind pointers
		cudaMemcpy(&(d_out->r), &d_out_r, sizeof(d_out->r), cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_out->g), &d_out_g, sizeof(d_out->g), cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_out->b), &d_out_b, sizeof(d_out->b), cudaMemcpyHostToDevice);

	// create GPU convo matrix
		int * d_n;
		int * d_cm;
		gpuErrchk(cudaMalloc((void**)&d_n, 1*sizeof(int)));
		gpuErrchk(cudaMemcpy(d_n, &n, 1*sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(&d_cm, sizeof(int)*n*n));
		gpuErrchk(cudaMemcpy(d_cm, cm, sizeof(int)*n*n, cudaMemcpyHostToDevice));
	// Launch Kernels
		int blocksize = atoi(argv[4]);
		int blockcount = (img.ysize/1.0/blocksize)+1;

	// Start CPU timer and GPU timer
		t.start();
		gpuErrchk(cudaEventRecord(start));

	// Actual kernel function
		convolve_gpu<<< blockcount, blocksize >>>(d_img, d_out, d_n, d_cm, blockcount, blocksize);
		getLastCudaError("Kernel execution failed (convolve_gpu).");

	// Stop CPU and GPU timer
		t.stop();
		gpuErrchk(cudaEventRecord(stop));
		gpuErrchk(cudaEventSynchronize(stop));

	// save time to total float
		gpuErrchk(cudaEventElapsedTime(&total, start, stop));
		gpuErrchk(cudaMemcpy(&out, d_out, sizeof(ppm), cudaMemcpyDeviceToHost));

	// COPY BACK TO CPU
	//	int *out_r, *out_g, *out_b;
	//	out_b = (int *) malloc(size * sizeof(int));
	//	gpuErrchk(cudaMemcpy(out_b, d_out_b, size * sizeof(*out_b), cudaMemcpyDeviceToHost));
	//	out.b = out_b;


	// Print times
		printf("%d, %d, %llu, %f, %s, %s\n", blocksize, blockcount, t.duration(), total, argv[2], argv[1]);
	*/

	// CUDA cleanup
		gpuErrchk(cudaEventDestroy(start));
		gpuErrchk(cudaEventDestroy(stop));
	}

	return 0;
}
