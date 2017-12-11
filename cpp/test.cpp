// normal_distribution
#include <iostream>
#include <string>
#include <random>
#include <curl/curl.h>
#include <jsoncpp/json/json.h>

using namespace std;

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);
	return size * nmemb;
}

int main()
{
	int nrolls = 1000;
	int nstars = 10;

	default_random_engine generator;
	normal_distribution<double> distribution(5.0,2.0);

	int p[10]={};

	for (int i=0; i<nrolls; ++i) {
		double number = distribution(generator);
		cout << number << endl;
	}

	string baseURL = "https://api.bitfinex.com/v1";
	string type = "book";
	string symbol = "ltcbtc";
	string URL = baseURL + "/" + type + "/" + symbol;
	CURL *curl;
	CURLcode res;
	string readBuffer;
	Json::Value root;
	Json::Reader reader;

	curl = curl_easy_init();
	if(curl) {
		curl_easy_setopt(curl, CURLOPT_URL, URL.c_str());
		/* example.com is redirected, so we tell libcurl to follow redirection */ 
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		res = curl_easy_perform(curl);
		curl_easy_cleanup(curl);
		bool parsingSuccessful = reader.parse( readBuffer.c_str(), root );     //parse process
		if ( !parsingSuccessful )
		{
			cout << "Failed to parse" << reader.getFormattedErrorMessages();
			return 0;
		}
	}

	cout << root["bids"].size() << endl;
	return 0;

	for (Json::Value::ArrayIndex i = 0; i != root["bids"].size(); i++) {
		cout << root["bids"][i] << endl;
	}

	return 0;
}
