import requests
import json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import stats

URL = "https://api.bitfinex.com/v1"

def orderbook(symbol='btcusd'): # get the full order book.
	r = requests.get(URL + "/book/" + symbol, verify=True)
	rep = r.json()

	return rep

def stats(symbol='btcusd'): # Various statistics about the requested pairs.

	r = requests.get(URL + "/stats/" + symbol, verify=True) # <== UPDATED TO LATEST VERSION OF BFX!
	rep = r.json()
	return rep

	try:
		rep['volume']
	except KeyError:
		return rep['message']


scrape = False

if scrape:
	driver = webdriver.Firefox()
	driver.get("https://www.bitfinex.com/order_book/ltcbtc")

	bid_table = driver.find_element_by_class_name("sell").find_element_by_tag_name("tbody")
	ask_table = driver.find_element_by_class_name("buy").find_element_by_tag_name("tbody")
	orders = []

	for row in bid_table.find_elements_by_tag_name("tr"):
		count = int(row.find_element_by_class_name("col-info").text.replace(',',''))
		amount = float(row.find_element_by_class_name("depth-value-amt").text.replace(',',''))
		total = float(row.find_element_by_class_name("depth-value-cml").text.replace(',',''))
		price = float(row.find_element_by_class_name("price").text.replace(',',''))
		for i in range(0, count):
			orders.append(price)

	for row in ask_table.find_elements_by_tag_name("tr"):
		count = int(row.find_element_by_class_name("col-info").text.replace(',',''))
		amount = float(row.find_element_by_class_name("depth-value-amt").text.replace(',',''))
		total = float(row.find_element_by_class_name("depth-value-cml").text.replace(',',''))
		price = float(row.find_element_by_class_name("price").text.replace(',',''))
		for i in range(0, count):
			orders.append(price)
else:
	orders = []
	book = orderbook('ltcbtc')
	for data in book['bids']:
		for i in range(0, int(round(float(data['amount'])))):
			orders.append(float(data['price']))

	for data in book['asks']:
		for i in range(0, int(round(float(data['amount'])))):
			orders.append(float(data['price']))
	



orders.sort()

mu, std = norm.fit(orders)
print(mu, std, len(orders))
orderSize = 10000

for i in range(0, orderSize):
	mu, std = norm.fit(orders)
	value = np.random.normal(mu, std, size=None)
	match = False
	for idx, x in enumerate(orders):
		if idx+1 != len(orders):
			if orders[idx] >= value and orders[idx+1] > value:
				match = True
				if mu > value:
#					print("buy for " + str(orders[idx]))
					del orders[idx]
				else:	
#					print("sell for " + str(orders[idx+1]))
					del orders[idx+1]
				break
	if not match:
		orders.append(value)
		

mu, std = norm.fit(orders)
print(mu, std, len(orders))

