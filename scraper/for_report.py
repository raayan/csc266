import numpy as np
from scipy.stats import norm
from scipy.stats import stats

orders = #populate orderbook from exchange

orders.sort()

mu, std = norm.fit(orders)
print(mu, std, len(orders))

for i in range(0, 1000):
	mu, std = norm.fit(orders)
	value = np.random.normal(mu, std, size=None)
	match = False
	for idx, x in enumerate(orders):
		if idx+1 != len(orders):
			if orders[idx] >= value and orders[idx+1] > value:
				match = True
				if mu > value:
					del orders[idx]
				else:	
					del orders[idx+1]
				break
	if not match:
		orders.append(value)
mu, std = norm.fit(orders)


print(mu, std, len(orders))
