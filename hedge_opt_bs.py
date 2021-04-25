from keras.models import Model 
from keras.layers import Input, Dense, Subtract, Multiply, Lambda, Add
from keras.engine.topology import Layer
from keras import initializers as init
import keras.backend as K
import numpy as np
from numpy import log, exp, sqrt, ones, zeros, tanh, matmul, squeeze
from scipy.stats import norm

def bs(s0, strike, T, sigma):
    return s0 * norm.cdf((log(s0/strike) + 0.5 * T * sigma**2) / (sqrt(T) * sigma)) - strike * norm.cdf((log(s0 / strike) - 0.5 * T * sigma**2) / (sqrt(T) * sigma))

def delta_bs(s, k):
    return norm.cdf((log(s/strike) + 0.5 * (T - k * T / N) * sigma**2) / (sqrt(T - k * T / N) * sigma))

N, s0, strike, T, sigma = 30, 100, 100, 1.0, 0.2       

price_bs = bs(s0, strike, T, sigma)

m, d, n = 1, 2, 8 
layers = []
for j in range(N):
    for i in range(d):
        if i < d - 1:
            nodes = n
            layer = Dense(nodes, activation='tanh', trainable=True, kernel_initializer=init.RandomNormal(0, 0.1), bias_initializer=init.RandomNormal(0, 0), name=str(i)+str(j))
        else:
            nodes = m
            layer = Dense(nodes, activation='linear', trainable=True, kernel_initializer=init.RandomNormal(0, 0.1), bias_initializer=init.RandomNormal(0, 0), name=str(i)+str(j))
        layers = layers + [layer]

price = Input(shape=(m,))
hedge = Input(shape=(m,))
inputs = [price] + [hedge]

for j in range(N):
    strategy = price
    for k in range(d):
        strategy = layers[k + (j) * d](strategy) 
    incr = Input(shape=(m,))
    logprice = Lambda(lambda x: K.log(x))(price)
    logprice = Add()([logprice, incr])
    pricenew = Lambda(lambda x: K.exp(x))(logprice)
    priceincr = Subtract()([pricenew, price])
    hedgenew = Multiply()([strategy, priceincr])
    hedge = Add()([hedge, hedgenew]) 
    inputs = inputs + [incr]
    price = pricenew
payoff = Lambda(lambda x: 0.5 * (K.abs(x - strike) + x - strike) - price_bs)(price) 
outputs = Subtract()([payoff, hedge])

model_hedge = Model(inputs=inputs, outputs=outputs)
model_hedge.compile(optimizer='adam', loss='mean_squared_error')

def get_x(n):
    return [s0 * ones((n, m))] + [zeros((n, m))] + [np.random.normal(-(sigma)**2 * T / (2*N), sigma * sqrt(T) / sqrt(N), (n, m)) for i in range(N)]

n_train = 5 * 10**5
xtrain = get_x(n_train)
ytrain = zeros((n_train, 1))

n_test = 10**5
xtest = get_x(n_test)
ytest = zeros((n_test, 1))

model_hedge.fit(x=xtrain, y=ytrain, epochs=250, verbose=True, batch_size=100)

weights = model_hedge.get_weights()
def delta_nn(s, j):
    length = s.shape[0]
    g = zeros(length)
    for p in range(length):
        ghelper = tanh(s[p] * (weights[j * 2 * d]) + weights[j * 2 * d + 1])
        g[p] = np.sum(squeeze(weights[2 * (d - 1) + j * 2 * d]) * squeeze(ghelper))
        g[p] = g[p] + weights[2 * d - 1 + j * 2 * d]   
    return g
