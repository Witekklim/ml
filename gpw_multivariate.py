
"""
attributes:
	X: pandas DataFrame, 


"""
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import date


from sklearn.preprocessing import MinMaxScaler


class gpw_rnn_multivariate:
	def __init__(self, companies = [], currencies = [], TARGET = 0, future_target = 1, today_date = None):
		self.companies = companies
		self.currencies = currencies
		
		self.TARGET = TARGET
		self.future_target = future_target

		self.today = date.today().strftime("%d_%m_%Y") if today_date is None else today_date

		self.load_data()

	def load_data(self):
		print(f'loading data for comapnies: {self.companies}')
		
		l = 1e6
		datas = []

		for currency in self.currencies:
			print(f'collecting data for {currency}')
			filename = f'data/{currency}_{self.today}.csv'
			try:
				data = pd.read_csv(filename)
				print('loaded data from csv')
			except FileNotFoundError:
				data_url = f'https://stooq.pl/q/d/l/?s={currency}&i=d'  # eurpln
				data = pd.read_csv(data_url)
				print('loaded data from url')
				data.to_csv(filename)
				print('saved data to csv')

			X = data['Zamkniecie']
			X.index = data['Data']
			if len(X) < l:
				l = len(X)
				print(f'new shortest data time for {currency}, l = {l}')
			datas.append(X)

		for company in self.companies:
			print(f'collecting data for {company}')
			
			filename = f'data/{company}_{self.today}.csv'
			try:
				data = pd.read_csv(filename)
				print('loaded data from csv')
			except FileNotFoundError:
				data_url = f'https://stooq.pl/q/d/l/?s={company}&i=d'
				data = pd.read_csv(data_url)
				print('loaded data from url')
				data.to_csv(filename)
				print('saved data to csv')

			X = data['Zamkniecie']
			X.index = data['Data']
			if len(X) < l:
				l = len(X)
				print(f'new shortest data time for {company}, l = {l}')
			datas.append(X)

		self.X = pd.DataFrame()
		for data, company in zip(datas, companies):
			self.X[f'{company}'] = data.iloc[-l:-1]
		self.X = self.X.fillna(method ='pad')
		#X.plot(subplots = True)
		#plt.show()

	def prepare_data(self, past_history = 30, STEP = 1, train_split_frac = 0.75):
		
		TRAIN_SPLIT = int(train_split_frac * len(self.X))
		dataset = self.X.values
		data_mean = dataset[:TRAIN_SPLIT].mean(axis = 0)
		data_std = dataset[:TRAIN_SPLIT].std(axis = 0)
		dataset = (dataset - data_mean)/data_std

		self.X_train , self.y_train = self.multivariate_data(dataset, dataset[:, self.TARGET], 0, TRAIN_SPLIT, 
			past_history, self.future_target, STEP, single_step=True)

		self.X_test , self.y_test = self.multivariate_data(dataset, dataset[:, self.TARGET], TRAIN_SPLIT, None, 
			past_history, self.future_target, STEP, single_step=True)


	def fit_model(self, LSTM_SIZE= 32, BATCH_SIZE = 128, BUFFER_SIZE = 200, EVAL_INTERVAL = 200, EPOCHS = 10):
		data_train = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
		data_train = data_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

		data_test = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
		print(f'is before repeat the same as after? {data_test.batch(BATCH_SIZE) == data_test.batch(BATCH_SIZE).repeat()}')
		
		print(f'before BATCHING: {data_test}')
		print(f'before: {data_test.batch(BATCH_SIZE)}')
		data_test = data_test.batch(BATCH_SIZE).repeat()
		print(f'after: {data_test}')

		model = self.create_LSTM(self.X_train.shape[-2:], size = LSTM_SIZE, layers = 2, dropout = 0.2)

#		model = tf.keras.models.Sequential()
#		model.add(tf.keras.layers.LSTM(LSTM_SIZE, return_sequences=True, input_shape = self.X_train.shape[-2:]))
#		model.add(tf.keras.layers.Dropout(0.2))
#		model.add(tf.keras.layers.LSTM(LSTM_SIZE))
#		model.add(tf.keras.layers.Dense(1))
#		model.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss = 'mae')   #   or optimizers.RMSprop()
#		print(model.summary())

		self.history = model.fit(data_train, epochs = EPOCHS, steps_per_epoch = EVAL_INTERVAL,
			validation_data = data_test, validation_steps = 50)

		self.data_test = data_test
		self.model = model

		for x,y in data_test.take(3):
			print(f'the shapes of x,y are: {x.numpy().shape, y.numpy().shape}')
			plot = self.show_plot([x[0][:, self.TARGET].numpy(), y[0].numpy(),
				model.predict(x)[0]], self.future_target, 'single step prediction')
			plot.show()
		

	def plot_train_history(self, title):
		loss = self.history.history['loss']
		val_loss = self.history.history['val_loss']
		epochs = range(len(loss))

		plt.figure()
		plt.plot(epochs, loss, 'b', label = 'training loss')
		plt.plot(epochs, val_loss, 'r', label = 'validation loss')
		plt.title(title)
		plt.legend()
		plt.show()

	# +++++++++++++++++++++ STATIC METHODS ====================
	@staticmethod
	def create_LSTM(input_shape, size = 32, layers = 2, dropout = 0.2, optimizer = tf.keras.optimizers.Adam(1e-4), loss = 'mae' ):
		
		model = tf.keras.models.Sequential()
		
		if layers == 1:
			model.add(tf.keras.layers.LSTM(size, input_shape = input_shape))
		else:
			model.add(tf.keras.layers.LSTM(size, return_sequences = True, input_shape = input_shape))
		for layers in range(layers - 1):
			model.add(tf.keras.layers.Dropout(dropout))
			model.add(tf.keras.layers.LSTM(size))

		model.add(tf.keras.layers.Dense(1))
		model.compile(optimizer = optimizer, loss = loss)   #   or optimizers.RMSprop()
		print(model.summary())
		return model

	@staticmethod
	def create_time_steps(l):
		return list(range(-l,0))
		
	@staticmethod
	def show_plot(X, delta, title):
		labels = ['history', 'true future', 'prediction']
		markers = ['.-', 'rx', 'go']
		time_steps = gpw_rnn_multivariate.create_time_steps(X[0].shape[0])
		future = delta if delta else 0
		plt.title(title)
		for i,x in enumerate(X):
			if i:
				plt.plot(future, X[i], markers[i], markersize = 10, label = labels[i])
			else:
				plt.plot(time_steps, X[i].flatten(), markers[i], label = labels[i])
		plt.legend()
		return plt

	@staticmethod
	def multivariate_data(X, target, i_0, i_n, history_size, target_size, step, single_step = False):
		data = []
		labels = []
		i_0 += history_size
		if i_n is None:
			i_n = len(X) - target_size
		for i in range(i_0, i_n):
			indices = range(i - history_size, i , step)
			data.append(X[indices])

			if single_step:
				labels.append(target[i + target_size])
			else:
				labels.append(target[i:i+target_size])
		return np.array(data), np.array(labels)

if __name__ == '__main__':
	companies = ['bdx', 'ago', 'att', 'bnp', 'ccc', 'cmr', 'cps','dnp', 'alr', 'cdr', 'pko', 'cie', 'ena', 'eng', 'gtn', 'jsw', 'kgh','lpp']
	
	companies = ['ccc', 'cdr']
	gpw = gpw_rnn_multivariate(companies, TARGET = 0, future_target = 1, today_date = '10_07_2020')
	gpw.prepare_data(past_history = 30)
	gpw.fit_model(LSTM_SIZE = 20, BATCH_SIZE = 128, BUFFER_SIZE = 200, EPOCHS = 15)
	gpw.plot_train_history('lstm1')
