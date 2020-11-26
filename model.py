import numpy as np

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

import tensorflow as tf

def softmax_cross_entropy_with_logits(y_true, y_pred):
	p = y_pred
	pi = y_true

	zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
	where = tf.equal(pi, zero)

	negatives = tf.fill(tf.shape(pi), -100.0) 
	p = tf.where(where, negatives, p)

	return tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)


class Residual_CNN():
	def __init__(self, reg_const, learning_rate, input_dim,  output_dim, hidden_layers):
		self.hidden_layers = hidden_layers
		self.num_layers = len(hidden_layers)
		self.reg_const = reg_const
		self.model = self._build_model(input_dim, output_dim, learning_rate)

	def predict(self, x):
		return self.model.predict(x)

	def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
		return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split = validation_split, batch_size = batch_size)

	def save(self, version):
		self.model.save('./model_' + '{0:0>4}'.format(version) + '.h5')

	def load(self, version):
		return load_model('./models_' + '{0:0>4}'.format(version) + '.h5', custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

	def _build_model(self, input_dim, output_dim, learning_rate, momentum=0.9):

		main_input = Input(shape = input_dim, name = 'main_input')

		x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

		if len(self.hidden_layers) > 1:
			for h in self.hidden_layers[1:]:
				x = self.residual_layer(x, h['filters'], h['kernel_size'])

		vh = self.value_head(x)
		ph = self.policy_head(x, output_dim)

		model = Model(inputs=[main_input], outputs=[vh, ph])
		model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
			optimizer=SGD(lr=learning_rate, momentum=momentum),	
			loss_weights={'value_head': 0.5, 'policy_head': 0.5}	
		)

		return model

	def residual_layer(self, input_block, filters, kernel_size):
		x = self.conv_layer(input_block, filters, kernel_size)

		x = Conv2D(
			filters = filters,
			kernel_size = kernel_size,
			data_format='channels_first',
			padding = 'same',
			use_bias=False,
			activation='linear',
			kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=1)(x)
		x = add([input_block, x])
		x = LeakyReLU()(x)

		return (x)

	def conv_layer(self, x, filters, kernel_size):
		x = Conv2D(
			filters = filters,
			kernel_size = kernel_size,
			data_format='channels_first',
			padding = 'same',
			use_bias=False,
			activation='linear',
			kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		return (x)

	def value_head(self, x):
		x = Conv2D(
			filters = 1,
			kernel_size = (1,1),
			data_format='channels_first',
			padding = 'same',
			use_bias=False,
			activation='linear',
			kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)


		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)
		x = Flatten()(x)

		x = Dense(
			20,
			use_bias=False,
			activation='linear',
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)

		x = LeakyReLU()(x)
		x = Dense(
			1,
			use_bias=False,
			activation='tanh',
			kernel_regularizer=regularizers.l2(self.reg_const),
			name = 'value_head'
		)(x)

		return (x)

	def policy_head(self, x, output_dim):

		x = Conv2D(
			filters = 2,
			kernel_size = (1,1),
			data_format='channels_first',
			padding = 'same',
			use_bias=False,
			activation='linear',
			kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		x = Dense(
			output_dim,
			use_bias=False,
			activation='linear',
			kernel_regularizer=regularizers.l2(self.reg_const),
			name = 'policy_head'
		)(x)

		return (x)
