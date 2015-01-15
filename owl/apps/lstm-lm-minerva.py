#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is a simplified implementation of the LSTM language model (by Graham Neubig)
#
#  LSTM Neural Networks for Language Modeling
#  Martin Sundermeyer, Ralf Schlüter, Hermann Ney
#  InterSpeech 2012
# 
# The structure of the model is extremely simple. At every time step we
# read in the one-hot vector for the previous word, and predict the next word.
# Most of the learning code is based on the full-gradient update for LSTMs
#
#  Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures
#  Alex Graves and Jurgen Schmidhuber
#  Neural Networks 2005
#
# Note that this code is optimized for simplicity, not speed or accuracy, so it will
# be slow, and not necessarily get excellent performance. Also, it has not been checked all that
# carefully, but the likelihood does seem to be going down, so it's probably ok?

from collections import defaultdict
import sys
import math
import time
import numpy as np
from scipy import linalg
from scipy.special import expit         # Vectorized sigmoid function
import owl
from owl.conv import *
import owl.elewise as ele

class LSTMModel:

	def __init__(self, input_size, hidden_size, output_size):
		self.Layers = [input_size, hidden_size, output_size]
		# Recurrent weights: take x_t, h_{t-1}, and bias unit
		# and produce the 3 gates and the input to cell signal

		self.ig_weight_data = owl.randn([self.Layers[0], self.Layers[1]], 0.0, 0.1)
		self.fg_weight_data = owl.randn([self.Layers[0], self.Layers[1]], 0.0, 0.1)
		self.og_weight_data = owl.randn([self.Layers[0], self.Layers[1]], 0.0, 0.1)
		self.ff_weight_data = owl.randn([self.Layers[0], self.Layers[1]], 0.0, 0.1)

		self.ig_weight_prev = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)
		self.fg_weight_prev = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)
		self.og_weight_prev = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)
		self.ff_weight_prev = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)

		self.ig_weight_bias = owl.zeros([self.Layers[1], 1])
		self.fg_weight_bias = owl.zeros([self.Layers[1], 1])
		self.og_weight_bias = owl.zeros([self.Layers[1], 1])
		self.ff_weight_bias = owl.zeros([self.Layers[1], 1])

		# Decoder weights (e.g. mapping to vocabulary)
		self.decoder_weights = owl.randn([self.Layers[1], self.Layers[2]], 0.0, 0.1) # decoder
		self.decoder_bias = owl.zeros([output_size, 1])


def LSTM_init():

	# First read in the input
	wids = defaultdict(lambda: len(wids))
	wids['<BIAS>'] = 0
	wids['<s>'] = 1
	train_sents = []
	test_sents = []
	train_words = 0
	test_words = 0

	fin = open("./train")
	for line in fin:
		wordlist = ("<s> %s <s>" % line.strip()).split(' ')
		wordlist_id = [wids[w] for w in wordlist]
		train_words += len(wordlist) - 2
		train_sents.append(wordlist_id)

	fin = open("./test")
	for line in fin:
		wordlist = ("<s> %s <s>" % line.strip()).split(' ')
		wordlist_id = [wids[w] for w in wordlist]
		test_words += len(wordlist) - 2
		test_sents.append(wordlist_id)

	# Define input-dependent variables
	N = 10 # hidden units
	vocab_size = len(wids)       # Vocabulary size
	print "K", vocab_size, "words", train_words, test_words

	return LSTMModel(vocab_size, N, vocab_size), train_sents, test_sents, vocab_size, train_words, test_words

def LSTM_train(model, sents, vocab_size, words, NUM_EPOCHS = 100, tanhC_version = 1):

	# Constants
	ALPHA = 1             # Learning rate
	N = 10                # Number of units
	learning_rate = 1

	K = vocab_size       # Vocabulary size

	# For each epoch
	last_ll = 1e99
	last_time = time.time()
	for epoch_id in range(1, NUM_EPOCHS+1):
		epoch_ll = 0
		# For each sentence
		for sent_id, sent in enumerate(sents):
			#print "sent_id",sent_id
			#print "sent", sent
			#print "sents", sents
			##### Initialize activations #####
			Tau = len(sent)
			sent_ll = 0 # Sentence log likelihood
			batch_size = Tau

			data = [None] * Tau
			prev = [None] * Tau
			embed = np.zeros((K, 1))
			embed[sent[0]] = 1
			data[0] = owl.from_numpy(embed).trans()

			Hout = [None] * Tau
			Hout[0] = owl.zeros([N, 1])

			act_ig = [None] * Tau
			act_fg = [None] * Tau
			act_og = [None] * Tau
			act_ff = [None] * Tau

			C = [None] * Tau
			C[0] = owl.zeros([N, 1])
			Ym = [None] * Tau
			dY = [None] * Tau

			dBd = owl.zeros([model.Layers[2], 1]) #dY.sum(0)
			dWd = owl.zeros([model.Layers[1], model.Layers[2]]) #Hout.transpose().dot(dY)
			dHout = [None] * Tau #dY.dot(model.decoder_weights.transpose())

			##### Forward pass #####
			# For each time step
			for t in range(1, Tau):
				prev[t] = Hout[t - 1]
				embed = np.zeros((K, 1))
				embed[sent[t]] = 1
				data[t] = owl.from_numpy(embed).trans()

				act_ig[t] = model.ig_weight_data.trans() * data[t - 1] + model.ig_weight_prev.trans() * prev[t] + model.ig_weight_bias
				act_fg[t] = model.fg_weight_data.trans() * data[t - 1] + model.fg_weight_prev.trans() * prev[t] + model.fg_weight_bias
				act_og[t] = model.og_weight_data.trans() * data[t - 1] + model.og_weight_prev.trans() * prev[t] + model.og_weight_bias
				act_ff[t] = model.ff_weight_data.trans() * data[t - 1] + model.ff_weight_prev.trans() * prev[t] + model.ff_weight_bias

				act_ig[t] = ele.sigm(act_ig[t])
				act_fg[t] = ele.sigm(act_fg[t])
				act_og[t] = ele.sigm(act_og[t])
				act_ff[t] = ele.tanh(act_ff[t])

				C[t] = ele.mult(act_ig[t], act_ff[t]) + ele.mult(act_fg[t], C[t - 1])

				if tanhC_version:
					Hout[t] = ele.mult(act_og[t], ele.tanh(C[t]))
				else:
					Hout[t] = ele.mult(act_og[t], C[t])
				Ym[t] = softmax(model.decoder_weights.trans() * Hout[t] + model.decoder_bias)

				dY[t] = data[t] - Ym[t]
				dBd += dY[t] / batch_size
				dWd += Hout[t] * dY[t].trans() / batch_size
				dHout[t] = model.decoder_weights * dY[t]

				#print "Y_0[t]",Y_o[t]
				#print "Y_o[t][sent[t]]",Y_o[t][sent[t]]
				#print np.sum(output.to_numpy())
				# output = Ym[t].trans() * data[t]
				# sent_ll += math.log10( max(np.sum(output.to_numpy()),1e-20) )
			##### Initialize gradient vectors #####
			for t in range(1, Tau):
				output = Ym[t].trans() * data[t]
				sent_ll += math.log10( max(np.sum(output.to_numpy()),1e-20) )

			sen_ig = [None] * Tau
			sen_fg = [None] * Tau
			sen_og = [None] * Tau
			sen_ff = [None] * Tau

			weight_update_ig_data = owl.zeros([model.Layers[0], model.Layers[1]])
			weight_update_ig_prev = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_ig_bias = owl.zeros([model.Layers[1], 1])

			weight_update_fg_data = owl.zeros([model.Layers[0], model.Layers[1]])
			weight_update_fg_prev = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_fg_bias = owl.zeros([model.Layers[1], 1])

			weight_update_og_data = owl.zeros([model.Layers[0], model.Layers[1]])
			weight_update_og_prev = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_og_bias = owl.zeros([model.Layers[1], 1])

			weight_update_ff_data = owl.zeros([model.Layers[0], model.Layers[1]])
			weight_update_ff_prev = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_ff_bias = owl.zeros([model.Layers[1], 1])

			dHin = owl.zeros([model.Layers[1], model.Layers[1]])
			dC = [None] * Tau
			for t in xrange(Tau):
				dC[t] = owl.zeros(C[t].shape)

			# Calculate the error and add it
			for t in reversed(range(1, len(sent))):
				#print "sent",sent
				#print "t",t
				if tanhC_version:
					tanhCt = ele.tanh(C[t])
					sen_og[t] = ele.mult(tanhCt, dHout[t])
					dC[t] += ele.mult((1 - ele.mult(tanhCt, tanhCt)), ele.mult(act_og[t], dHout[t]))
				else:
					sen_og[t] = ele.mult(C[t], dHout[t])
					dC[t] += ele.mult(act_og[t], dHout[t])

				sen_fg[t] = owl.zeros([model.Layers[1], 1])
				if t > 0:
					sen_fg[t] = ele.mult(C[t - 1], dC[t])
					dC[t - 1] += ele.mult(act_og[t], dC[t])
				sen_ig[t] = ele.mult(act_ff[t], dC[t])
				sen_ff[t] = ele.mult(act_ig[t], dC[t])

				# backprop activation functions
				sen_ff[t] = ele.mult((1 - ele.mult(act_ff[t], act_ff[t])), sen_ff[t])
				sen_ig[t] = ele.mult(ele.mult(act_ig[t], (1.0 - act_ig[t])), sen_ig[t])
				sen_fg[t] = ele.mult(ele.mult(act_fg[t], (1.0 - act_fg[t])), sen_fg[t])
				sen_og[t] = ele.mult(ele.mult(act_og[t], (1.0 - act_og[t])), sen_og[t])

				# backprop matrix multiply
				weight_update_ig_data += data[t] * sen_ig[t].trans()
				weight_update_ig_prev += prev[t] * sen_ig[t].trans()
				weight_update_fg_bias += sen_ig[t] # sen_ig[t].sum(0 or 1)

				weight_update_fg_data += data[t] * sen_fg[t].trans()
				weight_update_fg_prev += prev[t] * sen_fg[t].trans()
				weight_update_fg_bias += sen_fg[t]

				weight_update_og_data += data[t] * sen_og[t].trans()
				weight_update_og_prev += prev[t] * sen_og[t].trans()
				weight_update_og_bias += sen_og[t]

				weight_update_ff_data += data[t] * sen_ff[t].trans()
				weight_update_ff_prev += prev[t] * sen_ff[t].trans()
				weight_update_ff_bias += sen_ff[t]

				if t > 1:
					dHout[t - 1] += model.ig_weight_prev.trans() * sen_ig[t]
					dHout[t - 1] += model.fg_weight_prev.trans() * sen_fg[t]
					dHout[t - 1] += model.og_weight_prev.trans() * sen_og[t]
					dHout[t - 1] += model.ff_weight_prev.trans() * sen_ff[t]

			# normalize the gradients
			# dWLSTM /= batch_size
			weight_update_ig_prev /= batch_size
			weight_update_ig_data /= batch_size
			weight_update_ig_bias /= batch_size

			weight_update_fg_prev /= batch_size
			weight_update_fg_data /= batch_size
			weight_update_fg_bias /= batch_size

			weight_update_og_prev /= batch_size
			weight_update_og_data /= batch_size
			weight_update_og_bias /= batch_size

			weight_update_ff_prev /= batch_size
			weight_update_ff_data /= batch_size
			weight_update_ff_bias /= batch_size

			# weight update
			model.ig_weight_prev += learning_rate * weight_update_ig_prev
			model.ig_weight_data += learning_rate * weight_update_ig_data
			model.ig_weight_bias += learning_rate * weight_update_ig_bias

			model.fg_weight_prev += learning_rate * weight_update_fg_prev
			model.fg_weight_data += learning_rate * weight_update_fg_data
			model.fg_weight_bias += learning_rate * weight_update_fg_bias

			model.og_weight_prev += learning_rate * weight_update_og_prev
			model.og_weight_data += learning_rate * weight_update_og_data
			model.og_weight_bias += learning_rate * weight_update_og_bias

			model.ff_weight_prev += learning_rate * weight_update_ff_prev
			model.ff_weight_data += learning_rate * weight_update_ff_data
			model.ff_weight_bias += learning_rate * weight_update_ff_bias

			model.decoder_weights += learning_rate * dWd
			model.decoder_bias += learning_rate * dBd

			# Print results
			epoch_ll += sent_ll
			# print(" Sentence %d LL: %f" % (sent_id, sent_ll))
		epoch_ent = epoch_ll*(-1) / words
		epoch_ppl = 10 ** epoch_ent
		cur_time = time.time()
		print("Epoch %d (alpha=%f) PPL=%f" % (epoch_id, learning_rate, epoch_ppl))
		print "  time consumed:", cur_time - last_time
		if last_ll > epoch_ll:
			learning_rate /= 2.0
		last_ll = epoch_ll
		last_time = cur_time

def LSTM_test(model, sents, vocab_size, words, tanhC_version = 1):

	N = 10
	K = vocab_size

	test_ll = 0
	# For each sentence
	for sent in enumerate(sents):
		#print "sent_id",sent_id
		#print "sent", sent
		#print "sents", sents
		##### Initialize activations #####
		Tau = len(sent)
		sent_ll = 0 # Sentence log likelihood
		batch_size = Tau

		data = [None] * Tau
		prev = [None] * Tau
		embed = np.zeros((K, 1))
		embed[sent[0]] = 1
		data[0] = owl.from_numpy(embed).trans()

		Hout = [None] * Tau
		Hout[0] = owl.zeros([N, 1])

		act_ig = [None] * Tau
		act_fg = [None] * Tau
		act_og = [None] * Tau
		act_ff = [None] * Tau

		C = [None] * Tau
		C[0] = owl.zeros([N, 1])
		Ym = [None] * Tau
		dY = [None] * Tau

		##### Forward pass #####
		# For each time step
		for t in range(1, Tau):
			prev[t] = Hout[t - 1]
			embed = np.zeros((K, 1))
			embed[sent[t]] = 1
			data[t] = owl.from_numpy(embed).trans()

			act_ig[t] = model.ig_weight_data.trans() * data[t - 1] + model.ig_weight_prev.trans() * prev[t] + model.ig_weight_bias
			act_fg[t] = model.fg_weight_data.trans() * data[t - 1] + model.fg_weight_prev.trans() * prev[t] + model.fg_weight_bias
			act_og[t] = model.og_weight_data.trans() * data[t - 1] + model.og_weight_prev.trans() * prev[t] + model.og_weight_bias
			act_ff[t] = model.ff_weight_data.trans() * data[t - 1] + model.ff_weight_prev.trans() * prev[t] + model.ff_weight_bias

			act_ig[t] = ele.sigm(act_ig[t])
			act_fg[t] = ele.sigm(act_fg[t])
			act_og[t] = ele.sigm(act_og[t])
			act_ff[t] = ele.tanh(act_ff[t])

			C[t] = ele.mult(act_ig[t], act_ff[t]) + ele.mult(act_fg[t], C[t - 1])

			if tanhC_version:
				Hout[t] = ele.mult(act_og[t], ele.tanh(C[t]))
			else:
				Hout[t] = ele.mult(act_og[t], C[t])
			Ym[t] = softmax(model.decoder_weights.trans() * Hout[t] + model.decoder_bias)

			#print "Y_0[t]",Y_o[t]
			#print "Y_o[t][sent[t]]",Y_o[t][sent[t]]
			output = Ym[t].trans() * data[t]
			test_ll += math.log10( max(np.sum(output.to_numpy()),1e-20) )

	print test_ll
	test_ent = test_ll * (-1) / words
	test_ppl = 10 ** test_ent

	print("Test PPL = %f" % (test_ppl))

if __name__ == '__main__':
	owl.initialize(sys.argv)
	gpu = owl.create_gpu_device(0)
	owl.set_device(gpu)
	model, train_sents, test_sents, vocab_size, train_words, test_words = LSTM_init()
	LSTM_train(model, train_sents, vocab_size, train_words, 100)
	LSTM_test(model, test_sents, vocab_size, test_words)
