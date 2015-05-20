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

	def initw(n, d):
		magic_number = 0.3
		npa = (np.random.rand(n, d) * 2 - 1) * magic_number # U[-0.1, 0.1]
		return owl.from_numpy(npa).trans()

	def __init__(self, vocab_size, input_size, hidden_size):
		output_size = vocab_size
		self.Layers = [input_size, hidden_size, output_size]
                print 'Model size:', self.Layers
		# Recurrent weights: take x_t, h_{t-1}, and bias unit
		# and produce the 3 gates and the input to cell signal

		# self.WIFOG = owl.randn([self.Layers[0] + self.Layers[1], self.Layers[1] * 4], 0.0, 0.1)
		# self.BIFOG = owl.zeros([self.Layers[1] * 4, 1])

		self.ig_weight_data = owl.randn([self.Layers[1], self.Layers[0]], 0.0, 0.1)
		self.fg_weight_data = owl.randn([self.Layers[1], self.Layers[0]], 0.0, 0.1)
		self.og_weight_data = owl.randn([self.Layers[1], self.Layers[0]], 0.0, 0.1)
		self.ff_weight_data = owl.randn([self.Layers[1], self.Layers[0]], 0.0, 0.1)

		self.ig_weight_prev = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)
		self.fg_weight_prev = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)
		self.og_weight_prev = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)
		self.ff_weight_prev = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)

		self.ig_weight_cell = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)
		self.fg_weight_cell = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)
		self.og_weight_cell = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)
		self.ff_weight_cell = owl.randn([self.Layers[1], self.Layers[1]], 0.0, 0.1)

		self.ig_weight_bias = owl.zeros([self.Layers[1], 1])
		self.fg_weight_bias = owl.zeros([self.Layers[1], 1])
		self.og_weight_bias = owl.zeros([self.Layers[1], 1])
		self.ff_weight_bias = owl.zeros([self.Layers[1], 1])

		# Decoder weights (e.g. mapping to vocabulary)
		self.decoder_weights = owl.randn([self.Layers[2], self.Layers[1]], 0.0, 0.1) # decoder
		self.decoder_bias = owl.zeros([output_size, 1])

		self.emb_weight = [None] * vocab_size
		for i in range(vocab_size):
			self.emb_weight[i] = owl.randn([input_size, 1], 0.0, 0.1)


def LSTM_init():

	# First read in the input
	wids = defaultdict(lambda: len(wids))
	wids['<bos>'] = 0 # begin of sentence
	wids['<eos>'] = 1 # end of sentence
	train_sents = []
	test_sents = []
	train_words = 0
	test_words = 0

	fin_train = open("./train")
	for line in fin_train:
		wordlist = ("<bos> %s <eos>" % line.strip()).split(' ')
		wordlist_id = [wids[w] for w in wordlist]
		train_words += len(wordlist) - 2
		train_sents.append(wordlist_id)

	fin_test = open("./test")
	for line in fin_test:
		wordlist = ("<bos> %s <eos>" % line.strip()).split(' ')
		wordlist_id = []
		for w in wordlist:
			if wids.has_key(w):
				wordlist_id.append(wids[w])
				test_words += 1
		test_sents.append(wordlist_id)

	# Define input-dependent variables
	N = 100 # hidden units
	D = N # embedding
	vocab_size = len(wids)       # Vocabulary size
	print "K", vocab_size, "words", train_words, test_words

	return LSTMModel(vocab_size, D, N), train_sents, test_sents, train_words, test_words

def LSTM_train(model, sents, words, learning_rate, EPOCH, tanhC_version = 1):

	# Constants
	N = model.Layers[1]       # Number of units
	K = model.Layers[2]       # Vocabulary size

	last_time = time.time()
	# For each epoch
	for epoch_id in range(1, EPOCH + 1):
		epoch_ll = 0
		# For each sentence
		for sent_id, sent in enumerate(sents):
			#print sent_id
			#print "sent", sent
			#print "sents", sents
			##### Initialize activations #####

			Tau = len(sent)
			sent_ll = 0 # Sentence log likelihood

			data = [None] * Tau

			Hout = [None] * Tau
			Hout[0] = owl.zeros([N, 1])

			act_ig = [None] * Tau
			act_fg = [None] * Tau
			act_og = [None] * Tau
			act_ff = [None] * Tau

			C = [None] * Tau
			C[0] = owl.zeros([N, 1])
			dY = [None] * Tau

			dBd = owl.zeros([model.Layers[2], 1]) #dY.sum(0)
			dWd = owl.zeros([model.Layers[2], model.Layers[1]]) 
			dHout = [None] * Tau #dY.dot(model.decoder_weights.transpose())
			dEmb = [None] * Tau

			##### Forward pass #####
			# For each time step

			for t in range(1, Tau):
				# predict the (t+1)'th word from the t'th word
				data[t] = model.emb_weight[sent[t - 1]]
				NVector = np.zeros((K, 1))
				NVector[sent[t]] = 1
				target = owl.from_numpy(NVector).trans()

				act_ig[t] = model.ig_weight_data * data[t] + model.ig_weight_prev * Hout[t - 1] + model.ig_weight_cell * C[t - 1] + model.ig_weight_bias
				act_ig[t] = ele.sigm(act_ig[t])

				act_fg[t] = model.fg_weight_data * data[t] + model.fg_weight_prev * Hout[t - 1] + model.fg_weight_cell * C[t - 1] + model.fg_weight_bias
				act_fg[t] = ele.sigm(act_fg[t])

				act_ff[t] = model.ff_weight_data * data[t] + model.ff_weight_prev * Hout[t - 1] + model.ff_weight_bias
				act_ff[t] = ele.tanh(act_ff[t])

				C[t] = ele.mult(act_ig[t], act_ff[t]) + ele.mult(act_fg[t], C[t - 1])

				act_og[t] = model.og_weight_data * data[t] + model.og_weight_prev * Hout[t - 1] + model.og_weight_cell * C[t] + model.og_weight_bias
				act_og[t] = ele.sigm(act_og[t])

				if tanhC_version:
					Hout[t] = ele.mult(act_og[t], ele.tanh(C[t]))
				else:
					Hout[t] = ele.mult(act_og[t], C[t])

				Y = softmax(model.decoder_weights * Hout[t] + model.decoder_bias)

				# BP to Hout
				dY[t] = Y - target
				dBd += dY[t]
				dWd += dY[t] * Hout[t].trans()
				dHout[t] = model.decoder_weights.trans() * dY[t]

				# evaluation
				output = Y.to_numpy()			# Can directly get a single element from Y
				# print output[0, sent[t]]
				sent_ll += math.log(max(output[0, sent[t]],1e-20), 2)

				#print "Y_0[t]",Y_o[t]
				#print "Y_o[t][sent[t]]",Y_o[t][sent[t]]
				#print np.sum(output.to_numpy())
				# output = Ym[t].trans() * data[t]
				# sent_ll += math.log10( max(np.sum(output.to_numpy()),1e-20) )
			##### Initialize gradient vectors #####
				

			weight_update_ig_data = owl.zeros([model.Layers[1], model.Layers[0]])
			weight_update_ig_prev = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_ig_cell = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_ig_bias = owl.zeros([model.Layers[1], 1])

			weight_update_fg_data = owl.zeros([model.Layers[1], model.Layers[0]])
			weight_update_fg_prev = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_fg_cell = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_fg_bias = owl.zeros([model.Layers[1], 1])

			weight_update_og_data = owl.zeros([model.Layers[1], model.Layers[0]])
			weight_update_og_prev = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_og_cell = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_og_bias = owl.zeros([model.Layers[1], 1])

			weight_update_ff_data = owl.zeros([model.Layers[1], model.Layers[0]])
			weight_update_ff_prev = owl.zeros([model.Layers[1], model.Layers[1]])
			weight_update_ff_bias = owl.zeros([model.Layers[1], 1])

			dC = [None] * Tau

			for t in xrange(Tau):
				dC[t] = owl.zeros(C[t].shape)

			# Calculate the error and add it
			for t in reversed(range(1, Tau)):
				#print "sent",sent
				#print "t",t

				# BP from og controled gate and og
				if tanhC_version:
					tanhC = ele.tanh(C[t])
					dTanhC = ele.mult(dHout[t], act_og[t])
					sen_og = ele.mult(dHout[t], tanhC)
					dC[t] += ele.mult((1 - ele.mult(tanhC, tanhC)), dTanhC)
				else:
					sen_og = ele.mult(C[t], dHout[t])
					dC[t] += ele.mult(act_og[t], dHout[t])

				# BP from og
				sen_og = ele.mult(ele.mult(act_og[t], (1.0 - act_og[t])), sen_og)
				dHout[t - 1] = model.og_weight_prev.trans() * sen_og
				dC[t] += model.og_weight_cell.trans() * sen_og
				dEmb[t] = model.og_weight_data.trans() * sen_og

				# BP from fg controled gate
				sen_fg = ele.mult(C[t - 1], dC[t])
				dC[t - 1] += ele.mult(act_fg[t], dC[t])
				
				# BP from ig controled gate
				sen_ig = ele.mult(act_ff[t], dC[t])
				sen_ff = ele.mult(act_ig[t], dC[t])
				sen_ff = ele.mult((1 - ele.mult(act_ff[t], act_ff[t])), sen_ff)
				dEmb[t] += model.ff_weight_data.trans() * sen_ff
				
				# BP from fg
				sen_fg = ele.mult(ele.mult(act_fg[t], (1.0 - act_fg[t])), sen_fg)
				dHout[t - 1] += model.fg_weight_prev.trans() * sen_fg
				dC[t - 1] += model.fg_weight_cell.trans() * sen_fg
				dEmb[t] += model.fg_weight_data.trans() * sen_fg

				# BP from ig
				sen_ig = ele.mult(ele.mult(act_ig[t], (1.0 - act_ig[t])), sen_ig)
				dHout[t - 1] += model.ig_weight_prev.trans() * sen_ig
				dC[t - 1] += model.ig_weight_cell.trans() * sen_ig
				dEmb[t] += model.ig_weight_data.trans() * sen_ig

				# derivatives on weight matrix and bias
				weight_update_ig_data += sen_ig * data[t].trans()
				weight_update_ig_prev += sen_ig * Hout[t - 1].trans()
				weight_update_ig_cell += sen_ig * C[t - 1].trans()
				weight_update_ig_bias += sen_ig

				weight_update_fg_data += sen_fg * data[t].trans()
				weight_update_fg_prev += sen_fg * Hout[t - 1].trans()
				weight_update_fg_cell += sen_fg * C[t - 1].trans()
				weight_update_fg_bias += sen_fg

				weight_update_og_data += sen_og * data[t].trans()
				weight_update_og_prev += sen_og * Hout[t - 1].trans()
				weight_update_og_cell += sen_og * C[t].trans()
				weight_update_og_bias += sen_og

				weight_update_ff_data += sen_ff * data[t].trans()
				weight_update_ff_prev += sen_ff * Hout[t - 1].trans()
				weight_update_ff_bias += sen_ff


			# normalize the gradients
			rate = learning_rate / Tau

			# weight update
			model.ig_weight_prev -= rate * weight_update_ig_prev
			model.ig_weight_data -= rate * weight_update_ig_data
			model.ig_weight_cell -= rate * weight_update_ig_cell
			model.ig_weight_bias -= rate * weight_update_ig_bias

			model.fg_weight_prev -= rate * weight_update_fg_prev
			model.fg_weight_data -= rate * weight_update_fg_data
			model.fg_weight_cell -= rate * weight_update_fg_cell
			model.fg_weight_bias -= rate * weight_update_fg_bias

			model.og_weight_prev -= rate * weight_update_og_prev
			model.og_weight_data -= rate * weight_update_og_data
			model.og_weight_cell -= rate * weight_update_og_cell
			model.og_weight_bias -= rate * weight_update_og_bias

			model.ff_weight_prev -= rate * weight_update_ff_prev
			model.ff_weight_data -= rate * weight_update_ff_data
			model.ff_weight_bias -= rate * weight_update_ff_bias

			model.decoder_weights -= rate * dWd
			model.decoder_bias -= rate * dBd

			for t in range(1, Tau):
				model.emb_weight[sent[t - 1]] -= rate * dEmb[t]

			# Print results
			epoch_ll += sent_ll
			# print(" Sentence %d LL: %f" % (sent_id, sent_ll))

			
		epoch_ent = epoch_ll * (-1) / words
		epoch_ppl = 2 ** epoch_ent
		cur_time = time.time()
		print("Epoch %d (alpha=%f) PPL=%f" % (epoch_id, learning_rate, epoch_ppl))
		print "  time consumed:", cur_time - last_time
		last_time = cur_time

	return model, learning_rate

def LSTM_test(model, sents, words, tanhC_version = 1):

	N = model.Layers[1]
	K = model.Layers[2]

	test_ll = 0
	# For each sentence
	for sent_id, sent in enumerate(sents):
		#print sent_id
		#print "sent", sent
		#print "sents", sents
		##### Initialize activations #####

		Tau = len(sent)
		sent_ll = 0 # Sentence log likelihood

		data = [None] * Tau

		Hout = [None] * Tau
		Hout[0] = owl.zeros([N, 1])

		act_ig = [None] * Tau
		act_fg = [None] * Tau
		act_og = [None] * Tau
		act_ff = [None] * Tau

		C = [None] * Tau
		C[0] = owl.zeros([N, 1])

		##### Forward pass #####
		# For each time step

		for t in range(1, Tau):
			# predict the (t+1)'th word from the t'th word
			data[t] = model.emb_weight[sent[t - 1]]

			act_ig[t] = model.ig_weight_data * data[t] + model.ig_weight_prev * Hout[t - 1] + model.ig_weight_cell * C[t - 1] + model.ig_weight_bias
			act_ig[t] = ele.sigm(act_ig[t])

			act_fg[t] = model.fg_weight_data * data[t] + model.fg_weight_prev * Hout[t - 1] + model.fg_weight_cell * C[t - 1] + model.fg_weight_bias
			act_fg[t] = ele.sigm(act_fg[t])

			act_ff[t] = model.ff_weight_data * data[t] + model.ff_weight_prev * Hout[t - 1] + model.ff_weight_bias
			act_ff[t] = ele.tanh(act_ff[t])

			C[t] = ele.mult(act_ig[t], act_ff[t]) + ele.mult(act_fg[t], C[t - 1])

			act_og[t] = model.og_weight_data * data[t] + model.og_weight_prev * Hout[t - 1] + model.og_weight_cell * C[t] + model.og_weight_bias
			act_og[t] = ele.sigm(act_og[t])

			if tanhC_version:
				Hout[t] = ele.mult(act_og[t], ele.tanh(C[t]))
			else:
				Hout[t] = ele.mult(act_og[t], C[t])

			Y = softmax(model.decoder_weights * Hout[t] + model.decoder_bias)

			# evaluation
			output = Y.to_numpy()			# Can directly get a single element from Y
			# print output[0, sent[t]]
			sent_ll += math.log(max(output[0, sent[t]],1e-20), 2)

		test_ll += sent_ll

	test_ent = test_ll * (-1) / words
	test_ppl = 2 ** test_ent

	print "Test PPL =", test_ppl

if __name__ == '__main__':
	#gpu = owl.create_gpu_device(1)
	cpu = owl.create_cpu_device()
	owl.set_device(cpu)
	model, train_sents, test_sents, train_words, test_words = LSTM_init()
	learning_rate = 0.1
	for i in range(5):
		model, learning_rate = LSTM_train(model, train_sents, train_words, learning_rate, 1)
		LSTM_test(model, test_sents, test_words)
