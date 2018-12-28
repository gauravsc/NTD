import unicodedata
import string
import re
import random as rd
import time
import math
import cPickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from attention_model import *
import numpy as np
import tree_of_mesh as tom 


# max english words
num_english_words = 0



# encoder_test = EncoderRNN(10, 10, 2)
# decoder_test = AttnDecoderRNN('general', 10, 10, 2)
# print encoder_test 
# print decoder_test

# encoder_hidden = encoder_test.init_hidden()
# word_input = Variable(torch.LongTensor([1, 2, 3]))

# if USE_CUDA:
#     encoder_test.cuda()
#     word_input = word_input.cuda()

# encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

# word_inputs = Variable(torch.LongTensor([1, 2, 3]))
# decoder_attns = torch.zeros(1, 3, 3)
# decoder_hidden = encoder_hidden
# decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))

# if USE_CUDA:
#     decoder_test.cuda()
#     word_inputs = word_inputs.cuda()
#     decoder_context = decoder_context.cuda()

# for i in range(3):
#     decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(word_inputs[i], decoder_context, decoder_hidden, encoder_outputs)
#     print decoder_output.size(), decoder_hidden.size(), decoder_attn.size(), decoder_output 
#     decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().data


teacher_forcing_ratio = 0.5
clip = 5.0

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

	# Zero gradients of both optimizers
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0 # Added onto for each word

	# Get size of input and target sentences
	# print input_variable.size(), target_variable.size()
	input_length = input_variable.size()[0]
	target_length = target_variable.size()[0]

	# Run words through encoder
	encoder_hidden = encoder.init_hidden()
	encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
	
	# Prepare input and output variables
	decoder_input = Variable(torch.LongTensor([[code_2_index['SOS']]]))
	decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
	decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
	if USE_CUDA:
		decoder_input = decoder_input.cuda()
		decoder_context = decoder_context.cuda()

	# Choose whether to use teacher forcing
	use_teacher_forcing = random.random() < teacher_forcing_ratio
	if use_teacher_forcing:
		
		# Teacher forcing: Use the ground-truth target as the next input
		for di in range(target_length):
			decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
			loss += criterion(decoder_output, target_variable[di])
			decoder_input = target_variable[di] # Next target is next input

	else:
		# Without teacher forcing: use network's own prediction as the next input
		for di in range(target_length):
			decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
			loss += criterion(decoder_output, target_variable[di])
			
			# Get most likely word index (highest value) from output
			topv, topi = decoder_output.data.topk(1)
			ni = topi[0][0]
			
			decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
			if USE_CUDA: decoder_input = decoder_input.cuda()

			# Stop at end of sentence (not necessary when using known targets)
			if ni == code_2_index['EOS']: break

	# Backpropagation
	loss.backward()
	torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
	encoder_optimizer.step()
	decoder_optimizer.step()
	
	return loss.data[0] / target_length


def as_minutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def time_since(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def convert_data_format(list_items):
	global max_seq_len
	abstracts = []
	targets = []
	for item in list_items:
		for seq in item['sequence']:
			abstracts.append(item['abstract'])
			seq_mesh = seq[1].split(".")
			seq_mesh.append('EOS')
			seq_mesh.insert(0,'SOS')
			targets.append(seq_mesh)
			break

	return abstracts, targets


def fit_tokenizer(X):
	all_words = []
	for abstract in X:
		all_words += word_tokenize(abstract.decode('utf8', 'ignore')) 
	all_words = list(set(all_words))

	word2index = {}
	for i in xrange(len(all_words)):
		word2index[all_words[i]] = i

	return word2index, len(all_words)

def build_sequences(X, word2index):
	X_new = []
	for abstract in X:
		seq = word_tokenize(abstract.decode('utf8', 'ignore'))
		X_new.append([word2index[k] for k in seq])

	print "build_sequences ", len(X_new) 
	return X_new


def compute_vocab_target(Y):
	code_2_index = {}
	index_2_code = {}
	vocab = {}
	for i in xrange(len(Y)):
		for j in xrange(len(Y[i])):
			vocab[Y[i][j]] = 1
	
	i = 0
	for code in vocab.keys():
		code_2_index[code] = i
		i += 1

	index_2_code = [0 for i in xrange(len(vocab.keys()))]
	for code, ind in code_2_index.items():
		index_2_code[ind] = code

	return code_2_index, index_2_code


def get_y_index_sequences(Y):
	Y_new = []
	for sequence in Y:
		Y_new.append([code_2_index[s] for s in sequence])

	return Y_new


def split_data(data, split_ratio):
	ind1 = rd.sample(range(len(data)), int(split_ratio*len(data)))
	ind2 = [i for i in xrange(len(ind1)) if i not in ind1 ]

	data1 = [data[i] for i in ind1]
	data2 = [data[i] for i in ind2]

	return data1, data2

def get_data():
	list_items = cPickle.load(open('pubmed.pkl','r'))
	
	list_items_train, list_items_test = split_data(list_items, 0.6)
	list_items_train, list_items_val = split_data(list_items_train, 0.8)

	X_train, Y_train = convert_data_format(list_items_train)
	X_test, Y_test = convert_data_format(list_items_test)
	X_val, Y_val = convert_data_format(list_items_val)

	word2index, num_english_words = fit_tokenizer(X_train+X_test+X_val)

	X_train = build_sequences(X_train, word2index)
	X_test = build_sequences(X_test, word2index)
	X_val = build_sequences(X_val, word2index)

	return (X_train, Y_train, X_test, Y_test, X_val, Y_val, num_english_words)


X_train, Y_train, X_test, Y_test, X_val, Y_val, num_english_words = get_data()
code_2_index, index_2_code = compute_vocab_target(Y_train+Y_test+Y_val)

Y_train = get_y_index_sequences(Y_train)
Y_test = get_y_index_sequences(Y_test)
Y_val = get_y_index_sequences(Y_val)

output_size = len(code_2_index.keys())
sequences = Y_train+Y_test+Y_val
node_to_children = tom.get_node_children(sequences)

# Running Training
attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05

# Initialize models
encoder = EncoderRNN(num_english_words, hidden_size, n_layers)
decoder = AttnDecoderRNN(attn_model, hidden_size, output_size, n_layers, dropout_p=dropout_p)

# Move models to GPU
if USE_CUDA:
	encoder.cuda()
	decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()    


# Configuring training
n_epochs = 20000
plot_every = 200
print_every = 1000

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# Begin!
for epoch in range(1, n_epochs + 1):
	# Get training data for this cycle
	# training_pair = variables_from_pair(random.choice(pairs))
	ind = rd.choice(xrange(len(X_train)))
	if len(X_train[ind]) == 0:
		continue
	input_variable = Variable(torch.LongTensor(X_train[ind]).view(-1, 1)).cuda()
	target_variable = Variable(torch.LongTensor(Y_train[ind]).view(-1, 1)).cuda()

	# Run the train function
	loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

	# Keep track of loss
	print_loss_total += loss
	plot_loss_total += loss

	if epoch == 0: continue

	if epoch % print_every == 0:
		print_loss_avg = print_loss_total / print_every
		print_loss_total = 0
		print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / float(n_epochs)), epoch, epoch / n_epochs * 100, print_loss_avg)
		print(print_summary)

	if epoch % plot_every == 0:
		plot_loss_avg = plot_loss_total / plot_every
		plot_losses.append(plot_loss_avg)
		plot_loss_total = 0


