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
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from attention_model import *
import numpy as np
import tree_of_mesh as tom 
import os.path
import nltk
import operator
import read_embeddings as re
from collections import Counter

ps = PorterStemmer()

np.random.seed(9001)
rd.seed(9001)

USE_CUDA = True

teacher_forcing_ratio = 1.0
clip = 5.0
num_top_words = 50000


def get_y_index_sequences(Y):
	Y_new = []
	for sequences in Y:
		sequences_for_one_abstarct = []
		for sequence in sequences:
			sequence = ["SSOS"] + sequence
			sequences_for_one_abstarct.append([code_2_index[s] for s in sequence])
		Y_new.append(sequences_for_one_abstarct)
	return Y_new

def fit_tokenizer(X):
	all_words = []
	for abstract in X:
		all_words += word_tokenize(abstract.decode('utf8', 'ignore'))

	# removing stop words
	all_words = [word for word in all_words if word not in stopwords.words('english')]

	all_words_new = []
	for word in all_words:
		word = word.split("/")
		all_words_new += word

	all_words = all_words_new
	all_words = [word.lower() for word in all_words]

	wordcount = Counter(all_words)
	sorted_wc = sorted(wordcount.items(), key=operator.itemgetter(0), reverse=True)
	top_words = [l[0] for l in sorted_wc[0:num_top_words]]
	top_words = top_words + ["UNK"]

	word2index = {}
	for i in xrange(len(top_words)):
		word2index[top_words[i]] = i

	return word2index, len(top_words)

def build_sequences(X, word2index):
	X_new = []
	for abstract in X:
		seq = word_tokenize(abstract.decode('utf8', 'ignore'))
		seq_new = []
		for s in seq :
			seq_new += s.split("/")
		seq = seq_new
		seq = [s.lower() for s in seq]
		seq = [ps.stem(s) for s in seq]
		# X_new.append([word2index[k] if k in word2index else word2index["UNK"]  for k in seq])
		X_new.append([word2index[k] for k in seq if k in word2index])
	print "build_sequences ", len(X_new) 
	return X_new

def flatten_list(Y):
	Y = [code for listoflist in Y for sublist in listoflist for code in sublist]
	return Y 




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

	return (X_train, Y_train, X_test, Y_test, X_val, Y_val, num_english_words, word2index)

def convert_data_format(list_items):
	global max_seq_len
	abstracts = []
	targets = []
	for item in list_items:
		target_for_one_abstract = []
		abstracts.append(item['abstract'])
		for seq in item['sequence']:
			seq_mesh = seq[1].split(".")
			seq_mesh = [ ":".join(seq_mesh[:i+1]) for i in xrange(len(seq_mesh))]
			seq_mesh.append('EOS')
			seq_mesh.insert(0,'SOS')
			target_for_one_abstract.append(seq_mesh)
		targets.append(target_for_one_abstract)
	return abstracts, targets

def traverse_prediction(decoder_input, decoder_context, decoder_hidden, encoder_outputs, depth, x):
	sequences = []
	input_for_decoder = decoder_input.data[0].cpu().numpy()[0]
	if decoder_input.data[0].cpu().numpy()[0] == code_2_index['EOS'] or depth == 20:
		return [[]]

	decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

	predictions = (decoder_output > 0).data[0].cpu().numpy()
	predictions = [i for i in node_to_children[int(decoder_input.data[0].cpu().numpy()[0])] if predictions[i] == True]
	
	for prediction in predictions:
		print index_2_code[input_for_decoder] + " -> " + index_2_code[prediction]

	top_attentive_words = 30
	attention = decoder_attention.data[0].cpu().numpy()[0]
	indices = np.argsort(attention)[::-1][0:top_attentive_words]

	abstract = [index2words[x[i]] for i in xrange(len(x))]
	print " ".join(abstract)

	for i in xrange(len(x)):
		if i in indices:
			print index2words[x[i]], ":", attention[i]
		# else:
		# 	print index2words[x[i]], ":", 0.0

	if len(predictions) == 0:
		return [[]]

	for pred in predictions:
		decoder_input = Variable(torch.LongTensor([[pred]])).cuda()
		lists_returned = traverse_prediction(decoder_input, decoder_context, decoder_hidden, encoder_outputs, depth+1, x)
		sequences += [[pred] + sublist for sublist in lists_returned]
	return sequences


def predict(input_variable, encoder, decoder, x):
	# Run words through encoder
	encoder_hidden = encoder.init_hidden()
	encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

	# Prepare input and output variables
	decoder_input = Variable(torch.LongTensor([[code_2_index['SSOS']]]))
	decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
	decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
	
	decoder_input = decoder_input.cuda()
	decoder_context = decoder_context.cuda()

	predictions = traverse_prediction(decoder_input, decoder_context, decoder_hidden, encoder_outputs, 0, x)
	predictions = [[code_2_index["SSOS"]]+sublist for sublist in predictions]

	return predictions 


def compute_vocab_target(Y):
	Y = flatten_list(Y)
	code_2_index = {}
	index_2_code = {}
	vocab = {}

	for i in xrange(len(Y)):
		vocab[Y[i]] = 1
	
	i = 0
	for code in vocab.keys():
		code_2_index[code] = i
		i += 1

	index_2_code = [0 for i in xrange(len(vocab.keys()))]
	for code, ind in code_2_index.items():
		index_2_code[ind] = code

	return code_2_index, index_2_code


def generate_predictions(X, Y, encoder=None, decoder=None):	
	if encoder is None and decoder is None:
		encoder, decoder = load_model()
	
	predictions = []
	metrics = []
	for i in xrange(len(X)):	
		if len(X[i]) == 0:
			continue
		input_variable = Variable(torch.LongTensor(X[i]).view(-1, 1)).cuda()
		predicton = predict(input_variable, encoder, decoder, X[i])
		# true_mesh_terms, pred_mesh_terms = get_mesh_terms_from_sequences(Y[i], predicton)
		# metrics += [get_metrics(true_mesh_terms, pred_mesh_terms)]
		# predictions.append(predicton)

	# cPickle.dump((X, Y, predictions, metrics), open('predictions.pkl','w'))
	# return metrics

def load_model():
	encoder, decoder = cPickle.load(open('trained_model.pkl','r'))
	encoder = encoder.cuda()
	decoder = decoder.cuda()
	return encoder, decoder


def get_mesh_terms_from_sequences(true_sequences, predicted_sequences):
	true_mesh_terms = []
	pred_mesh_terms = []
	
	for sequence in true_sequences:
		sequence = [str(index_2_code[s]) for s in sequence[2:-1]]
		if ".".join(sequence) in seq2mesh:
			true_mesh_terms.append(seq2mesh[".".join(sequence)])

	for sequence in predicted_sequences:
		sequence = [str(index_2_code[s]) for s in sequence[2:-1]]
		if ".".join(sequence) in seq2mesh:
			pred_mesh_terms.append(seq2mesh[".".join(sequence)])

	print true_mesh_terms, pred_mesh_terms
	return true_mesh_terms, pred_mesh_terms


seq2mesh = cPickle.load(open('seq_to_mesh.pkl','r'))
X_train, Y_train, X_test, Y_test, X_val, Y_val, num_english_words, word2index = get_data()

index2words = [0]*num_english_words
for word, ind in word2index.items():
	index2words[int(ind)] = word


word_embeddings = re.read_word_embeddings(word2index)
node_embeddings = re.read_node_embeddings()

# code_2_index, index_2_code = compute_vocab_target(Y_train+Y_test+Y_val)
# code_2_index['SSOS'] = len(code_2_index.keys())
# index_2_code = index_2_code + ['SSOS']


code_2_index = cPickle.load(open('code_2_index.pkl', 'r'))
index_2_code = cPickle.load(open('index_2_code.pkl', 'r'))

Y_train = get_y_index_sequences(Y_train)
Y_test = get_y_index_sequences(Y_test)
Y_val = get_y_index_sequences(Y_val)

# #reduce size of validation set
# X_val = [X_val[i] for i in xrange(100)]
# Y_val = [Y_val[i] for i in xrange(100)]

output_size = len(code_2_index.keys())
sequences = Y_train+Y_test+Y_val
sequences = [[code_2_index['SSOS']]+s for sublist in sequences for s in sublist]
node_to_children = tom.get_node_children(sequences)

#load model
encoder, decoder = cPickle.load(open('trained_model.pkl','r'))
encoder = encoder.cuda()
decoder = decoder.cuda()
print Y_val[:1]
for i in xrange(500):
	print "Here"
	generate_predictions([X_test[i]], [Y_test[i]])
