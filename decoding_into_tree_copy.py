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
from attention_model_copy import *
import numpy as np
import tree_of_mesh as tom 
import os.path
import nltk
import operator
import read_embeddings as re
from collections import Counter

np.random.seed(9001)
rd.seed(9001)

USE_CUDA = True

teacher_forcing_ratio = 1.0
clip = 100.0
num_top_words = 70000

def traverse_tree(node, decoder_input, decoder_context, decoder_hidden, encoder_outputs, teacher_forcing, depth):

	if len(decoder_input) == 0:
		return 0

	loss = 0
	target = [0]*output_size
	for i in node.children.keys():
		# print i, index_2_code[i]
		target[i] = 1
	
	print "target:", node.children.keys()
	target = [target]
	target = Variable(torch.FloatTensor(target)).cuda()
	
	if teacher_forcing:
		prev_decoder_context = decoder_context.clone()
		prev_decoder_hidden = decoder_hidden.clone()
		decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
		prev_decoder_output = decoder_output.clone()

		# loss += float(np.exp(-depth))*criterion(decoder_output, target)
		loss = criterion(decoder_output, target)
		print "before update:", decoder_output.data[0].cpu().numpy()[node.children.keys()]
		# if type(loss) != int:
		loss.backward(retain_graph=True)
		# torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
		# torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
		encoder_optimizer.step()
		decoder_optimizer.step()
		decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, prev_decoder_context, prev_decoder_hidden, encoder_outputs)
		print "after update:", decoder_output.data[0].cpu().numpy()[node.children.keys()]
		if len(node.children.keys()) > 0:
			cPickle.dump((prev_decoder_output.data[0].cpu().numpy(), decoder_output.data[0].cpu().numpy(), target.data[0].cpu().numpy()), open('dump.pkl','w'))

	else:
		# print "length decoder input: ", len(decoder_input)
		for one_input in decoder_input:
			one_input = Variable(torch.LongTensor([[one_input]])).cuda()
			decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(one_input, decoder_context, decoder_hidden, encoder_outputs)
			loss += criterion(decoder_output, target)
		if type(loss) != int:
			loss.backward()
		torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
		torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
		encoder_optimizer.step()
		decoder_optimizer.step()

	# print index_2_code[decoder_input.data[0].cpu().numpy()[0]], index_2_code[node.name], index_2_code[node.children.keys()[0]], len(node.children.keys())
	
	if len(node.children.keys()) == 0:
		return loss

	for child in node.children:
		if teacher_forcing:
			decoder_input = Variable(torch.LongTensor([[node.children[child].name]])).cuda()
		else:
			# print decoder_output
			# print "max decoder output: ", np.max(decoder_output.data[0].cpu().numpy())
			decoder_input = (decoder_output > 0.5).data[0].cpu().numpy()
			decoder_input = np.where(decoder_input)[0]
			decoder_input = rd.sample(decoder_input, min(len(decoder_input), 5))
		loss += traverse_tree(node.children[child], decoder_input, decoder_context, decoder_hidden, encoder_outputs, teacher_forcing, depth+1)

	return loss


def train(input_variables, target_variables, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
	
	# Zero gradients of both optimizers
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0 # Added onto for each word

	teacher_forcing = rd.random() < teacher_forcing_ratio

	for it in xrange(len(input_variables)):

		input_variable = input_variables[it]
		target_variable = target_variables[it]

		# Convert it to LongTensor
		input_variable = Variable(torch.LongTensor(input_variable).view(-1, 1)).cuda()

		# Get size of input and target sentences
		# print input_variable.size(), target_variable.size()
		input_length = input_variable.size()[0]
		target_length = float(sum([len(a) for a in target_variable]))

		# Run words through encoder
		encoder_hidden = encoder.init_hidden()
		encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
		
		# Prepare input and output variables
		if teacher_forcing:			
			decoder_input = Variable(torch.LongTensor([[code_2_index['SSOS']]]))
		else:
			decoder_input = [code_2_index['SSOS']]
		
		decoder_context = Variable(torch.zeros(1, decoder.hidden_size), requires_grad=True)
		decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
		
		if USE_CUDA:
			if teacher_forcing:
				decoder_input = decoder_input.cuda()
			decoder_context = decoder_context.cuda()

		root = tom.generate_tree(target_variable)
		loss += traverse_tree(root, decoder_input, decoder_context, decoder_hidden, encoder_outputs, teacher_forcing, 0)

	# Backpropagation
	# if type(loss) != int:
	# 	loss.backward()
	# torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	# torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
	# encoder_optimizer.step()
	# decoder_optimizer.step()


	if type(loss) != int:
		return loss.data[0]
	else:
		return 0

def traverse_prediction(decoder_input, decoder_context, decoder_hidden, encoder_outputs, depth):
	sequences = []
	# print "depth: "+ str(depth) + "base condition traverse_prediction: "+ str(decoder_input.data[0].cpu().numpy()[0])
	# print "code: ", index_2_code[int(decoder_input.data[0].cpu().numpy()[0])]	
	if decoder_input.data[0].cpu().numpy()[0] == code_2_index['EOS'] or depth == 20:
		return [[]]

	decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
	# print decoder_output
	# 
	if index_2_code[int(decoder_input.data[0].cpu().numpy()[0])] =="SSOS":
		print "SOS:", decoder_output.data[0].cpu().numpy()[code_2_index["SOS"]]
	predictions = (decoder_output > 0.5).data[0].cpu().numpy()
	predictions = np.where(predictions)[0]
	
	predictions = [p for p in predictions if p in node_to_children[int(decoder_input.data[0].cpu().numpy()[0])]]

	if len(predictions) == 0:
		# print "predictions length 0"
		return [[]]

	for pred in predictions:
		decoder_input = Variable(torch.LongTensor([[pred]])).cuda()
		lists_returned = traverse_prediction(decoder_input, decoder_context, decoder_hidden, encoder_outputs, depth+1)
		sequences += [[pred] + sublist for sublist in lists_returned]
	return sequences


def predict(input_variable, encoder, decoder):

	# Run words through encoder
	encoder_hidden = encoder.init_hidden()
	encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

	# Prepare input and output variables
	decoder_input = Variable(torch.LongTensor([[code_2_index['SSOS']]]))
	decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
	decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
	if USE_CUDA:
		decoder_input = decoder_input.cuda()
		decoder_context = decoder_context.cuda()

	predictions = traverse_prediction(decoder_input, decoder_context, decoder_hidden, encoder_outputs, 0)
	predictions = [[code_2_index["SSOS"]]+sublist for sublist in predictions]

	return predictions 

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
		target_for_one_abstract = []
		abstracts.append(item['abstract'])
		for seq in item['sequence']:
			seq_mesh = seq[1].split(".")
			seq_mesh.append('EOS')
			seq_mesh.insert(0,'SOS')
			target_for_one_abstract.append(seq_mesh)
		targets.append(target_for_one_abstract)

	return abstracts, targets

def fit_tokenizer(X):
	all_words = []
	for abstract in X:
		all_words += word_tokenize(abstract.decode('utf8', 'ignore'))
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
		seq = [s.lower() for s in seq]
		X_new.append([word2index[k] if k in word2index else word2index["UNK"]  for k in seq])

	print "build_sequences ", len(X_new) 
	return X_new

def flatten_list(Y):
	Y = [code for listoflist in Y for sublist in listoflist for code in sublist]
	return Y 

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

def get_y_index_sequences(Y):
	Y_new = []
	for sequences in Y:
		sequences_for_one_abstarct = []
		for sequence in sequences:
			sequence = ["SSOS"] + sequence
			sequences_for_one_abstarct.append([code_2_index[s] for s in sequence])
		Y_new.append(sequences_for_one_abstarct)
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

	return (X_train, Y_train, X_test, Y_test, X_val, Y_val, num_english_words, word2index)

def get_mesh_terms_from_sequences(true_sequences, predicted_sequences):
	true_mesh_terms = []
	pred_mesh_terms = []
	seq2mesh = cPickle.load(open('seq_to_mesh.pkl','r'))
	
	for sequence in true_sequences:
		sequence = [str(s) for s in sequence[2:-1]]
		if ".".join(sequence) in seq2mesh:
			true_mesh_terms.append(seq2mesh[".".join(sequence)])

	for sequence in predicted_sequences:
		sequence = [str(s) for s in sequence[2:-1]]
		if ".".join(sequence) in seq2mesh:
			pred_mesh_terms.append(seq2mesh[".".join(sequence)])

	return true_mesh_terms, pred_mesh_terms

def get_metrics(true_mesh_terms, pred_mesh_terms):
	true_mesh_terms = set(true_mesh_terms)
	pred_mesh_terms = set(pred_mesh_terms)
	
	if len(pred_mesh_terms) == 0:
		precision = 0.0
	else:
		precision = len(true_mesh_terms.intersection(pred_mesh_terms))/float(len(pred_mesh_terms))

	if len(true_mesh_terms) == 0:
		recall = 1.0
	else:
		recall = len(true_mesh_terms.intersection(pred_mesh_terms))/float(len(true_mesh_terms))
	
	if precision ==0 and recall == 0:
		f1_score = 0.0
	else:
		f1_score = 2*precision*recall/(precision+recall)

	return (precision, recall, f1_score)


def save_model_after_training(encoder, decoder):
	encoder = encoder.cpu()
	decoder = decoder.cpu()
	cPickle.dump((encoder, decoder), open('trained_model.pkl','w'))




X_train, Y_train, X_test, Y_test, X_val, Y_val, num_english_words, word2index = get_data()
X_train = [X_train[i] for i in xrange(1)]
Y_train = [Y_train[i] for i in xrange(1)]

word_embeddings = re.read_word_embeddings(word2index)
node_embeddings = re.read_node_embeddings()

code_2_index, index_2_code = compute_vocab_target(Y_train+Y_test+Y_val)
code_2_index['SSOS'] = len(code_2_index.keys())
index_2_code = index_2_code + ['SSOS']

print "got code to index stuff"

if os.path.isfile('./code_2_index.pkl'):
	code_2_index = cPickle.load(open('code_2_index.pkl', 'r'))
	index_2_code = cPickle.load(open('index_2_code.pkl', 'r'))
else:
	cPickle.dump(code_2_index, open('code_2_index.pkl', 'w'))
	cPickle.dump(index_2_code, open('index_2_code.pkl', 'w'))

Y_train = get_y_index_sequences(Y_train)
Y_test = get_y_index_sequences(Y_test)
Y_val = get_y_index_sequences(Y_val)

output_size = len(code_2_index.keys())
sequences = Y_train+Y_test+Y_val
sequences = [[code_2_index['SSOS']]+s for sublist in sequences for s in sublist]
node_to_children = tom.get_node_children(sequences)

# Running Training
attn_model = 'general'
hidden_size = 500
n_layers = 1
dropout_p = 0.00

# Initialize models
encoder = EncoderRNN(num_english_words, hidden_size, n_layers, embeddings=word_embeddings)
decoder = AttnDecoderRNN(attn_model, hidden_size, output_size, n_layers, dropout_p=dropout_p, embeddings=node_embeddings)

# Move models to GPU
if USE_CUDA:
	encoder.cuda()
	decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
# criterion = nn.MultiLabelSoftMarginLoss()    
# criterion = nn.MSELoss()
criterion = nn.BCELoss()
# Configuring training
n_epochs = 	20
plot_every = 200
print_every = 1
batch_size = 1

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every
seq_len_to_train = 2

partial_training = False

if partial_training:
	encoder, decoder = cPickle.load(open('trained_model.pkl','r'))
	encoder = encoder.cuda()
	decoder = decoder.cuda()

load_trained_model = False

if load_trained_model:
	encoder, decoder = cPickle.load(open('trained_model.pkl','r'))

else:
	for epoch in range(1, n_epochs + 1):

		# if epoch > 20:
		# 	teacher_forcing_ratio = 1.0

		if epoch > int(n_epochs/3) and epoch < int((2/3.0)*n_epochs):
			seq_len_to_train = 2

		if epoch > int((2/3.0)*n_epochs):
			seq_len_to_train = 2

		# Get training data for this cycle
		# training_pair = variables_from_pair(random.choice(pairs))
		ind = rd.sample(xrange(len(X_train)), batch_size)

		ind = [i for i in ind if len(X_train[i])>0]
		
		# input_variables = Variable(torch.LongTensor(X_train[ind]).view(-1, 1)).cuda()
		input_variables = [X_train[i] for i in ind]
		# target_variable = Variable(torch.LongTensor(Y_train[ind]).view(-1, 1)).cuda()
		target_variables = [Y_train[i][:seq_len_to_train] for i in ind]
		# Run the train function
		loss = train(input_variables, target_variables, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

		# Keep track of loss
		print_loss_total += loss
		plot_loss_total += loss

		if epoch == 0: continue

		if epoch % print_every == 0:
			print_loss_avg = print_loss_total / float(epoch)
			print_loss_total = 0
			print_summary = '%s (%d %d%%) %.8f' % (time_since(start, epoch / float(n_epochs)), epoch, epoch / float(n_epochs) * 100, print_loss_avg)
			print(print_summary)

		if epoch % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0

		input_variable = input_variables[0]
		input_variable = Variable(torch.LongTensor(input_variable).view(-1, 1)).cuda()
		predicton = predict(input_variable, encoder, decoder)
	save_model_after_training(encoder, decoder)

print "finished training"

# Move back to Cuda
encoder = encoder.cuda()
decoder = decoder.cuda()

# # Just for debugging purposes
X_test = X_train
Y_test = Y_train

# predictions = []
# metrics = []
# for i in xrange(len(X_test)):	
# 	if len(X_test[i]) == 0:
# 		continue

# 	input_variable = Variable(torch.LongTensor(X_test[i]).view(-1, 1)).cuda()
# 	predicton = predict(input_variable, encoder, decoder)

# 	print "************************************************"
# 	print predicton, Y_test[i]
# 	print "************************************************"
# 	true_mesh_terms, pred_mesh_terms = get_mesh_terms_from_sequences(Y_test[i], predicton)
# 	print true_mesh_terms, pred_mesh_terms
# 	metrics += [get_metrics(true_mesh_terms, pred_mesh_terms)]
# 	# print true_mesh_terms, pred_mesh_terms
# 	predictions.append(predicton)

cPickle.dump((X_test, Y_test, predictions, metrics), open('predictions.pkl','w'))

