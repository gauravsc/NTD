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
from attention_model_logit import *
import numpy as np
import tree_of_mesh as tom 
import os.path
import nltk
import operator
import read_embeddings as re
from collections import Counter

def sigmoid(x):
	try:
		ans = np.exp(-x)
	except OverflowError:
		ans = float('inf')
	return 1 / (1 + ans)


ps = PorterStemmer()
np.random.seed(9001)
rd.seed(9001)

USE_CUDA = True

teacher_forcing_ratio = 1.0
clip = 5.0
num_top_words = 30000
depth_threshold = 0


abstract_train = None
abstract_test = None
abstract_val = None

epoch_suff = 100*np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

# @profile
def traverse_tree(node, decoder_input, decoder_context, decoder_hidden, encoder_outputs, teacher_forcing, depth, target_len, threshold):
	if len(decoder_input) == 0 or len(node.children.keys()) == 0:
		return 0

	loss = 0
	# target = [0.0]*output_size
	target = np.zeros(output_size)
	for i in node.children.keys():
		target[i] = 1.0
	
	mask = np.zeros(output_size)
	# mask = [0.0]*output_size

	if teacher_forcing:
		# prev_decoder_hidden = decoder_hidden.clone()
		# prev_decoder_context = decoder_context.clone()
		decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
		p = rd.random()
		compute_p = 0.50 + (freq_nodes_min+1)*1.0/float(freq_nodes[node.name])
		# print compute_p
		if p < compute_p:  
			act_children = np.array([i for i in node_to_children[node.name] if target[i] == 1])
			inact_children = np.array([i for i in node_to_children[node.name] if target[i] == 0])
			# inact_children_len = inact_children.shape[0]
			# act_children_len = act_children.shape[0]
			# if inact_children_len > 15:
			# 	sampled_inact_children = np.random.choice(inact_children,  min(inact_children_len,3*act_children_len))
			# else:
			sampled_inact_children = inact_children

			# for i in node_to_children[node.name]:
			for i in sampled_inact_children:
				# if target_new[i] == 0:
				# 	if rd.random() < 0.5:
				mask[i] = 1.0

			for i in act_children:
				mask[i] = 1.0

			# mask = [mask]
			# target = [target]
			mask = mask.reshape((1, -1))
			target = target.reshape((1, -1))

			sum_masks = np.sum(mask)

			target = Variable(torch.FloatTensor(target)).cuda()
			mask = Variable(torch.FloatTensor(mask)).cuda()
	 
			# decoder_output_before = decoder_output.data[0].cpu().numpy()
			loss = criterion(decoder_output, target)/(target_len*batch_size*sum_masks)
			loss.backward(mask, retain_graph=True)
			
			# encoder_optimizer.step()
			# decoder_optimizer.step()
			# decoder_output_after, _, _, _ =  decoder(decoder_input, prev_decoder_context, prev_decoder_hidden, encoder_outputs)
			# decoder_output_after = decoder_output_after.data[0].cpu().numpy()

			# target = target.data[0].cpu().numpy()
			# print len(decoder_output_before), len(decoder_output_after), len(target)
			# for i in act_children+inact_children:
			# 	print sigmoid(decoder_output_before[i]), sigmoid(decoder_output_after[i]), target[i]

		
	# else:
	# 	# print "length decoder input: ", len(decoder_input)
	# 	for one_input in decoder_input:
	# 		target_new = []
	# 		one_input = Variable(torch.LongTensor([[one_input]])).cuda()
	# 		decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(one_input, decoder_context, decoder_hidden, encoder_outputs)
	# 		decoder_output_new = decoder_output.data.cpu().numpy()
	# 		target_new = [target[i] if i in node_to_children[node.name] else decoder_output_new[0,i] for i in xrange(len(target))]		
	# 		target_new = [[float(t) for t in target_new]]
	# 		target_new = Variable(torch.FloatTensor(target_new)).cuda()
	# 		loss += criterion(decoder_output, target_new)
	# 		loss.backward(retain_graph=True)
	# 		torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	# 		torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
	# 		encoder_optimizer.step()
	# 		decoder_optimizer.step()

	# print index_2_code[decoder_input.data[0].cpu().numpy()[0]], index_2_code[node.name], index_2_code[node.children.keys()[0]], len(node.children.keys())

	for child in node.children:
		if teacher_forcing:
			decoder_input = Variable(torch.LongTensor([[node.children[child].name]])).cuda()
		else:
			# print decoder_output
			# print "max decoder output: ", np.max(decoder_output.data[0].cpu().numpy())
			decoder_input = (decoder_output > threshold).data[0].cpu().numpy()
			decoder_input = [i for i in node_to_children[node.name] if decoder_input[i]]
		loss += traverse_tree(node.children[child], decoder_input, decoder_context, decoder_hidden, encoder_outputs, teacher_forcing, depth+1, target_len, threshold)

	return loss

# @profile	
def train(input_variables, target_variables, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, threshold):
	
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
		input_length = input_variable.size()[0]
		unique_target = list(set([a for sublist in target_variable for a in sublist]))
		target_length = len(unique_target)
		# target_length = float(sum([len(a) for a in target_variable]))

		# Run words through encoder
		encoder_hidden = encoder.init_hidden()
		encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
		
		# Prepare input and output variables
		if teacher_forcing:			
			decoder_input = Variable(torch.LongTensor([[code_2_index['SSOS']]]))
		else:
			decoder_input = [code_2_index['SSOS']]
		
		decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
		decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
		
		if USE_CUDA:
			if teacher_forcing:
				decoder_input = decoder_input.cuda()
			decoder_context = decoder_context.cuda()

		root = tom.generate_tree(target_variable)
		# cPickle.dump((root, target_variable), open('root.pkl', 'w'))
		# return
		loss += traverse_tree(root, decoder_input, decoder_context, decoder_hidden, encoder_outputs, teacher_forcing, 0, target_length, threshold)

	torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
	encoder_optimizer.step()
	decoder_optimizer.step()
	# Backpropagation
	# loss.backward()
	# torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	# torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
	# encoder_optimizer.step()
	# decoder_optimizer.step()

	if type(loss) != int:
		return loss.data[0]
	else:
		return 0


# @profile
def traverse_prediction(decoder_input, decoder_context, decoder_hidden, encoder_outputs, depth, threshold):
	sequences = []
	if decoder_input.data[0].cpu().numpy()[0] == code_2_index['EOS'] or depth == 20:
		return [[]]

	decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
	

	# decoder_output = np.array([sigmoid(i) for i in decoder_output.data[0].cpu().numpy()])
	decoder_output = sigmoid(decoder_output.data[0].cpu().numpy())
	predictions = decoder_output > threshold

	predictions = [i for i in node_to_children[int(decoder_input.data[0].cpu().numpy()[0])] if predictions[i] == True]
	
	# print "depth:", depth, "decoder output activated:", len(predictions), "total outputs: ", len(node_to_children[int(decoder_input.data[0].cpu().numpy()[0])])
	# predictions = np.where(predictions)[0]
	# predictions = [p for p in predictions if p in node_to_children[int(decoder_input.data[0].cpu().numpy()[0])]]

	if len(predictions) == 0:
		return [[]]

	for pred in predictions:
		decoder_input = Variable(torch.LongTensor([[pred]])).cuda()
		lists_returned = traverse_prediction(decoder_input, decoder_context, decoder_hidden, encoder_outputs, depth+1, threshold)
		sequences += [[pred] + sublist for sublist in lists_returned]
	return sequences

# @profile
def predict(input_variable, encoder, decoder, threshold):
	# Run words through encoder
	encoder_hidden = encoder.init_hidden()
	encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

	# Prepare input and output variables
	decoder_input = Variable(torch.LongTensor([[code_2_index['SSOS']]]))
	decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
	decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
	
	decoder_input = decoder_input.cuda()
	decoder_context = decoder_context.cuda()

	predictions = traverse_prediction(decoder_input, decoder_context, decoder_hidden, encoder_outputs, 0, threshold)
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

def fit_tokenizer(X):
	all_words = []
	for abstract in X:
		all_words += word_tokenize(abstract.decode('utf8', 'ignore'))

	# removing stop words
	# all_words = [word for word in all_words if word not in stopwords.words('english')]

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
		# seq = [ps.stem(s) for s in seq]
		# X_new.append([word2index[k] if k in word2index else word2index["UNK"]  for k in seq])
		X_new.append([word2index[k] for k in seq if k in word2index])
	print "build_sequences ", len(X_new) 
	return X_new

def flatten_list(Y):
	Y = [code for listoflist in Y for sublist in listoflist for code in sublist]
	return Y 


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
	ind2 = [i for i in xrange(len(data)) if i not in ind1]

	data1 = [data[i] for i in ind1]
	data2 = [data[i] for i in ind2]

	return data1, data2

def get_data():
	global abstract_train, abstract_test, abstract_val
	list_items = cPickle.load(open('pubmed.pkl','r'))
	
	list_items_train, list_items_test = split_data(list_items, 0.6)
	list_items_train, list_items_val = split_data(list_items_train, 0.8)

	abstract_train, Y_train = convert_data_format(list_items_train)
	abstract_test, Y_test = convert_data_format(list_items_test)
	abstract_val, Y_val = convert_data_format(list_items_val)

	word2index, num_english_words = fit_tokenizer(abstract_train+abstract_test+abstract_val)

	X_train = build_sequences(abstract_train, word2index)
	X_test = build_sequences(abstract_test, word2index)
	X_val = build_sequences(abstract_val, word2index)

	return (X_train, Y_train, X_test, Y_test, X_val, Y_val, num_english_words, word2index)

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
	
	if precision == 0 and recall == 0:
		f1_score = 0.0
	else:
		f1_score = 2*precision*recall/(precision+recall)

	return (precision, recall, f1_score)


def save_model_after_training(encoder, decoder):
	encoder = encoder.cpu()
	decoder = decoder.cpu()
	cPickle.dump((encoder, decoder), open('trained_model_'+str(len(X_train)) +'.pkl','w'))
	print "writing model"
	encoder = encoder.cuda()
	decoder = decoder.cuda()


def load_model():
	encoder, decoder = cPickle.load(open('trained_model_'+str(len(X_train)) +'.pkl','r'))
	encoder = encoder.cuda()
	decoder = decoder.cuda()
	return encoder, decoder

def generate_predictions(X, Y, encoder=None, decoder=None, is_test=0, abstracts=None, threshold=0.5):	
	if encoder is None and decoder is None:
		encoder, decoder = load_model()
	
	mesh_terms_pred = []
	mesh_terms_test = []
	seq_pred = []
	metrics = []
	for i in xrange(len(X)):	
		if len(X[i]) == 0:
			continue
		input_variable = Variable(torch.LongTensor(X[i]).view(-1, 1)).cuda()
		predicton = predict(input_variable, encoder, decoder, threshold)
		# if i < 3:
		# 	print "***********begins***********"
		# 	print Y[i], predicton 
		# 	print "************ends************"
		if i < 3:
			print predicton
		true_mesh_terms, pred_mesh_terms = get_mesh_terms_from_sequences(Y[i], predicton)
		true_mesh_terms = list(set([t.strip() for t in true_mesh_terms]))
		pred_mesh_terms = list(set([t.strip() for t in pred_mesh_terms]))
		metrics += [get_metrics(true_mesh_terms, pred_mesh_terms)]
		mesh_terms_pred.append(pred_mesh_terms)
		mesh_terms_test.append(true_mesh_terms)
		seq_pred.append(predicton)
	if is_test == 1:
		cPickle.dump((mesh_terms_test, mesh_terms_pred, seq_pred), open('predictions_decoder_'+str(len(X_train))+'.pkl','w'))
	return metrics

def get_freq(sequences_of_sequences):
	freq_nodes = {}
	for code in code_2_index.values():
		freq_nodes[code] = 0

	final_sequences = []
	for sequences in sequences_of_sequences:
		sequences_per_sample = []
		for sequence in sequences:
			sequences_per_sample += sequence
		final_sequences.append(list(set(sequences_per_sample)))

	for seq in final_sequences:
		for node in seq:
			freq_nodes[node] += 1

	f = np.array(freq_nodes.values())
	f = f[f!=0]	
	return freq_nodes, np.min(f)


X_train, Y_train, X_test, Y_test, X_val, Y_val, num_english_words, word2index = get_data()

X_train = [X_train[i] for i in xrange(20000)]
Y_train = [Y_train[i] for i in xrange(20000)]

word_embeddings = re.read_word_embeddings(word2index)
node_embeddings = re.read_node_embeddings()

# # code_2_index, index_2_code = compute_vocab_target(Y_train+Y_test+Y_val)
# code_2_index['SSOS'] = len(code_2_index.keys())
# index_2_code = index_2_code + ['SSOS']

code_2_index = cPickle.load(open('code_2_index.pkl', 'r'))
index_2_code = cPickle.load(open('index_2_code.pkl', 'r'))
seq2mesh = cPickle.load(open('seq_to_mesh.pkl','r'))

Y_train = get_y_index_sequences(Y_train)
Y_test = get_y_index_sequences(Y_test)
Y_val = get_y_index_sequences(Y_val)


# X_test = X_val[0:]
# Y_test = Y_val[0:]

# #reduce size of validation set
X_val = [X_val[i] for i in xrange(2000)]
Y_val = [Y_val[i] for i in xrange(2000)]


output_size = len(code_2_index.keys())
sequences = Y_train+Y_test+Y_val
freq_nodes, freq_nodes_min = get_freq(sequences)
sequences = [s for sublist in sequences for s in sublist]
node_to_children = tom.get_node_children(sequences)

# Running Training
attn_model = 'general'
hidden_size = 512
n_layers = 1
dropout_p = 0.0

# Initialize models
encoder = EncoderRNN(num_english_words, hidden_size, n_layers, embeddings=word_embeddings)
decoder = AttnDecoderRNN(attn_model, hidden_size, output_size, n_layers, dropout_p=dropout_p, embeddings=node_embeddings)

# Move models to GPU
if USE_CUDA:
	encoder.cuda()
	decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.1
# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=0)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=0)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0)
# criterion = nn.MultiLabelSoftMarginLoss(reduce=False)    
# criterion = nn.MSELoss(reduce=False)
# criterion = nn.BCELoss(reduce=False)
criterion = nn.BCEWithLogitsLoss(reduce=False)
# Configuring training
n_epochs = 	100000000
plot_every = 200
batch_size = 1
print_every = 1000/batch_size

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every
seq_len_to_train = 25
threshold = 0.5
partial_training = False

ind_list = []
samples =  np.random.permutation(len(X_train))
ind_list = [samples[i*batch_size:(i+1)*batch_size].tolist() for i in xrange(len(X_train)/batch_size)]

print "training samples: ", len(X_train)
print "hidden size: ", hidden_size

if partial_training:
	encoder, decoder = cPickle.load(open('trained_model_'+str(len(X_train))+'.pkl','r'))
	encoder = encoder.cuda()
	decoder = decoder.cuda()

load_trained_model = False

epoch_cur = 0

if load_trained_model:
	encoder, decoder = cPickle.load(open('trained_model_'+str(len(X_train))+'.pkl','r'))
	encoder = encoder.cuda()
	decoder = decoder.cuda()

else:
	f1_avg = -20
	for epoch in range(0, n_epochs + 1):
		epoch_cur = epoch/float(len(X_train))
		if epoch > 20:
			teacher_forcing_ratio = 1.0

		if epoch > int(n_epochs/3) and epoch < int((2/3.0)*n_epochs):
			seq_len_to_train = 25

		if epoch > int((2/3.0)*n_epochs):
			seq_len_to_train = 25

		# Get training data for this cycle
		ind = rd.sample(xrange(len(X_train)), batch_size)
	
		# ind = ind_list[epoch%len(ind_list)]
		ind = [i for i in ind if len(X_train[i])>0]
		# print ind
		input_variables = [X_train[i] for i in ind]
		target_variables = [Y_train[i] for i in ind]

		# target_variables_sampled = []
		# for i in xrange(len(target_variables)):
		# 	target_variables_sampled.append(rd.sample(target_variables[i], min(5, len(target_variables[i]))))
		# 	# target_variables_sampled.append(target_variables[i])

		# target_variables = list(target_variables_sampled)
		# print target_variables[0:5]
		# target_variables_trun = []
		# for i in xrange(len(target_variables)):
		# 	target_variables_trun.append([sublist[:seq_len_to_train] for sublist in target_variables[i]])

		# target_variables = target_variables_trun
		# Run the train function
		loss = train(input_variables, target_variables, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, threshold)
		# loss /= np.sum([len(sublist) for main_list in target_variables for sublist in main_list])
		# Keep track of loss
		print_loss_total += loss
		plot_loss_total += loss

		# if epoch == 0: continue
		if epoch % print_every == 0 and epoch != 0 and epoch*batch_size > 10000:
			metrics = generate_predictions(X_val, Y_val, encoder, decoder, threshold)
			pre_avg_cur = np.average([m[0] for m in metrics])
			re_avg_cur = np.average([m[1] for m in metrics])
			f1_avg_cur = np.average([m[2] for m in metrics])
			print "f1 score: ", f1_avg_cur, "precision: ", pre_avg_cur, "recall: ", re_avg_cur, "Samples:", epoch*batch_size
			if f1_avg_cur >= f1_avg:
				f1_avg = f1_avg_cur
				save_model_after_training(encoder, decoder)

			print_loss_avg = print_loss_total / float(epoch)
			print_loss_total = 0
			print_summary = '%s (%d %d%%)' % (time_since(start, epoch / float(n_epochs)), epoch, epoch / float(n_epochs) * 100)
			print(print_summary)

print "finished training"

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
f1_val_score = []
for threshold in thresholds:
	metrics = generate_predictions(X_val, Y_val, is_test=0, abstracts=abstract_test, threshold=threshold)
	avg_precision = np.average([m[0] for m in metrics])
	avg_recall = np.average([m[1] for m in metrics])
	avg_f1 = np.average([m[2] for m in metrics])
	f1_val_score.append(avg_f1)
	print avg_precision, avg_recall, avg_f1

threshold = thresholds[np.argmax(f1_val_score)]
print "best val f1 score: ", np.max(f1_val_score) 
print f1_val_score, 
threshold = 0.5
metrics = generate_predictions(X_test, Y_test, is_test=1, abstracts=abstract_test, threshold=threshold)
avg_precision = np.average([m[0] for m in metrics])
avg_recall = np.average([m[1] for m in metrics])
avg_f1 = np.average([m[2] for m in metrics])

print "Precision: ", avg_precision
print "Recall: ", avg_recall
print "F1 Micro: ", avg_f1
