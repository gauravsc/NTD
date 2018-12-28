from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Flatten, Merge, Reshape
from keras.activations import softmax
from keras.layers.merge import Dot
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers.core import RepeatVector
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.layers.wrappers import Bidirectional

import cPickle
import numpy as np
from preprocessor import Preprocessor
import random as rd

# hyper-params
input_dim=5
hidden_dim=10
output_length=8
output_dim=20
depth=(4, 5)
split_ratio = 0.6
max_seq_len = 0
code_2_index = {}
index_2_code = {}
y_vocab_size = 0
hidden_size= 64
num_layers = 4
batch_size = 128

# preprocessor args
X_vocab_size = 20000
X_max_len = 70

def softMaxAxis2(x):
    return softmax(x,axis=2)

def softMaxAxis1(x):
    return softmax(x,axis=1)

def softMaxAxis0(x):
    return softmax(x,axis=0)

def build_model(preprocessor):
	inputs = Input(shape=(X_max_len,))

	# encoder network
	x = Embedding(X_vocab_size, 200, input_length=X_max_len, mask_zero=False, weights=preprocessor.init_vectors)(inputs)
	hidden_state_encoder = LSTM(hidden_size, return_sequences=True)(x)
	e = Flatten()(hidden_state_encoder)
	g = Dense(max_seq_len*X_max_len)(e)
	g = Reshape((X_max_len, max_seq_len))(g)
	distribituion_over_input = Activation(softMaxAxis1, name='attention_wts')(g)
	encoder_output_after_attention = Dot(axes=(1,1))([distribituion_over_input, hidden_state_encoder])

	# decoder network
	for _ in range(num_layers):
	    x = LSTM(hidden_size, return_sequences=True)(encoder_output_after_attention)
	hidden_state_output = TimeDistributed(Dense(y_vocab_size+1))(x)
	y = Activation('softmax')(hidden_state_output)
	model = Model(input=[inputs], output=[y])
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model

def train(model, X_train, Y_train, X_val, Y_val, nb_epoch=1, batch_size=128):
	iterations = int(X_train.shape[0]/batch_size)*nb_epoch
	# iterations = 1
	for i in xrange(iterations):
		ind = rd.sample(range(X_train.shape[0],), batch_size)
		X_train_i = X_train[ind, :]
		Y_train_i = process_seq_data([Y_train[i] for i in ind])
		# print "model training started"
		# print Y_train_i.shape
		model.fit([X_train_i], [Y_train_i], batch_size=batch_size, nb_epoch=1, verbose=2)
		# print "model training done"
	return model


def get_not_in_level_nodes():
	not_in_level_nodes = []
	for i in xrange(max_seq_len):
		not_in_level_nodes.append([k for k in xrange(y_vocab_size) if k not in level_nodes[i]])
	return not_in_level_nodes


def get_level_consistent_data(y_pred):
	# print y_pred[0,0,not_in_level_nodes[0]] 

	for i in xrange(max_seq_len):
		y_pred[:,i,not_in_level_nodes[i]] = 0 

	# print y_pred[0,0,not_in_level_nodes[0]] 
	
	return y_pred


def predict(model, X_test, batch_size=128):
	y = []
	for i in xrange(int(X_test.shape[0]/batch_size)+1):
		X_test_i = X_test[i*batch_size:(i+1)*batch_size, :]
		y_pred = model.predict([X_test_i], batch_size=batch_size)
		y_pred = get_level_consistent_data(y_pred)
		y.append(np.argmax(y_pred, axis=2))

	y = np.vstack(y)
	print "y shape: ", y.shape
	return y

def evaluate (Y_test, Y_pred):
	print Y_pred[0,:], Y_pred.shape
	print Y_test[0,:], Y_test.shape
	depth_match = 0.0
	for i in xrange(len(Y_test)):
		for j in xrange(len(Y_test[0])):
			if Y_test[i][j] == Y_pred[i][j]:
				depth_match += 1.0
			else:
				break
	return depth_match/float(len(Y_test))


def read_word_embeddings():
	wve = {}
	fread1 = open('../data/vectors.txt', 'r')
	fread2 = open('../data/types.txt', 'r')
	
	while True:
		word = fread2.readline().strip()
		# print word
		# print fread1.readline().strip().split(" ")
		
		try:
			embedding = [float(l) for l in fread1.readline().strip().split(" ")]
			embedding = np.array(embedding)/np.linalg.norm(embedding)
			embedding = embedding.tolist()
			wve[word] = embedding
		except ValueError:
			print "skipped"

		if not word:
			break
	return wve

def split_data(data, split_ratio):
	ind1 = rd.sample(range(len(data)), int(split_ratio*len(data)))
	ind2 = [i for i in xrange(len(ind1)) if i not in ind1 ]

	data1 = [data[i] for i in ind1]
	data2 = [data[i] for i in ind2]

	return data1, data2


def convert_data_format(list_items):
	global max_seq_len
	abstracts = []
	targets = []
	indices = []

	i = 0
	for item in list_items:
		for seq in item['sequence']:
			indices.append(i)
			abstracts.append(item['abstract'])
			seq_mesh = seq[1].split(".")
			seq_mesh.append('EOS')
			if len(seq_mesh) > max_seq_len:
				max_seq_len = len(seq_mesh)
			targets.append(seq_mesh)
		i += 1

	return abstracts, targets, indices

def get_data(preprocessor):
	list_items = cPickle.load(open('pubmed.pkl','r'))
	
	list_items_train, list_items_test  = split_data(list_items, 0.6)
	list_items_train, list_items_val = split_data(list_items_train, 0.8)

	X_train, Y_train = convert_data_format(list_items_train)
	X_test, Y_test, indices_test = convert_data_format(list_items_test)
	X_val, Y_val = convert_data_format(list_items_val)

	preprocessor.preprocess(X_train+X_test+X_val)

	X_train = preprocessor.build_sequences(X_train)
	X_test = preprocessor.build_sequences(X_test)
	X_val = preprocessor.build_sequences(X_val)

	return (X_train, Y_train, X_test, Y_test, X_val, Y_val, indices_test)

def one_hot_encode(seq):
	list_seq = []
	for i in xrange(len(seq)):
		seq_vertical = np.zeros(y_vocab_size+1)
		seq_vertical[seq[i]] = 1
		list_seq.append(seq_vertical)
	return np.array(list_seq)


def process_seq_data(Y, one_hot=True):
	Y_new = []
	for i in xrange(len(Y)):
		seq = [y_vocab_size]*max_seq_len
		for j in xrange(len(Y[i])):
			seq[j] = code_2_index[Y[i][j]]
		if one_hot:
			seq = one_hot_encode(seq)
		# print "seq sum: ", np.sum(seq, axis=1)
		Y_new.append(seq)
	return np.array(Y_new)
	

def compute_vocab(Y):
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


def extract_exact_sequences(Y, index_2_code):
	seq = []
	print Y.shape, len(index_2_code)
	for i in xrange(Y.shape[0]):
		row = []
		for j in xrange(Y.shape[1]):
			
			if Y[i][j] < y_vocab_size:
				if index_2_code[Y[i][j]] == 'EOS':
					break
				else:
					row.append(index_2_code[Y[i][j]])
			elif Y[i][j] == y_vocab_size:
				row.append('PAD')
		seq.append(row)
	return seq

def evaluate_sequences(test_seq, pred_seq):
	mad = 0.0
	for i in xrange(len(test_seq)):
		ad = 0.0
		print "************"
		print test_seq[i]
		print pred_seq[i]
		print "************"
		len_test_i = len(test_seq[i])
		len_pred_i = len(pred_seq[i])
		for j in xrange(min(len_test_i, len_pred_i)):
			if test_seq[i][j] == pred_seq[i][j]:
				ad += 1.0
			else:
				break
		ad = float(ad)/len_test_i
		mad += ad
	return mad/len(test_seq)


def get_node_levels(Y_all):
	level_nodes = [{} for i in xrange(max_seq_len)]
	for one_seq in Y_all:
		for i in xrange(len(one_seq)-1):
			level_nodes[i][code_2_index[one_seq[i]]] = 1

	for i in xrange(len(level_nodes)):
		level_nodes[i][code_2_index['EOS']] = 1

	return level_nodes

def get_attention_weights(model, X_test, Y_test, preprocessor):
	X_test = X_test[0:40]
	Y_test = Y_test[0:40]
	attention_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer("attention_wts").output)
	attention_output = attention_layer_model.predict(X_test)
	cPickle.dump((attention_output, X_test, Y_test), open('attention_output.pkl','w'))



wve = read_word_embeddings()
preprocessor = Preprocessor(max_features=X_vocab_size, maxlen=X_max_len, wvs=wve)
X_train, Y_train, X_test, Y_test, X_val, Y_val, indices_test = get_data(preprocessor)

code_2_index, index_2_code = compute_vocab(Y_train+Y_test+Y_val)
y_vocab_size = len(code_2_index.keys())

Y_all = Y_train+Y_test
level_nodes = get_node_levels(Y_all)
not_in_level_nodes = get_not_in_level_nodes()

X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)
model = build_model(preprocessor)
print "*********model built*********"

# cPickle.dump(X_test, open('xtest.pkl','w'))

model = train(model, X_train, Y_train, X_val, Y_val, nb_epoch=20, batch_size=batch_size)

get_attention_weights(model, X_test, Y_test, preprocessor)

Y_pred = predict(model, X_test, batch_size=batch_size)

Y_test = process_seq_data(Y_test, one_hot=False)

test_seq = extract_exact_sequences(Y_test, index_2_code)
pred_seq = extract_exact_sequences(Y_pred, index_2_code)

print evaluate_sequences(test_seq, pred_seq)

# eval_results = evaluate(Y_test, Y_pred)
# print eval_results

