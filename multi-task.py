from model_mtl import Preprocessor, CNN
from nltk.tokenize import word_tokenize
import cPickle
import random as rd
import numpy as np

def split_data(data, split_ratio):
	ind1 = rd.sample(range(len(data)), int(split_ratio*len(data)))
	ind2 = [i for i in xrange(len(data)) if i not in ind1]

	data1 = [data[i] for i in ind1]
	data2 = [data[i] for i in ind2]

	return data1, data2

def get_data():
	# Reading data from the file
	list_items = cPickle.load(open('pubmed.pkl','r'))
	list_items_train, list_items_test = split_data(list_items, 0.6)
	list_items_train, list_items_val = split_data(list_items_train, 0.8)

	return list_items_train, list_items_test, list_items_val

def extract_data(list_items):
	abstracts = []
	mesh_terms = []
	for list_item in list_items:
		abstracts.append(list_item['abstract'])
		mesh_terms.append([l.lower() for l in list_item['mesh terms']])

	return abstracts, mesh_terms

def get_mesh_to_index_mapping(all_mesh_terms):
	mesh_to_index = {}
	all_mesh_terms = set([l for sublist in all_mesh_terms for l in sublist])
	i = 0
	for mesh_term in all_mesh_terms:
		mesh_to_index[mesh_term] = i
		i += 1

	index_to_mesh = ['null' for i in xrange(len(mesh_to_index))]

	for mesh in mesh_to_index:
		index_to_mesh[mesh_to_index[mesh]] = mesh

	return mesh_to_index, index_to_mesh

def preprocess_text(preprocessor, data):
	all_texts = []
	for d in data:
		all_texts += d

	preprocessor.preprocess(all_texts)
	return preprocessor

def read_word_embeddings():
    ftypes = open('wordembeddings/types.txt', 'r')
    words = ftypes.readlines()
    word_2_vectors = {}
    with open('wordembeddings/vectors.txt','r') as f:
        i = 0
        for line in f:
            word_2_vectors[words[i].strip().lower()] = map(float, line.strip().split(" "))
            i += 1
    return word_2_vectors

def train(model, preprocessor, abstract_train, mesh_terms_train, abstract_val, mesh_terms_val):
	X_train = model.preprocessor_text.build_sequences(abstract_train)
	X_val = model.preprocessor_text.build_sequences(abstract_val)

	Y_train = []
	for i in xrange(len(mesh_terms_train)):
		y_temp = [0]*len(index_to_mesh)
		for mesh_term in mesh_terms_train[i]:
			if mesh_term in mesh_to_index:
				y_temp[mesh_to_index[mesh_term]] = 1
		Y_train.append(y_temp)

	Y_train = np.array(Y_train)

	Y_val = []
	for i in xrange(len(mesh_terms_val)):
		y_temp = [0]*len(index_to_mesh)
		for mesh_term in mesh_terms_val[i]:
			if mesh_term in mesh_to_index:
				y_temp[mesh_to_index[mesh_term]] = 1
		Y_val.append(y_temp)

	Y_val = np.array(Y_val)

	model.train(X_train, Y_train, X_val, Y_val, nb_epoch=50, batch_size=256)

	return model

def predict(model, preprocessor, abstract_test, mesh_terms_test, threshold=0.5):
	X_test = model.preprocessor_text.build_sequences(abstract_test)
	Y_test = model.predict(X_test, batch_size=256)
	Y_test[Y_test>=threshold] = 1
	Y_test[Y_test<threshold] = 0
	Y_test = Y_test.tolist()
	
	mesh_terms_pred = []
	metrics = []
	for i in xrange(len(mesh_terms_test)):
		# print "i: ", i, len(Y_test)
		indices = np.where(Y_test[i])[0]
		predicted_mesh_terms = [index_to_mesh[j] for j in indices]
		mesh_terms_pred.append(predicted_mesh_terms)
		metrics.append(get_metrics(mesh_terms_test[i], predicted_mesh_terms))
	
	cPickle.dump((mesh_terms_test, mesh_terms_pred), open('predictions_mtl_'+str(len(list_items_train))+'.pkl','w'))
	return metrics

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


list_items_train, list_items_test, list_items_val = get_data()

list_items_train = [list_items_train[i] for i in xrange(5000)]

abstract_train, mesh_terms_train = extract_data(list_items_train)
abstract_test, mesh_terms_test = extract_data(list_items_test)
abstract_val, mesh_terms_val = extract_data(list_items_val[0:2000])
# abstract_test, mesh_terms_test = extract_data(list_items_val[0:])


print "number of train items: ", len(abstract_train)
print "number of test items: ", len(abstract_test)
print "number of validation items: ", len(abstract_val)

# mesh_to_index, index_to_mesh = get_mesh_to_index_mapping(mesh_terms_train+mesh_terms_test+mesh_terms_val)

mesh2seq = cPickle.load(open('mesh_to_seq.pkl','r'))
mesh_to_index, index_to_mesh = get_mesh_to_index_mapping([list(set(mesh2seq.keys()))])
print "len: ", len(index_to_mesh)
word_embeddings = read_word_embeddings()
preprocessor = Preprocessor(max_features=50000, maxlen=300, wvs=word_embeddings)
preprocessor = preprocess_text(preprocessor, [abstract_train+abstract_test+abstract_val])

model = CNN(preprocessor, len(index_to_mesh))
model = train(model, preprocessor, abstract_train, mesh_terms_train, abstract_val, mesh_terms_val)

#heldout data

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
f1_val = []
for threshold in thresholds:
	metrics = predict(model, preprocessor, abstract_val, mesh_terms_val, threshold)
	avg_f1 = np.average([m[2] for m in metrics])
	f1_val.append(avg_f1)


print f1_val
threshold = thresholds[int(np.argmax(f1_val))]

print threshold
metrics = predict(model, preprocessor, abstract_test, mesh_terms_test, threshold)

avg_precision = np.average([m[0] for m in metrics])
avg_recall = np.average([m[1] for m in metrics])
avg_f1 = np.average([m[2] for m in metrics])

print "Precision: ", avg_precision
print "Recall: ", avg_recall
print "F1 Micro: ", avg_f1
