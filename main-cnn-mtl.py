import cPickle
from collections import Counter
import os
import networkx as nx
import json
import numpy as np
from model_mtl import Preprocessor, sequenceCNN
import random



random.seed(9001)
# global variables
hops = 2

def recall(precision, recall):
	if recall+precision == 0:
		return 0.0
	return 2*(precision*recall)/(precision+recall)


def get_cochrane_to_cui():
	cui_to_cochrane = json.load(open('../../data/cochrane_to_cui.json', 'r'))
	cochrane_to_cui = {}
	for cui in cui_to_cochrane:
		concepts = cui_to_cochrane[cui]
		for concept in concepts:
			conceptid = concept[0].split("/")[-1]
			cochrane_to_cui[conceptid] = [cui, concept[1]]

	return cochrane_to_cui


def get_graph():
	cui_to_cochrane = json.load(open('../../data/cui-json.txt', 'r'))
	G = cPickle.load(open('../../data/cui_graph_py2.pck','r'))
	return G.to_undirected()

def get_cuis_from_concepts(concepts, cochrane_to_cui):
	cuis = []
	for concept in concepts:
		if concept in cochrane_to_cui:
			cuis.append(cochrane_to_cui[concept][0])

	cuis = list(set(cuis))
	return cuis


def read_cui_embeddings():
	cui_embeddings = {}
	with open('../../data/cuis.embeddings','r') as f:
			data = f.readlines() 
	data = data[1:]
	for line in data:
		line = line.strip().split(" ")
		cui = line[0].strip()
		embedding = [float(l) for l in line[1:]]
		embedding = np.array(embedding)/np.linalg.norm(embedding)
		embedding = embedding.tolist()
		cui_embeddings[cui] = embedding
	return cui_embeddings


def read_word_embeddings():
	wve = {}
	fread1 = open('../../data/vectors.txt', 'r')
	fread2 = open('../../data/types.txt', 'r')
	
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


def prepare_data():
	data = cPickle.load(open('../../data/data_csv.pkl', 'r'))
	cochrane_to_cui = get_cochrane_to_cui()
	prepared_data = []
	dtypes = ['outcome condition', 'outcome classification', 'outcome broader concept'
		, 'intervention classification', 'intervention applied', 'population condition', 'population age'
		, 'metamap population', 'metamap intervention', 'metamap outcome']
	for d in data:
		temp_dict = dict(d)
		for dtype in dtypes:
			l = [a[0] for a in temp_dict[dtype]]
			temp_dict[dtype] = get_cuis_from_concepts(l, cochrane_to_cui)
		prepared_data.append(temp_dict)
	return prepared_data


def preprocess_text(preprocessor, data, train_data, cuis):

	all_texts = [d['population text'] for d in data]
	all_texts += [d['intervention text'] for d in data]
	all_texts += [d['outcome text'] for d in data]

	preprocessor.preprocess(all_texts)

	dtypes = ['outcome condition', 'intervention applied', 'population condition']
	
	all_cuis = {}	
	for d in train_data:
		for dtype in dtypes:
			if dtype not in all_cuis:
				all_cuis[dtype] = []	
			all_cuis[dtype] += d[dtype] 
			

	for dtype in dtypes:
		# all_cuis[dtype] += random.sample(cuis, 100000)
		all_cuis[dtype] = list(set(all_cuis[dtype]))

	popcnt = len(all_cuis['population condition'])
	intcnt = len(all_cuis['intervention applied'])
	outcnt = len(all_cuis['outcome condition'])

	pop_to_ind = {}	
	int_to_ind = {}
	out_to_ind = {}

	for i in xrange(popcnt):
		pop_to_ind[all_cuis['population condition'][i]] = i

	for i in xrange(intcnt):
		int_to_ind[all_cuis['intervention applied'][i]] = i

	for i in xrange(outcnt):
		out_to_ind[all_cuis['outcome condition'][i]] = i

	con_to_ind = {}
	con_to_ind['population condition'] = pop_to_ind
	con_to_ind['intervention applied'] = int_to_ind
	con_to_ind['outcome condition'] = out_to_ind

	cnts = {}
	cnts['population condition'] = popcnt
	cnts['intervention applied'] = intcnt
	cnts['outcome condition'] = outcnt

	return preprocessor, cnts, all_cuis, con_to_ind



def split_data(data, perc):
	size = len(data)
	train_size = int(perc*size)
	indices = random.sample(xrange(size), train_size)
	train_data = [data[i] for i in indices]
	test_data = [data[i] for i in xrange(size) if i not in indices]
	return train_data, test_data


def get_metrics(predicted, manual, G):
	predicted = set(predicted)
	manual = set(manual)
	predicted = set([a for a in predicted if a != "not_found"])
	manual = set([a for a in manual if a != "not_found"])

	correct = len(predicted.intersection(manual))
	
	if len(predicted) == 0:
		precision = 0.0
	else: 
		precision = float(correct)/len(predicted)

	if len(manual) == 0:
		recall = 0.0
	else:
		recall = float(correct)/len(manual)

	corr_prec = 0.0
	corr_recall = 0.0
	for cui1 in predicted:
		for cui2 in manual:
			try:
				if nx.shortest_path_length(G, cui1, cui2) <= hops:
					corr_prec += 1.0
					break
			except nx.exception.NetworkXNoPath:
				corr_prec += 0.0
			except nx.exception.NetworkXError:
				corr_prec += 0.0

	if len(predicted):
		corr_prec /= len(predicted)
	else:
		corr_prec = 0.0

	for cui2 in manual:
		for cui1 in predicted:
			try:
				if nx.shortest_path_length(G, cui1, cui2) <= hops:
					corr_recall += 1.0
					break
			except nx.exception.NetworkXNoPath:
				corr_recall += 0.0
			except nx.exception.NetworkXError:
				corr_recall += 0.0
	
	if len(manual):
		corr_recall /= len(manual)
	else:
		corr_recall = 0.0

	return np.array([precision, recall, corr_prec, corr_recall])



def training(model, train_data, all_cuis, con_to_ind, cnts):
	
	all_texts = []
	pop_gtruth = []
	int_gtruth = []
	out_gtruth = []

	for one_annotation in train_data:
		all_texts += [one_annotation['population text']+ " "+ one_annotation['intervention text']
		+ " " + one_annotation['outcome text']]
		pop_gtruth.append(one_annotation['population condition'])
		int_gtruth.append(one_annotation['intervention applied'])
		out_gtruth.append(one_annotation['outcome condition'])


	X = model.preprocessor_text.build_sequences(all_texts)
	Y1 = []
	Y2 = []
	Y3 = []
	
	print "population concept indexes: ", len(con_to_ind['population condition'].keys())

	for i in xrange(len(train_data)):
		y_pop = [0 for k in range(cnts['population condition'])]
		y_int = [0 for k in range(cnts['intervention applied'])]
		y_out = [0 for k in range(cnts['outcome condition'])]

		for j in xrange(len(pop_gtruth[i])):
			y_pop[con_to_ind['population condition'][pop_gtruth[i][j]]] = 1

		for j in xrange(len(int_gtruth[i])):
			y_int[con_to_ind['intervention applied'][int_gtruth[i][j]]] = 1

		for j in xrange(len(out_gtruth[i])):
			y_out[con_to_ind['outcome condition'][out_gtruth[i][j]]] = 1

		Y1.append(y_pop)
		Y2.append(y_int)
		Y3.append(y_out)

	Y1 = np.array(Y1)
	Y2 = np.array(Y2)
	Y3 = np.array(Y3)

	size = int(0.8*Y1.shape[0]) 
	ind = random.sample(xrange(Y1.shape[0]), size)
	indc = [i for i in xrange(Y1.shape[0]) if i not in ind]
	
	X_train = X[ind, :]
	Y1_train = Y1[ind, :]
	Y2_train = Y2[ind, :]
	Y3_train = Y3[ind, :]


	X_val = X[indc, :]
	Y1_val = Y1[indc, :]
	Y2_val = Y2[indc, :]
	Y3_val = Y3[indc, :]

	model.train(X_train, Y1_train, Y2_train, Y3_train, X_val, Y1_val, Y2_val, Y3_val, nb_epoch=100, batch_size=200)
	return model

def write_prediction_data(test_data, predicted_pop, predicted_int, predicted_out):
	for i in  xrange(len(test_data)):
		test_data[i]['population prediction'] = list(set([a for a in predicted_pop[i] if a != 'not_found']))
		test_data[i]['intervention prediction'] = list(set([a for a in predicted_int[i] if a != 'not_found']))
		test_data[i]['outcome prediction'] = list(set([a for a in predicted_out[i] if a != 'not_found']))

	cPickle.dump(test_data,open('../../data/testdatapred_mtl.pkl','w')) 


def evaluate(model, test_data, G, cui_embeddings, threshold):
	all_texts = []
	pop_gtruth = []
	int_gtruth = []
	out_gtruth = []

	for one_annotation in test_data:
		all_texts += [one_annotation['population text']+ " "+ one_annotation['intervention text']
		+ " " + one_annotation['outcome text']]

		pop_gtruth.append(one_annotation['population condition'])
		int_gtruth.append(one_annotation['intervention applied'])
		out_gtruth.append(one_annotation['outcome condition'])


	X = model.preprocessor_text.build_sequences(all_texts)
	Y1, Y2, Y3 = model.predict(X, batch_size=200)
	

	Y1[Y1>=threshold] = 1
	Y1[Y1<threshold] = 0
	Y2[Y2>=threshold] = 1
	Y2[Y2<threshold] = 0
	Y3[Y3>=threshold] = 1
	Y3[Y3<threshold] = 0

	print "predicted Y1:", Y1.sum(axis=1), "fraction: ", np.sum(Y1.sum(axis=1)==0)/float(Y1.shape[0])
	print "predicted Y2:", Y2.sum(axis=1), "fraction: ", np.sum(Y2.sum(axis=1)==0)/float(Y2.shape[0])

	predicted_pop = [[] for i in xrange(len(test_data))]
	predicted_int = [[] for i in xrange(len(test_data))]
	predicted_out = [[] for i in xrange(len(test_data))]

	for i in xrange(Y1.shape[0]):
		for j in xrange(Y1.shape[1]):
			if Y1[i][j] == 1:
				predicted_pop[i].append(all_cuis['population condition'][j])
				

	for i in xrange(Y2.shape[0]):
		for j in xrange(Y2.shape[1]):
			if Y2[i][j] == 1:
				predicted_int[i].append(all_cuis['intervention applied'][j])


	for i in xrange(Y3.shape[0]):
		for j in xrange(Y3.shape[1]):
			if Y3[i][j] == 1:
				predicted_out[i].append(all_cuis['outcome condition'][j])


	pico_type = ['population condition', 'intervention applied', 'outcome condition']

	data_for_eval = {}

	data_for_eval['population condition'] = [predicted_pop, pop_gtruth]
	data_for_eval['intervention applied'] = [predicted_int, int_gtruth]
	data_for_eval['outcome condition'] = [predicted_out, out_gtruth]

	fwrite = open('../../data/deep_learn_performance_mtl.txt', 'w')
	f1score = 0.0
	for pico in pico_type:
		result = np.zeros(4)
		for i in xrange(len(test_data)):
			result += get_metrics(data_for_eval[pico][0][i], data_for_eval[pico][1][i], G)		

		result /= float(len(test_data))
		f1score += recall(result[0], result[1])
		fwrite.write("%s : Precision: %f Recall: %f Precision-2hops: %f Recall-2hops: %f F1: %f  F1-2hops: %f\n" %(pico, result[0], result[1], result[2], result[3], recall(result[0], result[1]), recall(result[2], result[3])))
		
	fwrite.close()
	write_prediction_data(test_data, predicted_pop, predicted_int, predicted_out)
	return f1score
	

def combine_pop_inter(pop_output, inter_output):
	pop_inter_output = []
	for p,i in zip(pop_output, inter_output):
		pop_inter_output.append(p+i)
	return pop_inter_output

if __name__ == '__main__':
	data = prepare_data()
	wve = read_word_embeddings()
	cuie = read_cui_embeddings()
	train_data, test_data = split_data(data, 0.60)
	train_data, val_data = split_data(train_data, 0.90)
	print "train data size: ",len(train_data), " test data size: ", len(test_data) 
	preprocessor = Preprocessor(max_features=14740, maxlen=300, wvs=wve)
	preprocessor, cnts, all_cuis, con_to_ind = preprocess_text(preprocessor, data, train_data, cuie.keys())
	print "pocnt: ", cnts['population condition'], " intcnt: ", cnts['intervention applied'], "outcnt: ", cnts['outcome condition']
	
	G = get_graph()
	
	model = sequenceCNN(preprocessor, cnts['population condition'], cnts['intervention applied'], cnts['outcome condition'])
	model = training(model, train_data, all_cuis, con_to_ind, cnts)
	thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	f1scores = []
	for threshold in thresholds:
		f1scores.append(evaluate(model, val_data, G, all_cuis, threshold))

	print "thresholds: ", thresholds
	print "chosen threshold: ", thresholds[np.argmax(f1scores)]
	threshold = thresholds[np.argmax(f1scores)]
	evaluate(model, test_data, G, all_cuis, threshold)



