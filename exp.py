import cPickle
import numpy as np
import operator
import random as rd
# prev_out, out, target = cPickle.load(open('dump.pkl','r'))


# for i in xrange(len(prev_out)):
# 	if target[i]==1: 
# 		print prev_out[i], out[i], target[i]


# for i in xrange(len(prev_out)):
# 	print prev_out[i], out[i], target[i]

# all_mesh_terms = []
# all_mesh_terms += [l for sublist in mesh_terms_train for l in sublist]
# all_mesh_terms += [l for sublist in mesh_terms_test for l in sublist]
# all_mesh_terms += [l for sublist in mesh_terms_val for l in sublist]
# all_mesh_terms = set(all_mesh_terms)
# print "distinct mesh terms: ", len(list(all_mesh_terms))


# num_mesh_terms = []
# num_mesh_terms += [len(sublist) for sublist in mesh_terms_train]
# num_mesh_terms += [len(sublist) for sublist in mesh_terms_test]
# num_mesh_terms += [len(sublist) for sublist in mesh_terms_val]
# print np.sum(num_mesh_terms)/float((len(mesh_terms_train)+len(mesh_terms_test)+len(mesh_terms_val)))




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

def split_data(data, split_ratio):
	ind1 = rd.sample(range(len(data)), int(split_ratio*len(data)))
	ind2 = [i for i in xrange(len(data)) if i not in ind1]

	data1 = [data[i] for i in ind1]
	data2 = [data[i] for i in ind2]

	return data1, data2


list_items_train, list_items_test, list_items_val = get_data()
abstract_train, mesh_terms_train = extract_data(list_items_train)

word_count = {}

for mesh_terms in mesh_terms_train:
	for mesh in list(set(mesh_terms)):
		if mesh not in word_count:
			word_count[mesh] = 1
		else:
			word_count[mesh] += 1


sorted_x = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
