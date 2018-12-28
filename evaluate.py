import cPickle
import networkx as nx


def make_tree(sequences):

	G = nx.Graph()
	
	for sequence in sequences:
		for i in xrange(len(sequence)-1):
			G.add_edge(code_2_index[sequence[i]], code_2_index[sequence[i+1]])

	return G

def compute_min_dist(true_nodes_list, pred_nodes_list):
	dist_final_out = 0
	for i in xrange(len(true_nodes_list)):
		dist_final = 0
		true_nodes = true_nodes_list[i]
		pred_nodes = pred_nodes_list[i]
		for true_node in true_nodes:
			min_dist = 200
			for pred_node in pred_nodes:
				# print true_node, pred_node
				dist = nx.shortest_path_length(G,source=true_node,target=pred_node)
				if dist < min_dist:
					min_dist = dist
			# if min_dist >10:
			# 	print true_node, pred_nodes
			# print min_dist
			dist_final += min_dist
		dist_final_out += dist_final/len(true_nodes)	
	return dist_final_out, dist_final_out/float(len(true_nodes_list))


mesh_to_seq = cPickle.load(open('mesh_to_seq.pkl','r'))
sequences = mesh_to_seq.values()
sequences = [['SSOS','SOS']+sequence.split(".")+['EOS'] for sequence in sequences]

code_2_index = cPickle.load(open('code_2_index.pkl', 'r'))
G = make_tree(sequences)

filenames =['predictions_decoder_2500.pkl']
# filenames = ['predictions_mtl_2500.pkl', 'predictions_bioasq.pkl', 'predictions_decoder_2500.pkl']
for filename in filenames:
	true_mesh_terms, pred_mesh_terms = cPickle.load(open(filename,'r'))
	true_mesh_terms = true_mesh_terms
	pred_mesh_terms = pred_mesh_terms
	true_nodes = []
	pred_nodes = []
	for  i in xrange(len(true_mesh_terms)):
		mesh_list = true_mesh_terms[i]
		true_nodes.append([code_2_index[mesh_to_seq[mesh].split(".")[-1]] if mesh in mesh_to_seq else 9165 for mesh in mesh_list])
	for i in xrange(len(pred_mesh_terms)):
		mesh_list = pred_mesh_terms[i]
		pred_nodes.append([code_2_index[mesh_to_seq[mesh].split(".")[-1]]  for mesh in mesh_list if mesh in mesh_to_seq])

	measure_sum, measure_avg = compute_min_dist(true_nodes, pred_nodes)
	print filename, measure_sum, measure_avg

# # filenames = ['predictions_decoder.pkl']
# for filename in filenames:
# 	true_mesh_terms, pred_mesh_terms = cPickle.load(open(filename,'r'))
# 	true_mesh_terms = true_mesh_terms[:10000]
# 	pred_mesh_terms = pred_mesh_terms[:10000]
# 	true_nodes = []
# 	pred_nodes = []
# 	for  i in xrange(len(true_mesh_terms)):
# 		mesh_list = true_mesh_terms[i]
# 		true_nodes.append([code_2_index[mesh_to_seq[mesh].split(".")[-1]] if mesh in mesh_to_seq else 9165 for mesh in mesh_list])
# 	for i in xrange(len(pred_mesh_terms)):
# 		mesh_list = pred_mesh_terms[i]
# 		pred_nodes.append([mesh[-2] if len(mesh)>1 else 9165  for mesh in mesh_list])


# 	measure_sum, measure_avg = compute_min_dist(true_nodes, pred_nodes)
# 	print filename, measure_sum, measure_avg
