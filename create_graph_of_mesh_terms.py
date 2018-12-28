import cPickle
import networkx as nx


def make_tree(sequences):

	G = nx.Graph()
	
	for sequence in sequences:
		for i in xrange(len(sequence)-1):
			G.add_edge(code_2_index[sequence[i]], code_2_index[sequence[i+1]])

	return G

mesh_to_seq = cPickle.load(open('mesh_to_seq.pkl','r'))
sequences = mesh_to_seq.values()
sequences = [['SSOS','SOS']+sequence.split(".")+['EOS'] for sequence in sequences]

code_2_index = cPickle.load(open('code_2_index.pkl', 'r'))
G = make_tree(sequences)
nx.write_adjlist(G, './mesh_codes_tree.adj')