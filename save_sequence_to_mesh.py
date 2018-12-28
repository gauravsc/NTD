import cPickle

def get_mesh_seq_map():
	f = open('mtrees2017.bin', 'r')
	data = f.readlines()
	code_to_index = {}
	mesh_to_seq = {}
	seq_to_mesh = {}
	for d in data:
		terms, seq = d.strip().split(";")
		terms = terms.split(",")
		for t in terms: 
			
			seq_1 = seq.split(".")
			seq_1 =  [":".join(seq_1[:i+1]) for i in xrange(len(seq_1))]
			seq_1_str = ".".join(seq_1)
			seq_to_mesh[seq_1_str] = t.lower()
			mesh_to_seq[t.lower()] = seq_1_str

	f.close()

	sequences = list(set(mesh_to_seq.values()))
	
	codes = []
	for i in xrange(len(sequences)):
		codes += sequences[i].split(".")

	codes += ["SSOS", "SOS", "EOS"]
	index_to_codes = list(set(codes))

	for i in xrange(len(index_to_codes)):
		code_to_index[index_to_codes[i]] = i
		# print index_to_codes[i], i, code_to_index[index_to_codes[i]]

	print len(code_to_index), len(index_to_codes)
	return mesh_to_seq, seq_to_mesh, code_to_index, index_to_codes

mesh_to_seq, seq_to_mesh, code_to_index , index_to_codes= get_mesh_seq_map()

cPickle.dump(seq_to_mesh, open('seq_to_mesh.pkl','w'))
cPickle.dump(mesh_to_seq, open('mesh_to_seq.pkl', 'w'))
cPickle.dump(code_to_index, open('code_2_index.pkl','w'))
cPickle.dump(index_to_codes, open('index_2_code.pkl','w'))


sequences = mesh_to_seq.values()
sequences = [seq.split(".") for seq in sequences]
len_seqs = [len(s) for s in sequences]
print np.unique(len_seqs, return_counts=True)
