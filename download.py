from Bio import Entrez
import cPickle
import csv
import random as rd

Entrez.email = "Your.Name.Here@example.org"

max_return = 50000

handle = Entrez.esearch(db="pubmed", retmax=max_return, term="Randomized Controlled Trial[Publication Type]", idtype="acc")
id_list = Entrez.read(handle)['IdList']
handle.close()


def get_mesh_seq_map():
	f = open('mtrees2017.bin', 'r')
	data = f.readlines()
	seq_dict = {}

	for d in data:
		terms, seq = d.strip().split(";")
		terms = terms.split(",")
		for t in terms: 
			seq_dict[t.lower()] = seq

	f.close()
	return seq_dict


num_iterations = 100
step_size = max_return/num_iterations
pubmed_data = []
seq_dict = get_mesh_seq_map()
pubmed_ids = []

ind_ids = rd.sample(range(len(id_list)), max_return)
id_list = [id_list[i] for i in ind_ids]

it = 0
while it < num_iterations:
	try: 
		handle = Entrez.efetch(db="pubmed", id=",".join(id_list[it*step_size:(it+1)*step_size]), rettype="null", retmode="xml")
		records = Entrez.read(handle)
		handle.close()
		ids = id_list[it*step_size:(it+1)*step_size]
	except:
		it += 1
		continue
	for i in xrange(len(records['PubmedArticle'])):
		pub_dict = {}
		if 'Abstract' in records['PubmedArticle'][i][u'MedlineCitation']['Article']:
			pubmed_ids.append(ids[i])
			pub_dict['abstract'] = records['PubmedArticle'][i][u'MedlineCitation']['Article']['Abstract']['AbstractText'][0].encode('utf8')
			# print records['PubmedArticle'][i][u'MedlineCitation']['Article']['Abstract']['AbstractText'][0]
			if  'MeshHeadingList' in records['PubmedArticle'][0]['MedlineCitation']:
				mesh_terms = [str(mesh_term['DescriptorName']).encode('utf8') for mesh_term in records['PubmedArticle'][0]['MedlineCitation']['MeshHeadingList']]
				pub_dict['mesh terms'] = mesh_terms
				mesh_terms = [m.split(",") for m in mesh_terms]
				mesh_terms = [item for sublist in mesh_terms for item in sublist]
				
				seq = []
				for m in mesh_terms:
					m = m.strip().lower()
					if m in seq_dict:
						seq.append((m, seq_dict[m]))
				pub_dict['sequence'] = seq
				print ids[i]
				pubmed_data.append(pub_dict)

	if it % 100 == 0:
		cPickle.dump(pubmed_data, open('pubmed.pkl','w'))
		cPickle.dump(pubmed_ids, open("pubmed_ids.pkl",'w'))

	it += 1

keys = pubmed_data[0].keys()

with open('pubmed.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(pubmed_data)

cPickle.dump(pubmed_data, open('pubmed.pkl','w'))
cPickle.dump(pubmed_ids, open("pubmed_ids.pkl",'w'))
		
