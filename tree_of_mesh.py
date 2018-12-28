import cPickle



class Tree:
	def __init__(self, name):
		self.children = {}
		self.name = name



def generate_tree(sequences):
	root = None
	for sequence in sequences:
		curr = None
		prev = None
		for elem in sequence:
			if curr == None:
				if root == None:
					root = Tree(elem)
					curr = root
				else: 
					if elem == root.name:
						curr = root
			else:
				if elem in curr.children:
					prev = curr
					curr = curr.children[elem]
				else:
					curr.children[elem] = Tree(elem)
					prev = curr
					curr = curr.children[elem]
	return root
				


def get_node_children(sequences):

	# sequences = [s for sublist in sequences for s in sublist]

	node_2_children = {}
	nodes = []

	for sequence in sequences:
		nodes += sequence
	
	nodes = list(set(nodes))

	for node in nodes:
		node_2_children[node] = []		 	

	for sequence in sequences:
		for i in xrange(len(sequence)-1):
			node_2_children[sequence[i]].append(sequence[i+1])

	for node in nodes:
		node_2_children[node] = list(set(node_2_children[node]))

	return node_2_children



