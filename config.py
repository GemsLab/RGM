import numpy as np, networkx as nx
from scipy import sparse

class EmbeddingMethod():
	def __init__(self, align_info = None, dimensionality=None, max_layer=None, method="xnetmf", alpha = 0.1, normalize = False, abs_val = False, gammastruc = 1, gammaattr = 1, implicit_factorization = True, landmark_features = None, landmark_indices = None, l2l_decomp = None, use_landmarks = False):
		self.method = method #representation learning method
		self.normalize = normalize
		self.abs_val = abs_val
		self.dimensionality = dimensionality #sample p points
		self.max_layer = max_layer #furthest hop distance up to which to compare neighbors
		self.alpha = alpha #discount factor for higher layers
		self.gammastruc = gammastruc
		self.gammaattr = gammaattr
		self.implicit_factorization = implicit_factorization
		self.landmark_features = landmark_features
		self.landmark_indices = None
		self.l2l_decomp = l2l_decomp
		self.use_landmarks = False #use hard coded landmarks

class Graph():
	#Undirected, unweighted
	def __init__(self, adj, node_labels = None, edge_labels = None, graph_label = None, node_attributes = None, node_features = None, graph_id = None):
		self.graph_id = graph_id
		self.adj = adj #adjacency matrix
		self.N = self.adj.shape[0] #number of nodes

		if node_features is None:
			self.node_features = {}
		else:
			self.node_features = node_features
		
		self.max_features = {}
		for feature in self.node_features:
			self.max_features[feature] = np.max(self.node_features[feature])

		self.node_labels = node_labels
		self.edge_labels = edge_labels
		self.graph_label = graph_label
		self.node_attributes = node_attributes #N x A matrix, where N is # of nodes, and A is # of attributes

	#'''
	def set_node_features(self, node_features):
		self.node_features = node_features
		for feature in self.node_features: #TODO hacky to handle this case separately
			self.max_features[feature] = np.max(self.node_features[feature])


	def compute_node_features(self, features_to_compute):
		nx_graph = to_nx(self.adj)
		new_node_features = self.node_features
		if "degree" in features_to_compute:
			total_degrees = nx_graph.degree(nx_graph.nodes())
			new_node_features["degree"] = total_degrees

		self.set_node_features(new_node_features)

	def normalize_node_features(self):
		normalized_features_dict = dict()
		for feature in self.node_features:
			normalized_features = self.node_features[feature]
			if np.min(normalized_features) < 0: #shift so no negative values
				normalized_features += abs(np.min(normalized_features))
			#scale so no feature values less than 1 (for logarithmic binning)
			if np.min(normalized_features) < 1:
				normalized_features /= np.min(normalized_features[normalized_features != 0])
				if np.max(normalized_features) == 1: #e.g. binary features
					normalized_features += 1
				normalized_features[normalized_features == 0] = 1 #set 0 values to 1--bin them in first bucket (smallest values)
			normalized_features_dict[feature] = normalized_features
		self.set_node_features(normalized_features_dict)
	#'''

def to_nx(adj):
	if sparse.issparse(adj):
		return nx.from_scipy_sparse_matrix(adj)
	else:
		return nx.from_numpy_matrix(adj)


