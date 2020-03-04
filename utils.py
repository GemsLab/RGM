import numpy as np, networkx as nx
import scipy.io as sio
from config import *
import scipy.sparse as sp
from scipy.sparse import coo_matrix

dataset_lookup = {"mutag" : "MUTAG", 
"nci" : "NCI1", 
"ptc" : "PTC_MR",  
"imdb-b": "IMDB-BINARY",
"imdb-m" : "IMDB-MULTI",
"collab": "COLLAB"}

#Input: list of n embs of shape m_1 x d, m_2 x d, ...m_n x d
#Output: combined matrix of shape (sum_i = 1 to n m_i) x d
def combine_embs(embs):
	combined_embs = embs[0]
	for i in range(1, len(embs)):
		combined_embs = np.vstack((combined_embs, embs[i]))
	return combined_embs

#Combine multiple graphs into one big block diagonal graph
#Handles sparse graphs
def create_combined_graph(graphs, emb_method):
	dim_starts = [0] #where to start new graph
	for g in graphs:
		dim_starts.append(g.N + dim_starts[-1])

	combined_row = np.asarray([])
	combined_col = np.asarray([])
	combined_data = np.asarray([])

	combined_node_labels = None
	combined_edge_labels = None

	combined_edgelabel_row = np.asarray([])
	combined_edgelabel_col = np.asarray([])
	combined_edgelabel_data = np.asarray([])

	for i in range(len(graphs)):	
		adj = graphs[i].adj.tocoo()
		combined_row = np.concatenate((combined_row, adj.row + dim_starts[i]))
		combined_col = np.concatenate((combined_col, adj.col + dim_starts[i]))
		combined_data = np.concatenate((combined_data, adj.data))

		if graphs[i].edge_labels is not None:
			#add edge labels
			edge_labels = graphs[i].edge_labels.tocoo()
			combined_edgelabel_row = np.concatenate((combined_edgelabel_row, edge_labels.row + dim_starts[i]))
			combined_edgelabel_col = np.concatenate((combined_edgelabel_col, edge_labels.col + dim_starts[i]))
			combined_edgelabel_data = np.concatenate((combined_edgelabel_data, edge_labels.data))

			#add node label data
			if graphs[i].node_labels is not None:
				if combined_node_labels is None:
					combined_node_labels = graphs[i].node_labels
				else:
					combined_node_labels = np.concatenate((combined_node_labels, graphs[i].node_labels))

	combined_shape = (dim_starts[-1], dim_starts[-1])
	combined_adj = coo_matrix((combined_data, (combined_row, combined_col)), shape = combined_shape).tocsr()

	if combined_edgelabel_data.size > 0: #we have edge labels
		combined_edge_labels = coo_matrix((combined_edgelabel_data, (combined_edgelabel_row, combined_edgelabel_col)), shape = combined_shape).tocsr()

	#use node label as attribute
	combined_graph = Graph(combined_adj, node_labels = combined_node_labels, edge_labels = combined_edge_labels, node_attributes = combined_node_labels)

	return combined_graph, dim_starts

#Input list of graphs
#Output: array of combined node labels
def combine_labels(graphs):
	combined_labels = list()
	for graph in graphs:
		combined_labels += graph.node_labels
	return combined_labels
