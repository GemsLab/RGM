import numpy as np 
from itertools import product
from collections import defaultdict
from histograms import *
from scipy import sparse
from sklearn.preprocessing import normalize

#Input: length-N list of dict {str : int} of {d-dim coordinates of a cell : count}
#Output: N x D feature matrix, where D is the dimensionality of the feature mapping (# unique strings across all N items)
#NOTE: this is just for a single attribute 
#NOTE 2: will have to create training and test feature mappings together. 
#This seems to be OK, following (Rahimi an Recht, 2007).  
#The alternative is to create very long sparse vectors consisting of all (cell_width)^d possible cells
def combineddim_histogram_feature_map(combineddim_histograms):
	N = len(combineddim_histograms)
	#Get list of all unique non-empty cells
	combined_coordinates = set()
	for i in range(N):
		cells_hist = combineddim_histograms[i].keys()
		for cell in cells_hist:
			if cell not in combined_coordinates:
				combined_coordinates.add(cell)		

	#dimensionality of feature map
	D = len(combined_coordinates)

	#map each cell (coordinates) to a dimension in the feature map
	coordinate_to_featuredim = dict(zip(list(combined_coordinates), range(D)))

	rows = list()
	cols = list()
	data = list()

	#go through all our graphs' histograms...
	for i in range(N):
		for coord in combineddim_histograms[i].keys(): #for each cell
			d = coordinate_to_featuredim[coord] #get the dimension in the feature map
			rows.append(i)
			cols.append(d)
			data.append(combineddim_histograms[i][coord])
	combined_feature_map = sparse.csr_matrix((data, (rows, cols)), shape = (N,D))
	return combined_feature_map

#Input: length-N list of n_i x d dimensional node embeddings for graph i
#Possibly: length-N list of n_i dimensional lists of labels for nodes in graph i
#Possibly: float or d-dimensional numpy array cell width
#Output: N x D feature matrix
def compute_feature_maps(embs, args, labels = None, cell_width = None):
	#Get coordinates for each embeddings
	if cell_width is None: 
		#Get independent values along each dimension sampled from a gamma distribution with specified parameter
		cell_width = np.random.gamma(shape = 2, scale = 1.0/args.gammapitch, size = args.dimensionality)
		print("mean cell width %.3f" % (np.mean(cell_width)))#, cell_width
	offset = (np.random.uniform(size = args.dimensionality)*cell_width).tolist() #for each dimension, uniform between 0 and that dimension's cell width
	
	#Turn embeddings into histograms (with or without labels)
	if labels is not None:
		hists = get_labeled_embed_histograms(embs, args, cell_widths = cell_width, labels = labels, offsets = offset)
	else:
		hists = [get_histogram(emb, args, cell_widths = cell_width, offsets = offset) for emb in embs]

	#Turn histograms into feature maps
	feature_maps = combineddim_histogram_feature_map(hists) #no longer just for combined dim
	return feature_maps

#Compute feature maps (flattening the pyramid match kernel)
#Input: length-N list of n_i x d dimensional node embeddings for graph i
#Output: N x DL feature matrix, where L is the number of levels (args.numlevels)
def pyramid_feature_maps(embs, args, labels = None):
	feature_maps = None
	for level in range(args.numlevels + 1): #Nikolentzos et. al count levels from 0 to L inclusive
		cell_width = np.random.gamma(shape = 2, scale = 1.0/2**(level+1), size = args.dimensionality)
		print("Cell width level %d (mean cell width %.3f)" % (level, np.mean(cell_width)))#, cell_width
		discount_factor = np.sqrt(1.0/2**(args.numlevels - level)) #discount feature maps corresponding to looser grids, as those matches are considered less important in PM
		features_level = discount_factor * compute_feature_maps(embs, args, labels = labels, cell_width = cell_width)
		if feature_maps is None: 
			feature_maps = features_level
		else:
			if sparse.issparse(feature_maps):
				feature_maps = sparse.hstack((feature_maps, features_level), format = "csr")
			else:
				feature_maps = np.hstack((feature_maps, features_level))
	return feature_maps

#Full RGM-P or L with up to P normalized randomized feature maps
#Input: length-N list of n_i x d dimensional node embeddings for graph i
#Output: N x DL feature matrix, where P is the number of trials (args.numrmaps)
def rgm(embs, args, labels = None):
	return pyramid_feature_maps(embs, args, labels = labels)
			





