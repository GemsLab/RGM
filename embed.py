import numpy as np
from scipy.sparse import linalg as sp_linalg

try:
	import cPickle as pickle
except ImportError:
	import pickle
import os, sys, time
from utils import *
from config import *

from xnetmf import get_representations as xnetmf_embed
#Get train and test embeddings separately
def get_emb_inductive(emb_method, args, graphs = None, train_indices = None, test_indices = None, emb_dir = None, fold = None, fold_order = None, graph_labels = None):
	if (args.saveembed or args.loadembed):
		train_emb_fname = (emb_dir + "/fold%d_train" % fold)
		test_emb_fname = (emb_dir + "/fold%d_test" % fold)
		labels_fname = emb_dir + "/graph_labels"
		order_fname = emb_dir + "/fold_order" #IDs in each dataset (fold rotates through these)

	
	'''use precomputed embeddings...'''
	#TODO figure out how to save and load node labels
	if args.loadembed and os.path.exists(train_emb_fname) and os.path.exists(test_emb_fname):
		print("loading in embeddings...")

		with open(train_emb_fname, "rb") as trf:
			train_embs = pickle.load(trf)
		with open(test_emb_fname, "rb") as tef:
			test_embs = pickle.load(tef)
	else:
		if args.loadembed: #tried to load embeddings but they weren't there
			print("Could not load embeddings.  Learning embeddings...")

		#Get raw graphs
		if graphs is None:
			dataset_name = dataset_lookup[args.dataset] #inclined_utils.py
			graphs = read_combined(dataset_name)

		individual = (emb_method.method in ["eigenvector", "rpf"])
		'''learn embeddings'''
		train_adjs = graphs[train_indices]
		test_adjs = graphs[test_indices]
		before_embed = time.time()

		'''embed training graphs'''
		if emb_method.method == "xnetmf": #only get landmark embs if using xNetMF TODO hacky
			train_embs, emb_method_with_landmarks, train_max_features = multi_network_embeddings(train_adjs, emb_method, individual=individual)#[learn_embeddings(graph, emb_method) for graph in train_adjs]
		else:
			train_embs, emb_method_with_landmarks, train_max_features = multi_network_embeddings(train_adjs, emb_method, individual=individual)#[learn_embeddings(graph, emb_method) for graph in train_adjs]			
		if not args.noninductive:
			emb_method_with_landmarks.use_landmarks = True
		
		'''embed test graphs'''
		test_embs, rm, md = multi_network_embeddings(test_adjs, emb_method_with_landmarks, individual=individual, max_features = train_max_features)#[learn_embeddings(graph, emb_method) for graph in test_adjs]
		after_embed = time.time()
		print("learned embeddings in time: ", after_embed - before_embed)		

		'''save embeddings as required so dn't have to recompute'''
		if args.saveembed:
			print("saving training embeddings to %s and test to %s" % (train_emb_fname, test_emb_fname))
			dataset_dir = os.path.join(os.path.dirname(__file__), "reps/%s" % args.dataset)
			if not os.path.isdir(dataset_dir):
				os.system("mkdir %s" % dataset_dir)
			if not os.path.isdir(emb_dir):
				os.system("mkdir %s" % emb_dir)
			with open(train_emb_fname, 'wb') as trf:
				pickle.dump(train_embs, trf)
			with open(test_emb_fname, 'wb') as tef:
				pickle.dump(test_embs, tef)
			with open(labels_fname, 'wb') as lf:
				pickle.dump(graph_labels, lf)
			with open(order_fname, 'wb') as ordf:
				pickle.dump(fold_order, ordf)
	return train_embs, test_embs

#Get all embeddings at the same time
def get_emb_transductive(emb_method, args, emb_dir = ".", graphs = None):

	if (args.saveembed or args.loadembed):
		if args.method == "rpf":
			emb_fname = os.path.join(os.path.dirname(emb_dir), "rpf") #get RPF embeddings from up front (not specific to a trial TODO EIG could be like this too)
			emb_dir = os.path.join(os.path.dirname(emb_dir), "xnetmf-trial%d" % args.randomseed) #get graph labels and fold order from xNetMF (TODO these don't need to be emb specific)
		else:
			emb_fname = (emb_dir + "/emb")
		labels_fname = emb_dir + "/graph_labels"
		order_fname = emb_dir + "/fold_order" #IDs in each dataset (fold rotates through these)

	'''use precomputed embeddings...'''
	#TODO figure out how to save and load node labels
	if args.loadembed and os.path.exists(emb_fname) and os.path.exists(labels_fname):
		print("loading in embeddings...")

		with open(emb_fname, "rb") as trf:
			embs = pickle.load(trf)
		with open(labels_fname, "rb") as lf:
			graph_labels = pickle.load(lf)

	else:
		if args.loadembed: #tried to load embeddings but they weren't there
			print("Could not load embeddings.  Learning embeddings...")

		#Get raw graphs
		if graphs is None:
			dataset_name = dataset_lookup[args.dataset] #inclined_utils.py
			graphs = read_combined(dataset_name)
		graph_labels = np.asarray([G.graph_label for G in graphs]) #labels of graphs

		individual = (emb_method.method == "eigenvector")
		'''embed all graphs'''
		before_embed = time.time()
		if emb_method.method == "xnetmf": #only get landmark embs if using xNetMF TODO hacky
			embs, _, _ = multi_network_embeddings(graphs, emb_method, individual=individual)
		else:
			embs, _, _ = multi_network_embeddings(graphs, emb_method, individual=individual)
		after_embed = time.time()
		print("learned embeddings in time: ", after_embed - before_embed)	

		print(len(embs))

		'''save embeddings as required so dn't have to recompute'''
		if args.saveembed:
			print("saving embeddings to %s" % emb_fname)
			dataset_dir = os.path.join(os.path.dirname(__file__), "reps/%s" % args.dataset)
			if not os.path.isdir(dataset_dir):
				os.system("mkdir %s" % dataset_dir)
			if not os.path.isdir(emb_dir):
				os.system("mkdir %s" % emb_dir)
			with open(emb_fname, 'wb') as rf:
				pickle.dump(embs, rf)
			with open(labels_fname, 'wb') as lf:
			 	pickle.dump(graph_labels, lf)
	return embs, graph_labels

def multi_network_embeddings(graphs, emb_method, individual = True, max_features = None):
	if individual: #learn embeddings on graphs individually
		embs = [learn_embeddings(graph, emb_method)[0] for graph in graphs]
		return embs, emb_method, max_features
	else:
		embs = list()
		#Combine graphs into one big adjacency matrix
		combined_graph, dim_starts = create_combined_graph(graphs, emb_method)
		combined_graph.compute_node_features(["degree"])

		#this is necessary for xNetMF to determine the binning
		if max_features is not None:
			combined_graph.max_features = max_features
		else:
			max_features = combined_graph.max_features

		#Embed combined graph
		landmark_embs = None
		combined_embs, emb_method_with_landmarks = learn_embeddings(combined_graph, emb_method)
		
		if not emb_method.use_landmarks: #we've chosen new landmarks
			landmark_embs = combined_embs[emb_method_with_landmarks.landmark_indices]

		#Split into embeddings for individual matrices
		for i in range(len(graphs)):
			emb = combined_embs[dim_starts[i]:dim_starts[i + 1]]
			embs.append(emb)

		return embs, emb_method_with_landmarks, max_features

#Node embedding
def learn_embeddings(graph, emb_method):
	method = emb_method.method.lower()
	if method == "xnetmf":
		embeddings = xnetmf_embed(graph, emb_method, verbose = False)
	elif method == "eigenvector":
		try:
			k = min(emb_method.dimensionality, graph.N - 2) #can only find N - 2 eigenvectors
			eigvals, eigvecs = sp_linalg.eigsh(graph.adj.asfptype(), k = k)
			while eigvecs.shape[1] < emb_method.dimensionality:
				eigvecs = np.concatenate((eigvecs, eigvecs[:,-1].reshape((eigvecs.shape[0], 1))), axis = 1)
			eigvals = eigvals[:emb_method.dimensionality]
			eigvecs = eigvecs[:,:emb_method.dimensionality] 
		except Exception as e:
			print(e)
			eigvals, eigvecs = np.linalg.eig(graph.adj.todense())

			#append smallest eigenvector repeatedly if there are fewer eienvalues than embedding dimension
			while eigvecs.shape[1] < emb_method.dimensionality:
				eigvals = np.concatenate((eigvals, np.asarray([eigvals[-1]])))
				eigvecs = np.concatenate((eigvecs, eigvecs[:,-1].reshape((eigvecs.shape[0], 1))), axis = 1)

			eigvecs = eigvecs[:,np.argsort(-1*np.abs(eigvals))] #to match MATLAB
			eigvals = eigvals[:emb_method.dimensionality]
			eigvecs = eigvecs[:,:emb_method.dimensionality] 


		embeddings = np.abs(eigvecs)
	elif method == "rpf":
		embeddings = rpf(graph.adj, walk_length = emb_method.dimensionality)
	else:
		raise ValueError("Method %s not implemented yet" % method)

	#normalize, for graph similarity
	if emb_method.normalize:
		norms = np.linalg.norm(embeddings, axis = 1).reshape((embeddings.shape[0],1))
		norms[norms == 0] = 1 
		embeddings = embeddings / norms

	if emb_method.abs_val:
		embeddings = np.abs(embeddings)

	return embeddings, emb_method