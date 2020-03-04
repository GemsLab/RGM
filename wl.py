
import numpy as np
import time
try:
	import cPickle as pickle 
except ImportError:
	import pickle
import os

def get_wl_labels(graphs, args, label_dict_by_iter = None):
	n_iter = args.wliter
	wl_labels_fname = os.path.join(os.path.dirname(__file__), "reps/%s_wl%d_labels" % (args.dataset, n_iter))
	wl_labeldict_fname = os.path.join(os.path.dirname(__file__), "reps/%s_wl%d_label_dict" % (args.dataset, n_iter))

	#Load in WL labels, if already computed
	if args.loadwl and os.path.exists(wl_labels_fname) and os.path.exists(wl_labeldict_fname):
		print("loading in WL label expansions...")
		with open(wl_labels_fname, "rb") as wl_labels_file:
			wl_labels = pickle.load(wl_labels_file)
		with open(wl_labeldict_fname, "rb") as wl_labeldict_file:
			label_dict_by_iter = pickle.load(wl_labeldict_file)
		return wl_labels, label_dict_by_iter

	#Compute WL labels
	#At each iteration, store mapping of labels to 
	if label_dict_by_iter is None: #create empty list of dicts
		label_dict_by_iter = list()
		for i in range(n_iter):
			label_dict_by_iter.append(dict())

	#Initialize labels to be the node labels
	before_wl_init = time.time()
	wl_labels =  [[] for i in range(n_iter + 1)]
	for j in range(len(graphs)):
		if graphs[j].node_labels is None: 
			graphs[j].node_labels = np.ones(graphs[j].adj.shape[0])
		wl_labels[0].append(graphs[j].node_labels)
	print("WL label expansion time to initialize (iteration 0): ", (time.time() - before_wl_init))



	#Calculate new labels for WL
	for i in range(1, n_iter + 1): #One iteration of WL
		before_wl_iter = time.time()
		label_num = 0 #each distinct combination of neighbors' labels will be assigned a new label, starting from 0 at each iteration
		for j in range(len(graphs)): #for each graph
			graph = graphs[j]
			wl_labels[i].append(list())
			for k in range(graph.adj.shape[0]): #for each node
				neighbors = graph.adj[k].nonzero()[1] #get its neighbors
				neighbor_labels = wl_labels[i - 1][j][neighbors] #get their labels at previous iteration
				#prepend a node's own label, but sort neighbors' labels so that order doesn't matter
				neighbor_labels = np.insert(np.sort(neighbor_labels), 0, wl_labels[i - 1][j][k]) 

				#map these to a unique, order-independent string
				#this is a "label" for the node that is a multiset of its neighbors' labels
				#multiset_label = str(neighbor_labels)
				multiset_label = ''.join(map(str,neighbor_labels))

				#haven't encountered this label at this iteration
				if multiset_label not in label_dict_by_iter[i - 1]:
					#assign this a new numerical label that we haven't used at this iteration
					label_dict_by_iter[i - 1][multiset_label] = ("%d-%d") % (i, label_num) #new labeling number but also iteration number (so that we have all unique labels across iters)
					label_num += 1
				#For this iteration, assign the node a new WL label based on its neighbors' labels
				wl_labels[i][j].append(label_dict_by_iter[i - 1][multiset_label])
			wl_labels[i][j] = np.asarray(wl_labels[i][j])

		print("WL label expansion time at iteration %d: " % i, (time.time() - before_wl_iter))
	
	#Save WL labels
	if args.savewl:
		print("Saving WL label expansions...")
		with open(wl_labels_fname, "wb") as wl_labels_file:
			pickle.dump(wl_labels, wl_labels_file)
		with open(wl_labeldict_fname, "wb") as wl_labeldict_file:
			pickle.dump(label_dict_by_iter, wl_labeldict_file)
	return wl_labels, label_dict_by_iter




