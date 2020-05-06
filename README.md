# RGM: Randomized Grid Mapping for Fast Graph Classification
This is a reference implementation for RGM, an unsupervised method for constructing feature maps for graphs.  RGM characterizes a graph by the distribution of its node's latent features or embeddings in vector space.  The resulting feature maps may be used to perform machine learning tasks such as classification where the data points are graphs.  Here we provide an example use case of RGM for graph classification.

**Usage**: python main.py

**Dependencies**: NumPy, SciPy, scikit-learn, NetworkX

Please refer to our paper for more information, and consider citing it if you find this code useful.  

**Paper**: Mark Heimann, Tara Safavi, and Danai Koutra. <a href="https://gemslab.github.io/papers/heimann-2019-RGM.pdf">Distribution of Node Embeddings as Multiresolution Features for Graphs</a>. IEEE International Conference on Data Mining (ICDM), November 2019.
<!--Link: https://gemslab.github.io/papers/heimann-2019-RGM.pdf-->
<!--<p align=?center?>
<img src=?https://raw.githubusercontent.com/GemsLab/RGM/master/overview.jpg(869 kB)
https://raw.githubusercontent.com/GemsLab/RGM/master/overview.jpg
? width=?700?  alt=?Overview of RGM?>
</p>-->

**Citation (bibtex)**:

```
@inproceedings{heimann2019distribution,
  title={Distribution of Node Embeddings as Multiresolution Features for Graphs},
  author={Heimann, Mark and Safavi, Tara and Koutra, Danai},
  booktitle={2019 IEEE International Conference on Data Mining (ICDM)},
  organization={IEEE},
  year={2019}
}
```
