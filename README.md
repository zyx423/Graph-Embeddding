These are the graph embedding methods that I reproduce.

1. There are four tasks used to evaluate the effect of embeddings, i.e., node clustering, node classification, link_prediction,  and graph Visualization. Algorithms used in the tasks:

* Clustering：k-means;
* Classification: SVM;
* Link_Prediction;
* Visualization: t-SNE;

2. Requirement: Python 3.7, Pytorch: 1.5 and other pakeages which is illustrated in the code.
   
3. There are two types of datasets: graph datasets(cora, citeseer, pubmed) and no-graph datasets(att,imm,umist,orl).   If you want to use other datasets, you just need to put your dataset in the "Dataset" folder.   The adjacency matrix for the no-graph datasets is calculated by the KNN (k=9).

3. The reproduced algorithm for paper is as follows:

* **GAE_VGAE:** T. N. Kipf, M. Welling, "Variational graph auto-encoders," arXiv preprint arXiv:1611.07308, 2016.
* **LGAE_LVGAE:** G. Salha, R. Hennequin, and   M. Vazirgiannis, "Keep it simple: Graph autoencoders without graph  convolutional networks," arXiv preprint arXiv:1910.00942, 2019.
* **SDNE:** D. Wang, P. Cui, and W. Zhu,  "Structural deep network embedding," in Proc. IEEE Conf. Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and
  data mining, 2016, pp. 1225-1234.
