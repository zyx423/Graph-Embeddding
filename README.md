# GAE-and-VGAE
This is the implementation of paper 'Variational Graph Auto-Encoder' in NIPS Workshop on Bayesian Deep Learning, 2016. 

0. This is my reproduced Graph AutoEncoder （GAE） and variational Graph AutoEncoder (VGAE) by the Pytorch. If you find any errors or questions, please tell me.

1. Task: Unsupervised graph embedding for clustering， classification， and Visualization

2. Algorithms used in the tasks:

      Clustering：k-means; 
      Classification: SVM; 
      Visualization: t-SNE;

3. Requirement: Ptthon 3.7, Pytorch: 1.5 and other pakeages which is illustrated in the code. And the codes can be runned in the windows.

4. This is a code file containing the Graph AutoEncoder（GAE） and variational Graph AutoEncoder (VGAE). 
The purpose of this case is mainly to learn the latent representation, etc, the graph embedding. There are two datasets in this example，i.e., Cora and Yale. If you want to use other datasets, you just need to put your dataset in the "Dataset" folder. Cora is a graph dataset and ATT is a no-graph dataset. The adjacency matrix for the ATT is calculated by the KNN (k=9).

5. If you think my code is helpful to you, please light me a star, thank you.
