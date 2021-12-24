
This is the implementation of paper 'Variational Graph Auto-Encoder' in NIPS Workshop on Bayesian Deep Learning, 2016. 

1. This is my reproduced Graph AutoEncoder （GAE） and variational Graph AutoEncoder (VGAE) by the Pytorch. If you find any errors or questions, please tell me.

2. There are four tasks used to evaluate the effect of graph embedding, i.e., node clustering, node classification, link_prediction, and graph Visualization.

3. Algorithms used in the tasks:

      Clustering：k-means; 
      Classification: SVM; 
      Link_Prediction;
      Visualization: t-SNE;

4. Requirement: Python 3.7, Pytorch: 1.5 and other pakeages which is illustrated in the code. And the codes can be runned in the windows.

5. This is a code file containing the Graph AutoEncoder（GAE） and variational Graph AutoEncoder (VGAE). 
The purpose of this case is mainly to learn the latent representation, etc, the graph embedding. There are two datasets in this example，i.e., Cora and ATT. If you want to use other datasets, you just need to put your dataset in the "Dataset" folder. Cora is a graph dataset and ATT is a no-graph dataset. The adjacency matrix for the ATT is calculated by the KNN (k=9).

6. If you think my code is helpful to you, please light me a star, thank you.
