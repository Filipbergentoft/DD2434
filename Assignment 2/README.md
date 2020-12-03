## List of Questions of Assignment 2

For the below questions, we provided you the data and some helper functions. 
You need to use the given data, but you don't have to use the provided codes. 
You can modify the helper codes however you want. 

In order to use the provided data easily, we suggest you to use Python. 
You can simply load Pickle files (.pkl) and access all the components of data. 
If you don't want to use Pickle, you can use the provided Numpy (.npy) and .txt files to access the same properties. 
Let us know if you face any problems. 

In addition, you can use other packages such as NetworkX, ete3 to work with the tree and graph structures. 
If you use codes from external resources, don't forget to cite them! 

We encourage you to run *Tree.py* to get familiar with the basic functionality of the provided codes.  

**2.2 Likelihood of a Tree Graphical Model**

For question Q2.2.8, we provided three different trees (small, medium, large). Each tree has 5 samples. 

We want you to report the likelihoods of samples (15 values in total).

* *data/q2_2*: A folder to store all data 
* *2_2.py*: Function templates for students 
* *Tree.py*: Set of important components (Node, Tree and TreeMixture Classes) and their functions

**2.4 Mixture of Trees with Observable Variables**

For question Q2.4.13, we provided a tree mixture (q2_4_tree_mixture). The mixture has 3 trees. Each tree has 5 nodes. We generated 100 samples from this mixture.
 
After running your EM algorithm, we want you to report the likelihoods of the true and inferred mixture. 
In addition, we want you to report the Robinson-Foulds distance between the true trees and the inferred ones (3x3=9 values).

* *data/q2_4*: A folder to store all data 
* *2_4.py*, : Function templates for students 
* *Tree.py*: Set of important components (Node, Tree and TreeMixture Classes) and their functions
* *Kruskal_v1.py* and *Kruskal_v2.py*: Two different implementations of Kruskal's algorithm and modifications for maximum spanning tree
* *Phylogeny.py*: Usage of DendroPy module for Robinson-Foulds (RF) distance comparison

