""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_2_small_tree, q_2_2_medium_tree, q_2_2_large_tree).
    Each tree has 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.

    Note: The alphabet "K" is K={0,1,2,3,4}.
"""

from os import DirEntry
import numpy as np
from Tree import Tree
from Tree import Node
import os


def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.
    """


    # tree_topology: index corresponds to node, element corresponds to parent of that node, nan := root node
    # K defines the number of nodes in the tree
    K = np.asarray(theta).shape[1]
    number_of_nodes = np.asarray(theta).shape[0]
    node_ids = range(number_of_nodes-1,-1,-1)

    # Matrix of probabilities, each row coresponds to a node
    S = np.empty((number_of_nodes,K))
    
    # Walking backwards in the nodes
    for node_id in node_ids:

        # If leaf, do this:
        if node_id not in tree_topology:
            
            leaf_value = np.int(beta[node_id])
            s = np.zeros((K))
            # Setting the probability of the observed variable to 1
            s[leaf_value] = 1
            S[node_id,:] = s

        # If not leaf, do this:
        else:
            # Using the structure of tree_topology to find the node's children            
            children_node_id = np.where(tree_topology == node_id)[0]
            s = np.ones(K)

            for child_node_id in children_node_id:
                theta_child = np.stack(theta[child_node_id],axis=0)
                s_child = S[child_node_id,:]
                prob_child = np.matmul(theta_child, s_child)
                s = np.multiply(s, prob_child)
            S[node_id,:] = s

    s_root = S[0,:]
    theta_root = np.stack(theta[0],axis=0)
    likelihood = np.dot(s_root, theta_root)
    # End: Example Code Segment

    return likelihood


def main():
    print("\n1. Load tree data from file and print it\n")
    directory = '/Users/filipbergentoft/Desktop/Github/DD2434/Assignment 2/Tree Graphical Model/'
    #qqq: Need to change this to this years verison!!
    filename = directory + "data/q2_2/q2_2_large_tree.pkl"  # "data/q2_2/q2_2_medium_tree.pkl", "data/q2_2/q2_2_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    #t.print()
    #t.print_topology()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files
    for sample_idx in range(5):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
