from os import DirEntry
import numpy as np
from Tree import Tree
from Tree import Node
import os


def calculate_likelihood(tree_topology, theta, beta):


    # tree_topology: index corresponds to node,
    # element corresponds to parent of that node, nan := root node
    # K defines the number of nodes in the tree
    K = np.asarray(theta).shape[1]
    number_of_nodes = np.asarray(theta).shape[0]
    node_ids = range(number_of_nodes-1,-1,-1)
    print(tree_topology)
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
    directory = '/Users/filipbergentoft/Desktop/Github/DD2434/Assignment 2/2_2/'

    filename = directory + "data/q2_2/q2_2_small_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    #t.print()
    #t.print_topology()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")

    for sample_idx in range(5):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(),
         t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
