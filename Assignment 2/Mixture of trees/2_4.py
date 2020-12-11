""" This file is created as a template for question 2.4 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)! Previously, other students used NetworkX package to work with trees and graphs, keep in mind.

    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you a single tree mixture (q2_4_tree_mixture).
    The mixture has 3 clusters, 5 nodes and 100 samples.
    We want you to run your EM algorithm and compare the real and inferred results
    in terms of Robinson-Foulds metric and the likelihoods.
    """

import numpy as np
import matplotlib.pyplot as plt
from Tree import TreeMixture
from Kruskal_v2 import maximum_spanning_tree

def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)


def sample_likelihood(tree, sample, pi_tree):
    responsibility = np.nan
    tree_topology = tree.get_topology_array().astype(int)
    all_theta = np.array(tree.get_theta_array())
    for node_idx in range(tree.num_nodes):
        # If root do:
        if node_idx == 0:
            theta = all_theta[node_idx]
            node_value = sample[node_idx]
            responsibility = pi_tree*theta[node_value]
        # If not root do:
        else:
            theta = np.stack(all_theta[node_idx])
            parent_idx = tree_topology[node_idx]
            parent_value = sample[parent_idx]
            node_value = sample[node_idx]
            responsibility *= theta[node_value,parent_value]

    return responsibility
def q_joint(responsibilities, s_vec, t_vec, a, b):

    numerator = sum([r for (r,s,t) in zip(responsibilities, s_vec, t_vec) if s == a and t == b])
    denominator = np.sum(responsibilities)

    return numerator/denominator

def q_marginal(responsibilities, s_vec, a):
    numerator = np.sum(responsibilities[s_vec == a])
    denominator = np.sum(responsibilities)

    return numerator/denominator

def I(responsibilities, s_vec, t_vec, a, b):
    factor1 = q_joint(responsibilities, s_vec, t_vec, a, b)
    factor2 = np.log(q_joint(responsibilities, s_vec, t_vec, a, b) \
        /q_marginal(responsibilities, s_vec, a)/q_marginal(responsibilities, t_vec, b))
    if factor1 == 0:
        return 0
    else:
        return factor1*factor2

def mutual_information(responsibilities, samples, s_idx, t_idx):
    # Going from node s to node t
    s_vec = samples[:,s_idx]
    t_vec = samples[:,t_idx]
    I_matrix = [[I(responsibilities, s_vec, t_vec, a, b) for a in [0,1]] for b in [0,1]]

    return np.sum(I_matrix, axis=None)
    
    
def em_algorithm(seed_val, samples, num_clusters, max_num_iter=100):
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    This is a suggested template. Feel free to code however you want.
    """

    # Set the seed
    np.random.seed(seed_val)

    num_samples = samples.shape[0] # Should be 100
    num_nodes = samples.shape[1] # Should be 5
    print("Running EM algorithm...")

    loglikelihood = []
    #1. for each n,k compute responsibilities - (a) get pi, (b) get theta

    for iter_ in range(max_num_iter):
        loglikelihood.append(np.log((1 + iter_) / max_num_iter))


    tm = TreeMixture(num_clusters=num_clusters, num_nodes=num_nodes)
    tm.simulate_pi(seed_val=seed_val)
    tm.simulate_trees(seed_val=seed_val)
    tm.sample_mixtures(num_samples=num_samples, seed_val=seed_val)
    
    samples = tm.samples.astype(int)
    # 1. Compute responsibilities for every mixture and every sample
    sample_likelihoods = np.array([[sample_likelihood(tm.clusters[i], samples[j,:], tm.pi[i]) for i in range(num_clusters)] for j in range(num_samples)])
    sum_over_trees_likelihoods = np.reshape(np.sum(sample_likelihoods, axis = 1),(100,1))
    Responsibilities = np.divide(sample_likelihoods,sum_over_trees_likelihoods)

    # 2. Updating pi
    tm.pi = np.sum(Responsibilities,axis=0)/num_samples
    
    # 3. Updating each tree
    for i in range(num_clusters):
        # 3.1 Computing a directed graph
        tree = tm.clusters[i]
        responsibilities = Responsibilities[:,i]
        vertices = [x for x in range(num_nodes)]
        edges = set()
        [[edges.add((s_idx,t_idx,mutual_information(responsibilities, samples, s_idx, t_idx))) for s_idx in vertices] for t_idx in vertices]
        graph = {
        'vertices': vertices,
        'edges': edges
        }
        # 3.2 Finding the maximum spanning tree
        MST = maximum_spanning_tree(graph)

        # 3.3 Updating the tree
        pairs = []
        for pair in MST:
            pairs.append(pair[0:2])
        pairs = np.asarray(pairs)
        root_idx = np.int(np.sum(vertices) - np.sum(pairs[:,1]))
        
        print(MST)

    topology_list = []
    theta_list = []
    for i in range(num_clusters):
        topology_list.append(tm.clusters[i].get_topology_array())
        theta_list.append(tm.clusters[i].get_theta_array())

    loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)
    # End: Example Code Segment

    ###

    return loglikelihood, topology_list, theta_list


def main():
    print("Hello World!")
    print("This file demonstrates the flow of function templates of question 2.4.")

    seed_val = 123
    directory = '/Users/filipbergentoft/Desktop/Github/DD2434/Assignment 2/Mixture of trees/'
    sample_filename = directory + "data/q2_4/q2_4_tree_mixture.pkl_samples.txt"
    output_filename = "q2_4_results.txt"
    real_values_filename = directory + "data/q2_4/q2_4_tree_mixture.pkl"
    num_clusters = 3

    print("\n1. Load samples from txt file.\n")

    samples = np.loadtxt(sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    #print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")

    loglikelihood, topology_array, theta_array = em_algorithm(seed_val, samples, num_clusters=num_clusters)

    print("\n3. Save, print and plot the results.\n")

    save_results(loglikelihood, topology_array, theta_array, output_filename)

    for i in range(num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    if real_values_filename != "":
        print("\n4. Retrieve real results and compare.\n")
        print("\tComparing the results with real values...")

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        # TODO: Do RF Comparison

        print("\t4.2. Make the likelihood comparison.\n")
        # TODO: Do Likelihood Comparison


if __name__ == "__main__":
    main()
