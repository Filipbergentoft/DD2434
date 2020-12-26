import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dendropy
from Tree import TreeMixture
from Tree import Tree
from Kruskal_v2 import maximum_spanning_tree
from help_functions import q_joint, q_marginal, I_element, mutual_information, \
    create_ordered_pairs, create_ordered_nodes, create_tree_attributes,\
         create_graph, sample_likelihood, create_tree_attributes1


def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)

def em_algorithm(seed_val, samples, num_clusters, max_num_iter, tm=None):

    num_samples = samples.shape[0] 
    num_nodes = samples.shape[1]

    loglikelihood = []
    if tm is None:
        tm = TreeMixture(num_clusters=num_clusters, num_nodes=num_nodes)
        tm.simulate_pi(None)
        tm.simulate_trees(None)
        #samples = tm.samples

    for iter_ in range(max_num_iter):
        # 1. Compute responsibilities for all trees
        sample_likelihoods = np.array([[sample_likelihood(tm.clusters[ii], samples[jj,:]\
            , tm.pi[ii]) for ii in range(num_clusters)] for jj in range(num_samples)])
        sum_over_trees_likelihoods = np.reshape(np.sum(sample_likelihoods, axis = 1),(num_samples,1))
        Responsibilities = np.divide(sample_likelihoods,sum_over_trees_likelihoods)
        # Computing loglikelihood
        ll = np.sum(np.log(np.sum(sample_likelihoods,axis=1)),axis=None)
        loglikelihood.append(ll)

        tm.loglikelihood.append(ll)
        # 2. Updating pi for all trees
        tm.pi = np.sum(Responsibilities,axis=0)/num_samples
        vertices = list(range(num_nodes))
        # 3. Updating each tree
        for i in range(num_clusters):
            tree = tm.clusters[i]
            responsibilities = Responsibilities[:,i]
            # Creating the symmetric mutual information matrix
            mutual_information_matrix = np.asarray([[mutual_information(responsibilities, samples, s_idx, t_idx) \
                for s_idx in vertices] for t_idx in vertices])
            # Computing the graph
            graph = create_graph(num_nodes, responsibilities, samples, mutual_information_matrix, vertices)
            # Finding the maximum spanning tree
            MST = maximum_spanning_tree(graph)
            # Choosing the root as 0
            root_name = 0
            # Finding the order of nodes in the tree
            ordered_nodes, I_sum_tree = create_ordered_nodes(MST, root_name)
            # Getting attributes for tree to enable update
            topology_array, theta_array = create_tree_attributes1(ordered_nodes,
             root_name, samples, responsibilities, num_nodes)
            # Updating the tree
            tree.load_tree_from_direct_arrays(topology_array, theta_array)

    # -------------------------------------------
    topology_list = []
    theta_list = []
    for i in range(num_clusters):
        topology_list.append(tm.clusters[i].get_topology_array())
        theta_list.append(tm.clusters[i].get_theta_array())

    loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)

    return loglikelihood, topology_list, theta_list, tm

def sieving(n_first_mixtures, n_second_mixtures, n_first_iterations,n_second_iterations, samples, num_clusters):

    mixtures = []
    mixtures_loglikelihoods_verification = []

    for i in range(n_first_mixtures):
        
        # Training the model for 10 iterations
        tm = em_algorithm(seed_val=None, samples=samples,
         num_clusters=num_clusters, max_num_iter=n_first_iterations, tm=None)[-1]
        mixtures.append(tm)
        # Computing the loglikelihood on the verification samples
        loglikelihood_verification = mixture_likelihood(tm, samples)
        mixtures_loglikelihoods_verification.append(loglikelihood_verification)
    
    mixtures_loglikelihoods_verification = np.asarray(mixtures_loglikelihoods_verification)
    idx_best_mixtures = np.argsort(mixtures_loglikelihoods_verification)[0:10]

    second_mixtures = []
    second_mixtures_loglikelihoods = []

    for i in range(n_second_mixtures):
        print('Optimizing second mixture number: ', i)
        tm_idx = idx_best_mixtures[i]
        tm = mixtures[tm_idx]
        tm = em_algorithm(seed_val=None, samples=samples,
         num_clusters=num_clusters, max_num_iter=n_second_iterations, tm=tm)[-1]
        second_mixtures.append(tm)
        loglikelihood_verification = mixture_likelihood(tm, samples)
        second_mixtures_loglikelihoods.append(loglikelihood_verification)

    best_idx = np.argmax(second_mixtures_loglikelihoods)
    best_tm = second_mixtures[best_idx]
    print(second_mixtures_loglikelihoods)

    topology_list = []
    theta_list = []
    for i in range(num_clusters):
        topology_list.append(best_tm.clusters[i].get_topology_array())
        theta_list.append(best_tm.clusters[i].get_theta_array())

    #loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)

    return best_tm



def mixture_likelihood(mixture, samples):
    num_clusters = mixture.pi.shape[0]
    num_samples = samples.shape[0]
    sample_likelihoods = np.array([[sample_likelihood(mixture.clusters[ii], samples[jj,:]\
        , mixture.pi[ii]) for ii in range(num_clusters)] for jj in range(num_samples)])

    # Computing loglikelihood
    ll = np.sum(np.log(np.sum(sample_likelihoods,axis=1)),axis=None)

    return ll

def RF_comparison(inferred_tm, real_tm):
    tns = dendropy.TaxonNamespace()

    dendr_inferred_trees = []
    dendr_real_trees = []
    for inferred_tree in inferred_tm.clusters:
        inferred_tree = dendropy.Tree.get(data=inferred_tree.newick, schema="newick", taxon_namespace=tns)
        dendr_inferred_trees.append(inferred_tree)

    for real_tree in real_tm.clusters:
        real_tree = dendropy.Tree.get(data=real_tree.newick, schema="newick", taxon_namespace=tns)
        dendr_real_trees.append(real_tree)

    #x-dir: inferred trees, y-dir: real trees
    RF_matrix = [[dendropy.calculate.treecompare.symmetric_difference(inferred_tree, real_tree)\
         for inferred_tree in dendr_inferred_trees] for real_tree in dendr_real_trees]
        
    return np.asarray(RF_matrix)

def main():

    num_clusters = 3
    new_tm = TreeMixture(num_clusters=num_clusters, num_nodes=3)
    new_tm.simulate_pi(None)
    new_tm.simulate_trees(None)
    new_tm.sample_mixtures(100)
    new_samples = new_tm.samples
    #samples = tm.samples
    seed_val = None
    directory = '/Users/filipbergentoft/Desktop/Github/DD2434/Assignment 2/2_4/'
    sample_filename = directory + "data/q2_4/q2_4_tree_mixture.pkl_samples.txt"
    output_filename = directory + "data/q2_4/q2_4_own_results"
    real_values_filename = directory + "data/q2_4/q2_4_tree_mixture.pkl"


    samples = np.loadtxt(sample_filename, delimiter="\t", dtype=np.int32)
    np.random.shuffle(samples)

    best_tm = sieving(n_first_mixtures=50, n_second_mixtures=10,
     n_first_iterations=10, n_second_iterations=100, samples=samples, num_clusters=num_clusters)


    real_tm = TreeMixture(num_clusters = 3, num_nodes = 5)
    real_tm.load_mixture(real_values_filename)
    
    print('best tree',mixture_likelihood(best_tm, samples))
    print('best tree',best_tm.pi)
    print('real tree',mixture_likelihood(real_tm, samples))
    print('real tree',real_tm.pi)


    print(RF_comparison(best_tm, real_tm))
    for tree in new_tm.clusters:
        print('Real tree topology')
        print(tree.get_topology_array())

    for tree in best_tm.clusters:
        print('Inferred tree topology')
        print(tree.get_topology_array())


    sns.set_style('darkgrid')
    """
    plt.subplot(121)
    plt.plot(np.exp(best_tm.loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    """
    plt.plot(best_tm.loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
