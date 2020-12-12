import numpy as np

def q_joint(responsibilities, s_vec, t_vec, a, b):
    numerator = sum([r for (r,s,t) in zip(responsibilities, s_vec, t_vec) if s == a and t == b])
    denominator = np.sum(responsibilities)

    return numerator/denominator

def q_marginal(responsibilities, s_vec, a):
    numerator = np.sum(responsibilities[s_vec == a])
    denominator = np.sum(responsibilities)

    return numerator/denominator

def I_element(responsibilities, s_vec, t_vec, a, b):
    factor1 = q_joint(responsibilities, s_vec, t_vec, a, b)
    if factor1 <= 0:
        return 0
    q_s = q_marginal(responsibilities, s_vec, a)
    q_t = q_marginal(responsibilities, t_vec, b)

    factor2 = np.log(factor1/(q_s*q_t))
    if factor1 <= 0:
        return 0
    else:
        return factor1*factor2

def mutual_information(responsibilities, samples, s_idx, t_idx):
    # Going from node s to node t
    s_vec = samples[:,s_idx]
    t_vec = samples[:,t_idx]
    I_matrix = [[I_element(responsibilities, s_vec, t_vec, a, b) for a in [0,1]] for b in [0,1]]

    return np.sum(I_matrix, axis=None)

def create_graph(num_nodes, responsibilities, samples, mutual_information_matrix, vertices):
    #vertices = [x for x in range(num_nodes)]
    """Maybe dont add edges from same node to same node"""
    edges = set()
    [[edges.add((s_idx,t_idx,mutual_information_matrix[s_idx, t_idx])) \
        for s_idx in vertices] for t_idx in vertices]
    graph = {
    'vertices': vertices,
    'edges': edges
    }
    return graph

    
def create_ordered_pairs(ordered_nodes,parent_node_name, pairs):

    mask = np.where(np.sum(pairs == parent_node_name,axis=1) == 1)
    parent_children_pairs = pairs[mask,:]
    # Removing the connections from contention
    pairs = np.delete(pairs, mask,axis=0)
    children_names = parent_children_pairs[parent_children_pairs != parent_node_name]


    # Add children to ordered pairs
    if len(children_names) is 0:
        pass
    else:
        for child_name in children_names:
            ordered_nodes.append([parent_node_name,child_name])

    # Go recursive
    if len(children_names) is 0:
        pass
    else:
        for child_name in children_names:
            create_ordered_pairs(ordered_nodes, child_name, pairs)

def create_ordered_nodes(MST, root_name):
    # Finding the node pairs
    pairs = []
    I_sum = 0
    for pair in MST:
        pairs.append(pair[0:2])
        I_sum += pair[2]
    pairs = np.asarray(pairs)
    #Adding root to ordered list of nodes
    ordered_nodes = []
    ordered_nodes.append([np.nan,root_name])

    # creating ordered pairs of nodes
    create_ordered_pairs(ordered_nodes,root_name, pairs)
    return np.asarray(ordered_nodes), I_sum

def create_tree_attributes(ordered_nodes, root_name, samples, responsibilities, num_nodes):
        # Create theta matrices
    all_theta = []
    topology_array = []

    # Handling the root as a special case
    s_vec = samples[:,root_name]
    theta_root = np.asarray([q_marginal(responsibilities, s_vec, 0), q_marginal(responsibilities, s_vec, 1)])
    all_theta.append(theta_root)
    topology_array.append(np.nan)
    for j in range(1,num_nodes):
        # s is the parent
        # 
        s = ordered_nodes[j,0].astype(int)
        s_vec = samples[:,s]
        t = ordered_nodes[j,1].astype(int)
        t_vec = samples[:,t]

        theta = np.asarray([[q_joint(responsibilities, s_vec, t_vec, a, b)/q_marginal(responsibilities, s_vec, a) for b in [0,1]] for a in [0,1]])
        all_theta.append(theta)

        parent_idx = np.where(ordered_nodes[:,1]==s)[0][0]
        topology_array.append(parent_idx)
    topology_array = np.asarray(topology_array)#.astype(int)
    topology_array[0] = float('nan')

    return topology_array, all_theta


def sample_likelihood(tree, sample, pi_tree):

    likelihood = np.nan
    tree_topology = tree.get_topology_array().astype(int)
    all_theta = np.array(tree.get_theta_array())
    for node_idx in range(tree.num_nodes):
        # If root do:
        if node_idx == 0:
            theta = all_theta[node_idx]
            node_value = sample[node_idx]
            likelihood = pi_tree*theta[node_value]
        # If not root do:
        else:
            theta = np.stack(all_theta[node_idx],axis=1)
            parent_idx = tree_topology[node_idx]
            parent_value = sample[parent_idx]
            node_value = sample[node_idx]
            likelihood *= theta[node_value,parent_value]

    return likelihood

def create_tree_attributes1(ordered_nodes, root_name, samples, responsibilities, num_nodes):
    
    # Initializing the new topology array and setting root parent to nan
    topology_array = np.zeros(5)
    topology_array[0] = float('nan')


    # Initializing the new theta array and setting root theta 
    theta_array = list(range(5))
    root_samples = samples[:,root_name]
    theta_root = np.asarray([q_marginal(responsibilities, root_samples, 0), q_marginal(responsibilities, root_samples, 1)])
    theta_array[0]= theta_root

    # Iterating to set the remaining topology_array and theta_array
    for j in range(1, num_nodes):
        child_idx = ordered_nodes[j,1].astype(int)
        parent_idx = ordered_nodes[j,0].astype(int)
        topology_array[child_idx] = parent_idx

        child_samples = samples[:,child_idx]
        parent_samples = samples[:,parent_idx]
        theta = np.asarray([[q_joint(responsibilities, parent_samples, child_samples, a, b)\
            /q_marginal(responsibilities, parent_samples, a) for b in [0,1]] for a in [0,1]])
        theta_array[child_idx] = theta


    return topology_array, theta_array