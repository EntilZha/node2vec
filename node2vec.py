import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec
from pyspark import SparkConf, SparkContext


class N2VConfig:
    def __init__(self):
        self.input = 'graph/karate.edgelist'
        self.output = 'emb/karate.emb'
        self.dimensions = 128
        self.walk_length = 80
        self.num_walks = 10
        self.window_size = 10
        self.iterations = 1
        self.workers = 8
        self.p = 1
        self.q = 1
        self.weighted = False
        self.directed = False
        self.master = 'local[*]'


def simulate_walks(sc: SparkContext, graph, alias_nodes, alias_edges, num_walks, walk_length):
    nodes = list(graph.nodes()) * num_walks
    b_graph = sc.broadcast(graph)
    b_alias_nodes = sc.broadcast(alias_nodes)
    b_alias_edges = sc.broadcast(alias_edges)
    walks = sc.parallelize(nodes)\
        .map(lambda node: node2vec_walk(b_graph, b_alias_nodes, b_alias_edges, walk_length, node))\
        .collect()
    random.shuffle(walks)
    return walks


def node2vec_walk(b_graph, b_alias_nodes, b_alias_edges, walk_length, start_node):
    graph = b_graph.value
    alias_nodes = b_alias_nodes.value
    alias_edges = b_alias_edges.value

    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = sorted(graph.neighbors(cur))
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
            else:
                prev = walk[-2]
                next_node = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                                alias_edges[(prev, cur)][1])]
                walk.append(next_node)
        else:
            break

    return walk


class Graph:
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.alias_nodes = None
        self.alias_edges = None

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def read_graph(input_path, weighted, directed):
    """
    Reads the input network in networkx.
    """
    if weighted:
        G = nx.read_edgelist(input_path, nodetype=int, data=(('weight', float),),
                             create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input_path, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, dimensions, window_size, iterations, workers, output_path):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1,
                     workers=workers, iter=iterations)
    model.save_word2vec_format(output_path)


def run_n2v(config: N2VConfig):
    spark_conf = SparkConf().setAppName('node2vec').setMaster(config.master)
    sc = SparkContext.getOrCreate(spark_conf)

    nx_G = read_graph(config.input, config.weighted, config.directed)
    G = Graph(nx_G, config.directed, config.p, config.q)

    G.preprocess_transition_probs()

    walks = simulate_walks(
        sc, nx_G, G.alias_nodes, G.alias_edges, config.num_walks, config.walk_length)

    learn_embeddings(
        walks, config.dimensions, config.window_size, config.iterations,
        config.workers, config.output)
