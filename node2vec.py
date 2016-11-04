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


def get_alias_edge(src, dst, graph, p, q):
    """
    Get the alias edge setup lists for a given edge.
    """

    unnormalized_probs = []
    for dst_nbr in sorted(graph.neighbors(dst)):
        if dst_nbr == src:
            unnormalized_probs.append(graph[dst][dst_nbr]['weight'] / p)
        elif graph.has_edge(dst_nbr, src):
            unnormalized_probs.append(graph[dst][dst_nbr]['weight'])
        else:
            unnormalized_probs.append(graph[dst][dst_nbr]['weight'] / q)
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs)


def preprocess_transition_probs(sc: SparkContext, graph: nx.Graph, p, q, is_directed):
    """
    Preprocessing of transition probabilities for guiding the random walks.
    """

    alias_nodes = {}
    for node in graph.nodes():
        unnormalized_probs = [
            graph[node][nbr]['weight'] for nbr in sorted(graph.neighbors(node))]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        alias_nodes[node] = alias_setup(normalized_probs)

    b_graph = sc.broadcast(graph)

    if is_directed:
        alias_edges = sc.parallelize(graph.edges())\
            .map(lambda uv: ((uv[0], uv[1]), get_alias_edge(uv[0], uv[1], b_graph.value, p, q)))\
            .collectAsMap()
    else:
        alias_edges = sc.parallelize(graph.edges())\
            .flatMap(lambda uv: [
                ((uv[0], uv[1]), get_alias_edge(uv[0], uv[1], graph, p, q)),
                ((uv[1], uv[0]), get_alias_edge(uv[1], uv[0], graph, p, q))
            ]).collectAsMap()

    return alias_nodes, alias_edges


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
    spark_conf = SparkConf().setAppName('node2vec wiki').setMaster(config.master)
    sc = SparkContext.getOrCreate(spark_conf)

    graph = read_graph(config.input, config.weighted, config.directed)

    alias_nodes, alias_edges = preprocess_transition_probs(
        sc, graph, config.p, config.q, config.directed)

    walks = simulate_walks(
        sc, graph, alias_nodes, alias_edges, config.num_walks, config.walk_length)

    learn_embeddings(
        walks, config.dimensions, config.window_size, config.iterations,
        config.workers, config.output)
