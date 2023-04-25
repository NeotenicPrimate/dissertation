import numpy as np
import networkx as nx
from networkx.algorithms import community
from itertools import combinations
import math
import polars as pl

###########################################################################################################################################################
###########################################################################################################################################################

def delta_f(G, f, undirected=True):

    nodes = np.array(G.nodes)
    delta_mat = np.zeros((nodes.size, nodes.size))

    for (u, v) in combinations(nodes, 2):

        G_without = G.copy()
        G_with = G.copy()

        if not G_with.has_edge(u, v): G_with.add_edge(u, v)
        if G_without.has_edge(u, v): G_without.remove_edge(u, v)

        stat_delta = f(G_with) - f(G_without)

        idx_u = np.where(nodes == u)[0]
        idx_v = np.where(nodes == v)[0]

        delta_mat[idx_u, idx_v] = stat_delta
        delta_mat[idx_v, idx_u] = stat_delta

        # if standardize:
        #     delta_mat = (delta_mat - delta_mat.mean()) / delta_mat.std()

        if undirected:
            delta_mat[np.triu_indices_from(delta_mat)] = 0

    return delta_mat

###########################################################################################################################################################
###########################################################################################################################################################

def edges(G):
    return len(G.edges)

def triangles(G):
    return sum(nx.triangles(G).values()) / 3

def betweenness(G):
    return np.mean(np.fromiter(nx.betweenness_centrality(G).values(), float))

def closeness(G):
    return np.mean(np.fromiter(nx.closeness_centrality(G).values(), float))

def eigenvector(G):
    return np.mean(np.fromiter(nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-3).values(), float))

def centralization(G, centrality=nx.degree_centrality):
    G_centrality = np.fromiter(centrality(G).values(), float)
    G_diff = np.sum(G_centrality.max() - G_centrality)

    G_star = nx.star_graph(len(G)-1)
    G_star_centrality = np.fromiter(centrality(G_star).values(), float)
    G_star_diff = np.sum(G_star_centrality.max() - G_star_centrality)

    return G_diff / G_star_diff

def gini(G):
    x = list(dict(G.degree()).values())
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    gini = 0.5 * rmad
    return gini

def clustering(G):
    return nx.average_clustering(G)

def transitivity(G):
    return nx.transitivity(G)

def cliques(G):
    return len(list(nx.enumerate_all_cliques(G)))

def components(G):
    return nx.number_connected_components(G)

def louvain(G):
    return len(community.louvain_communities(G))

def star(G, k):
    return sum(math.comb(G.degree(i), k) for i in G.nodes)

def geodesic(G):
    comp = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(comp)
    return nx.average_shortest_path_length(G_sub)

###########################################################################################################################################################
###########################################################################################################################################################

def date(G, df, t):

    # if two nodes are within t years of each other 1, otherwise 0
    
    nodes = np.array(G.nodes)
    delta_mat = np.zeros((nodes.size, nodes.size))

    for (u, v) in combinations(nodes, 2):

        year_u, = df.filter(pl.col('Doi').eq(pl.lit(u))).select(pl.col('Date').dt.year()).row(0)
        year_v, = df.filter(pl.col('Doi').eq(pl.lit(v))).select(pl.col('Date').dt.year()).row(0)

        if abs(year_u - year_v) <= t:
            delta = 1
        else:
            delta = 0

        idx_u = np.where(nodes == u)[0]
        idx_v = np.where(nodes == v)[0]

        delta_mat[idx_u, idx_v] = delta
        delta_mat[idx_v, idx_u] = delta
        
        delta_mat[np.triu_indices_from(delta_mat)] = 0

    return delta_mat


###########################################################################################################################################################
###########################################################################################################################################################

def p_value_stars(p_value):
    match p_value:
        case _ if p_value <= 0.001:
            stars = '***'
        case _ if p_value <= 0.01:
            stars =  '**'
        case _ if p_value <= 0.05:
            stars =  '*'  
        case _:
            stars = ' '
    return stars

###########################################################################################################################################################
###########################################################################################################################################################

delta_edges = lambda G: delta_f(G, edges)
delta_triangles = lambda G: delta_f(G, triangles)
delta_betweenness = lambda G: delta_f(G, betweenness)
delta_closeness = lambda G: delta_f(G, closeness)
delta_eigenvector = lambda G: delta_f(G, eigenvector)
delta_centralization = lambda G: delta_f(G, centralization)
delta_gini = lambda G: delta_f(G, gini)
delta_clustering = lambda G: delta_f(G, clustering)
delta_transitivity = lambda G: delta_f(G, transitivity)
delta_cliques = lambda G: delta_f(G, cliques)
delta_components = lambda G: delta_f(G, components)
delta_louvain = lambda G: delta_f(G, louvain)
delta_star = lambda G, k: delta_f(G, lambda G: star(G, k))
delta_geodesic = lambda G: delta_f(G, geodesic)




