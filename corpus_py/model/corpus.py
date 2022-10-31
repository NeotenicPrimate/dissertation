import polars as pl
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from utils import clean_text



class Corpus():
    citation_df = pl.DataFrame()
    citation_graph = nx.DiGraph()

    def __init__(self, parquet_path):

        # DATAFRAME

        df = pl.read_parquet(parquet_path)
        df = df.with_column(pl.col('Text').map(clean_text))
        df = df.with_column(pl.col('Doi').str.replace_all(':', ''))
        
        # ADJLIST
        
        adjlist = df.select([pl.concat_list([pl.col("Doi"), pl.col("References")]).alias("AdjList")]).get_column("AdjList").to_list()
        lines = [" ".join(l) if l else "" for l in adjlist]
        g = nx.parse_adjlist(lines, nodetype=str, create_using=nx.DiGraph)
        
        # GRAPH
        
        large_component = sorted(nx.weakly_connected_components(g), key=len, reverse=True)[0]
        g = nx.subgraph(g, large_component)
        g = nx.DiGraph(g)
        
        g.remove_nodes_from(list(nx.isolates(g)))
        g.remove_edges_from(nx.selfloop_edges(g))
        for cycle in nx.simple_cycles(g):
            try:
                g.remove_edge(cycle[-1], cycle[0])
            except:
                continue

        # COMMUNITY

        communities = list(community.louvain_communities(g, weight=None))
        sorted_communities = sorted(communities, key=len, reverse=True)
        community_map = {}
        for (i, com) in enumerate(sorted_communities):
            for node in com:
                community_map[node] = i
        df_community = pl.DataFrame([
            pl.Series("Doi", list(community_map.keys()), dtype=pl.Utf8),
            pl.Series("Community", list(community_map.values()), dtype=pl.Int32),
        ])
        df = df.join(df_community, on="Doi")

        # CENTRALITY

        centrality_map = nx.in_degree_centrality(g)
        df_centrality = pl.DataFrame([
            pl.Series("Doi", list(centrality_map.keys()), dtype=pl.Utf8),
            pl.Series("InCentrality", list(centrality_map.values()), dtype=pl.Float32),
        ])

        df = df.join(df_centrality, on="Doi")

        # TF-IDF

        

        self.citation_graph = g
        print(f''' Nodes: {len(g.nodes())} \n Edges: {len(g.edges())} ''')

        self.citation_df = df
        print(f''' Shape: {df.shape} ''')

    def most_referenced(self, n):
        return self.citation_df.sort("InCentrality", reverse=True).head(n)
        

    def get_descendants(self, root):
        
        # FLIP EDGES !!!
        
        descendants = nx.descendants(self.citation_graph, root)
        descendants.add(root)
        G_descendants = self.citation_graph.subgraph(descendants)
        return G_descendants

    def stem(word):
        stemmer = PorterStemmer()
        return stemmer.stem(word)

    def draw_tree(self, G):
        pos = pos = nx.nx_pydot.graphviz_layout(G, prog="dot") # dot, twopi, fdp, sfdp, circo
        plt.figure(figsize=(20,20))
        nx.draw(G, pos, with_labels=True)
        plt.show()

    # def edge_dist(self, direction='in'):

    #     g = self.citation_graph

    #     if direction == 'in':
    #         degree = g.in_degree()
    #     elif direction == 'out':
    #         degree = g.out_degree()
    #     else:
    #         raise Exception('Expected: `in` | `out`')

    #     degree = list(dict(degree).values())
    #     d = {v: degree.count(v) for v in set(degree)}
    #     k, pk = list(d.keys()), list(d.values())

    #     df = pl.DataFrame()
    #     df = df.with_column(pl.Series('k', k))
    #     df = df.with_column(pl.Series('pk', pk))
    #     df = df.with_column(pl.col('k') / pl.col('pk').sum())

    #     plt.plot(df['k'], df['pk'], 'o', label='obs')

    #     plt.xlabel('$\it{k}$')
    #     plt.ylabel('$\it{p(k)}$')
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     plt.show()
