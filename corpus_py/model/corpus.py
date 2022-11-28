import polars as pl
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

class Corpus():
    citation_df = pl.DataFrame()
    citation_graph = nx.DiGraph()

    def __init__(self, dir_name):

        cols = [
            ("DI", "Doi"),
            ("AU", "Authors"), 
            ("PY", "Year"),
            ("PD", "Month"),
            ("TI", "Title"),
            ("AB", "Abstract"),
            ("SO", "Journal"),
            ("DE", "AuthorKeywords"),
            ("ID", "WosKeywords"),
            ("WC", "Category"),
            ("SC", "Areas"),
            ("CR", "References"),
        ]

        stopwords = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", 
        "during", "out", "very", "from", "re", "edu", "use", "published", "a", "has", "elsevier", "may", "paper", "studi", "discuss",
        "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", 
        "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", 
        "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", 
        "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", 
        "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", 
        "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", 
        "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", 
        "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]

        old_cols = [old for (old, _) in cols]
        new_cols = [new for (_, new) in cols]
        dtypes = [pl.Utf8 for _ in range(len(cols))]

        df = pl.read_csv(
            file=f"./data/{dir_name}/savedrecs*.txt",
            has_header=True, 
            columns=old_cols,
            new_columns=new_cols,
            sep="\t",
            dtypes=dtypes,
            null_values=None,
            ignore_errors=False,
            parse_dates=False,
            n_threads=None,
            infer_schema_length=100,
        )

        df = df.unique()
        df = df.with_columns([

            pl.col("Doi").str.replace_all(":", "").str.strip(),

            pl.when(pl.col("Month").str.starts_with("JAN")).then(pl.lit(1))
                .when(pl.col("Month").str.starts_with("FAL")).then(pl.lit(1))
                .when(pl.col("Month").str.starts_with("FEB")).then(pl.lit(2))
                .when(pl.col("Month").str.starts_with("MAR")).then(pl.lit(3))
                .when(pl.col("Month").str.starts_with("APR")).then(pl.lit(4))
                .when(pl.col("Month").str.starts_with("SPR")).then(pl.lit(4))
                .when(pl.col("Month").str.starts_with("MAY")).then(pl.lit(5))
                .when(pl.col("Month").str.starts_with("JUN")).then(pl.lit(6))
                .when(pl.col("Month").str.starts_with("JUL")).then(pl.lit(7))
                .when(pl.col("Month").str.starts_with("SUM")).then(pl.lit(7))
                .when(pl.col("Month").str.starts_with("AUG")).then(pl.lit(8))
                .when(pl.col("Month").str.starts_with("SEP")).then(pl.lit(9))
                .when(pl.col("Month").str.starts_with("WIN")).then(pl.lit(10))
                .when(pl.col("Month").str.starts_with("OCT")).then(pl.lit(10))
                .when(pl.col("Month").str.starts_with("NOV")).then(pl.lit(11))
                .when(pl.col("Month").str.starts_with("DEC")).then(pl.lit(12))
                .otherwise(pl.lit(1))
                .cast(pl.UInt32)
                .alias("Month"),

            pl.col("Authors").str.to_lowercase().str.split(";").arr.eval(pl.element().str.strip(None)),
            pl.col("WosKeywords").str.to_lowercase().str.split(";").arr.eval(pl.element().str.strip(None)),
            pl.col("AuthorKeywords").str.to_lowercase().str.split(";").arr.eval(pl.element().str.strip(None)),
            pl.col("Areas").str.to_lowercase().str.split(";").arr.eval(pl.element().str.strip(None)),
            pl.col("Category").str.to_lowercase().str.split(";").arr.eval(pl.element().str.strip(None)),
            pl.col("Journal").str.to_lowercase().str.strip(None),
            pl.col("Title").str.to_lowercase().str.strip(None),
            pl.col("References").str.extract_all(r"10.\d{4,9}/[-._()/:a-zA-Z0-9]+").arr.eval(pl.element().str.strip(None)),
        ])

        df = df.with_columns([

            pl.date(pl.col("Year"), pl.col("Month"), pl.lit(1)).alias("Date"),

            pl.concat_str([
                pl.col("Title"), 
                pl.col("Abstract").fill_null(pl.lit("")), 
                pl.col("Authors").arr.join(" ").fill_null(pl.lit("")),
            ], " ")
            .str.replace_all(r"[^a-zA-Z\s]", "")
            .alias("Text")

        ])

        df = df.drop(["Year", "Month"])

        df = df.drop_nulls(["Doi","Authors","Date","Title","References","Text"]) # "Abstract"

        lemmatizer = WordNetLemmatizer()

        clean_text = (
            df.select([
                pl.col("Doi"),
                pl.col("Text").str.split(" "),
            ])
            .explode("Text")
            .with_column(pl.col("Text").apply(lambda s: lemmatizer.lemmatize(s)))
            .filter(pl.col("Text").str.lengths() >= 3)
            .filter(~pl.col("Text").is_in(stopwords))
            .groupby("Doi")
            .agg(pl.col("Text"))
            .with_column(pl.col("Text").arr.join(" "))
        )

        df = (
            df
            .select([pl.all().exclude(["Text"])])
            .join(
                clean_text,
                on="Doi",
                how="inner",
            )
        )

        pruned_df = (
            df
                .select([pl.col("Doi"), pl.col("Date"), pl.col("References")])
                .explode(["References"])
                .filter(pl.col("References").is_in(pl.col("Doi")))
                .join(
                    df.select([pl.col("Doi"), pl.col("Date")]),
                    left_on=pl.col("References"),
                    right_on=pl.col("Doi"),
                    how="left",
                )
                .filter(pl.col("Date") >= pl.col("Date_right"))
                .groupby([pl.col("Doi")])
                .agg([pl.col("References").list()])
        )

        df = (
            df
                .select([pl.all().exclude(["References"])])
                .join(
                    pruned_df,
                    on="Doi",
                    how="inner",
                )
        )

        adjlist = df.select(
            pl.concat_list([
                pl.col("Doi"), 
                pl.col("References")
            ]).arr.join(" ")
        ).get_column("Doi").to_list()

        G = nx.parse_adjlist(adjlist, nodetype=str, create_using=nx.DiGraph)
        G.remove_nodes_from(set(G.nodes()).difference(set(df.get_column("Doi").to_list())))

        large_component = sorted(nx.weakly_connected_components(G), key=len, reverse=True)[0]
        G = nx.DiGraph(nx.subgraph(G, large_component))

        G.remove_nodes_from(nx.isolates(G))
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_edges_from([(c[-1], c[0]) for c in nx.simple_cycles(G)])

        centrality_map = nx.in_degree_centrality(G)
        df_centrality = pl.DataFrame([
            pl.Series("Doi", list(centrality_map.keys()), dtype=pl.Utf8),
            pl.Series("InCentrality", list(centrality_map.values()), dtype=pl.Float32),
        ])
        df = df.join(df_centrality, on="Doi")

        self.citation_df = df
        self.citation_graph = G

        print(f'''
            Nodes: {len(self.citation_graph.nodes())}
            Edges: {len(self.citation_graph.edges())}
            Df: {self.citation_df.shape}
        ''')

    def most_referenced(self, n: int):
        return self.citation_df.sort("InCentrality", reverse=True).head(n)