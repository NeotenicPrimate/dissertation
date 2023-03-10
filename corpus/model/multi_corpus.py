
import polars as pl
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np

from scipy.optimize import curve_fit
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import os
from collections import defaultdict
from itertools import combinations

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import inspect

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
sns.set(rc = {'figure.figsize':(15,8)})

from constants import DATA_PATH, OUTPUT_PATH, CITATION_DF_PATH
from utils.stopwords import STOPWORDS

################
# Citation DF
################

def citation_df():

    cols = [
        ('DI', 'Doi'),
        ('AU', 'Authors'), 
        ('PY', 'Year'),
        ('PD', 'Month'),
        ('TI', 'Title'),
        ('AB', 'Abstract'),
        ('SO', 'Journal'),
        ('DE', 'AuthorKeywords'),
        ('ID', 'WosKeywords'),
        ('WC', 'Category'),
        ('SC', 'Areas'),
        ('CR', 'References'),
    ]

    # cols_not_null = []

    month_map = {
        'JAN': 1, 
        'FAL': 1, 
        'FEB': 2, 
        'MAR': 3, 
        'APR': 4, 
        'SPR': 4, 
        'MAY': 5, 
        'JUN': 6, 
        'JUL': 7, 
        'SUM': 7, 
        'AUG': 8, 
        'SEP': 9, 
        'WIN': 10, 
        'OCT': 10, 
        'NOV': 11, 
        'DEC': 12
    }

    old_cols, new_cols = zip(*cols)
    dtypes = [pl.Utf8] * len(cols)

    lemmatizer = WordNetLemmatizer()

    if not os.path.exists(CITATION_DF_PATH): os.mkdir(CITATION_DF_PATH)
    for field_name in os.listdir(DATA_PATH):
        if not field_name.startswith('.'):
            field_dfs = []
            for journal_name in os.listdir(os.path.join(DATA_PATH, field_name)):
                if not journal_name.startswith('.'):
                    field_dfs.append(
                        pl.read_csv(
                            file=os.path.join(DATA_PATH, field_name, journal_name, 'savedrecs*.txt'),
                            has_header=True, 
                            columns=old_cols,
                            new_columns=new_cols,
                            sep='\t',
                            dtypes=dtypes,
                            null_values=None,
                            ignore_errors=False,
                            parse_dates=False,
                            n_threads=None,
                        )
                    )
            df = pl.concat(field_dfs)

            df = (
                df
                .with_columns([
                    pl.col('Doi').str.replace_all(':', '').str.strip(None).cast(pl.Utf8),
                    pl.col('Month').map_dict(month_map, default=pl.lit(1)).cast(pl.UInt32),
                    pl.col('Year').cast(pl.UInt32, strict=False),
                    pl.col('Authors').str.to_lowercase().str.split(';').arr.eval(pl.element().str.strip(None)).cast(pl.List(pl.Utf8)),
                    pl.col('WosKeywords').str.to_lowercase().str.split(';').arr.eval(pl.element().str.strip(None)).cast(pl.List(pl.Utf8)),
                    pl.col('Areas').str.to_lowercase().str.split(';').arr.eval(pl.element().str.strip(None)).cast(pl.List(pl.Utf8)),
                    pl.col('Category').str.to_lowercase().str.split(';').arr.eval(pl.element().str.strip(None)).cast(pl.List(pl.Utf8)),
                    pl.col('AuthorKeywords').str.to_lowercase().str.split(';').arr.eval(pl.element().str.strip(None)).cast(pl.List(pl.Utf8)),
                    pl.col('Title').str.to_lowercase().str.strip(None).cast(pl.Utf8),
                    pl.col('Journal').str.to_lowercase().str.strip(None).cast(pl.Utf8),
                    pl.col('Abstract').str.to_lowercase().str.strip(None).cast(pl.Utf8),
                    pl.col('References').str.extract_all(r'10.\d{4,9}/[-._()/:a-zA-Z0-9]+').arr.eval(pl.element().str.strip(None)).cast(pl.List(pl.Utf8)),
                ])
                .with_columns(
                    pl.when(pl.col('Authors').arr.lengths().eq(0)).then(None).otherwise(pl.col('Authors')).keep_name(),
                    pl.when(pl.col('Doi').str.lengths().eq(0)).then(None).otherwise(pl.col('Doi')).keep_name(),
                    pl.when(pl.col('Title').str.lengths().eq(0)).then(None).otherwise(pl.col('Title')).keep_name(),
                    pl.when(pl.col('Journal').str.lengths().eq(0)).then(None).otherwise(pl.col('Journal')).keep_name(),
                    pl.when(pl.col('Abstract').str.lengths().eq(0)).then(None).otherwise(pl.col('Abstract')).keep_name(),
                    pl.when(pl.col('References').is_null()).then([]).otherwise(pl.col('References')).keep_name(),
                    pl.when(pl.col('WosKeywords').is_null()).then([]).otherwise(pl.col('WosKeywords')).keep_name(),
                    pl.when(pl.col('Areas').is_null()).then([]).otherwise(pl.col('Areas')).keep_name(),
                    pl.when(pl.col('Category').is_null()).then([]).otherwise(pl.col('Category')).keep_name(),
                    pl.when(pl.col('AuthorKeywords').is_null()).then([]).otherwise(pl.col('AuthorKeywords')).keep_name(),
                )
                .drop_nulls(list(map(lambda c: c[1], cols)))
                .unique(subset=['Doi', 'Abstract', 'Title'])

                ################
                # Date
                ################

                .with_columns(pl.date(pl.col('Year'), pl.col('Month'), pl.lit(1)).alias('Date')).drop(['Month', 'Year'])
            )

            ################
            # Clean Text
            ################

            clean_text = (
                df
                .select(
                    pl.col('Doi'),
                    pl.concat_str(
                        [
                        pl.col('Title'), 
                        pl.col('Abstract'), 
                        pl.col('AuthorKeywords').arr.join(' '),
                        ], 
                        ' '
                    )
                        .str.replace_all(r'[^a-z\s]', ' ')
                        .str.replace_all(r'\s+', ' ')
                        .alias('Text')
                )
                .with_columns(
                    pl.col('Text')
                        .apply(
                            lambda text: pl.Series(word_tokenize(text)), 
                            return_dtype=pl.List(pl.Utf8)
                        )
                )
                .explode('Text')
                .with_columns(
                    pl.col('Text')
                        .apply(
                            lambda word: lemmatizer.lemmatize(word), 
                            return_dtype=pl.Utf8
                        )
                )
                .filter(~pl.col('Text').is_in(STOPWORDS))
                .filter(pl.col('Text').str.lengths().ge(3))
                .groupby(pl.col('Doi'))
                .agg(pl.col('Text'))
            )

            df = df.join(other=clean_text, on='Doi', how='left').filter(pl.col('Text').arr.lengths().le(300))

            ################
            # Remove references not in dois
            ################

            dois = df.select(pl.col('Doi')).to_numpy().flatten()
            df = (
                df
                .with_columns(
                    pl.col('References')
                        .arr.eval(
                            pl.element().filter(pl.element().is_in(list(dois)))
                        )
                )
                .explode('References')
                .filter((pl.col('Doi').is_in(pl.col('References')) | (pl.col('References').is_in(pl.col('Doi')))))
                .groupby(pl.col('Doi'))
                .agg(pl.col('References'))
                .with_columns(pl.col('References').arr.eval(pl.element().filter(~pl.element().is_null())))
                .join(other=df.select(pl.all().exclude('References')), on='Doi', how='left')
            )

            assert(
                (
                    df
                    .select(pl.col('Doi'), pl.col('References'))
                    .explode(pl.col('References'))
                    .select((pl.col('Doi').is_in(pl.col('References')) | (pl.col('References').is_in(pl.col('Doi')))))
                    .to_numpy()
                ).all()
            )

            ################
            # Remove self-loops
            ################

            df = (
                df
                .select(pl.all().exclude('References'))
                .join(
                    other = (
                        df
                        .select([pl.col('Doi'), pl.col('References')])
                        .explode(pl.col('References'))
                        .filter(~pl.col('References').eq(pl.col('Doi')))
                        .groupby(pl.col('Doi'))
                        .agg(pl.col('References'))
                    ),
                    on='Doi',
                    how='left'
                )
            )
            assert(
                (
                    df
                    .select([pl.col('Doi'), pl.col('References')])
                    .explode(pl.col('References'))
                    .select(~pl.col('References').eq(pl.col('Doi')))
                    .to_numpy()
                ).all()
            )

            ################
            # Doc date > Reference date
            ################

            df = (
                df
                .explode('References')
                .join(other=df.select('Doi', 'Date'), left_on='References', right_on='Doi', how='left', suffix='References')
                .filter(pl.col('Date').gt(pl.col('DateReferences')) | pl.col('References').is_null())
                .drop('DateReferences')
                .groupby(pl.col('Doi'))
                .agg(pl.col('References'))
                .with_columns(pl.col('References').arr.eval(pl.element().filter(~pl.element().is_null())))
                .join(other=df.select(pl.all().exclude('References')), on='Doi', how='left')
                
            )
            assert(
                (
                    df
                    .select([pl.col('Doi'), pl.col('Date'), pl.col('References')])
                    .explode(pl.col('References'))
                    .join(other=df.select([pl.col('Doi'), pl.col('Date')]), left_on='References', right_on='Doi', how='left', suffix='Reference')
                    .filter(pl.col('Date').gt(pl.col('DateReference')))
                    .to_numpy()
                ).all()
            )

            ################
            # Add Display Author Column
            ################

            df = (
                df
                .with_columns(
                    pl.col('Authors')
                        .arr.eval(
                            pl.element()
                                .str.split(', ')
                                .arr.eval(pl.element().first())
                                .arr.first()
                        )
                        .arr.eval(pl.element().apply(lambda s: s.capitalize()))
                        .alias('AuthorsDisplay'),
                    pl.col('Date').dt.year().cast(pl.Utf8).alias('DateDisplay'),
                )
                .with_columns(
                    pl.when(pl.col('AuthorsDisplay').arr.lengths().gt(1))
                    .then(pl.col('AuthorsDisplay').arr.slice(0,1).arr.concat(pl.lit('et al.')))
                    .otherwise(pl.col('AuthorsDisplay'))
                    .arr.join(' ')
                )
                .with_columns(
                    pl.concat_str([
                        pl.col('AuthorsDisplay'),
                        pl.lit(' '),
                        pl.lit('('),
                        pl.col('DateDisplay'),
                        pl.lit(')')
                    ])
                    .alias('AuthorsDisplay')
                )
                .drop(['DateDisplay'])
            )

            ################
            # Non null
            ################

            nulls = ~df.select(pl.all().is_null().any()).to_numpy()
            assert(nulls.all())

            df.write_parquet(os.path.join(CITATION_DF_PATH, field_name + '.parquet'))
            print(field_name, df.shape)

################
# Co-Citation DF
################

def co_citation_dfs():
    folder_name = inspect.currentframe().f_code.co_name
    for file_name in os.listdir(CITATION_DF_PATH):
        field_name = file_name.split('.parquet')[0]
        output_path = os.path.join(OUTPUT_PATH, folder_name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        df = (
            pl.read_parquet(os.path.join(CITATION_DF_PATH, file_name))
            .select('References')
            .filter(pl.col('References').arr.lengths().gt(1))
            .with_columns(pl.col('References').apply(lambda lst: pl.Series(combinations(lst, 2)), return_dtype=pl.List(pl.List(pl.Utf8))))
            .explode('References')
            .with_columns(pl.col('References').arr.sort())
            .with_columns(pl.col('References').arr.first().alias('u'), pl.col('References').arr.last().alias('v'))
            .filter(pl.col('u').ne(pl.col('v')))
            .groupby(['u', 'v'])
            .agg(pl.count().alias('Count'))
            .filter(pl.col('Count').gt(1))
            .sort('Count', descending=True)
            
        )
        df.write_parquet(os.path.join(output_path, field_name + '.parquet'))
        print(field_name, df.shape)

################
# Co-Occurrance DF
################

def co_occurence_dfs(n_samples=None):
    folder_name = inspect.currentframe().f_code.co_name
    for file_name in os.listdir(CITATION_DF_PATH):
        field_name = file_name.split('.parquet')[0]
        output_path = os.path.join(OUTPUT_PATH, folder_name)        
        if not os.path.exists(output_path): os.mkdir(output_path)
        df = pl.read_parquet(os.path.join(CITATION_DF_PATH, file_name))
        n_rows = df.shape[0]
        match n_samples:
            case _ if n_samples > n_rows:
                df = df.sample(n=n_rows)
            case _ if 0 < n_samples <= n_rows:
                df = df.sample(n=n_samples)
        df = (
            df
            .select('Text')
            .filter(pl.col('Text').arr.lengths().gt(1))
            .with_columns(pl.col('Text').apply(lambda lst: pl.Series(combinations(lst, 2)), return_dtype=pl.List(pl.List(pl.Utf8))))
            .explode('Text')
            .with_columns(pl.col('Text').arr.sort())
            .with_columns(pl.col('Text').arr.first().alias('u'), pl.col('Text').arr.last().alias('v'))
            .filter(pl.col('u').ne(pl.col('v')))
            .groupby(['u', 'v'])
            .agg(pl.count().alias('Count'))
            .filter(pl.col('Count').gt(1))
            .sort('Count', descending=True)
        )
        df.write_parquet(os.path.join(output_path, field_name + '.parquet'))
        print(field_name, df.shape)

################
# Temporal DF
################

def temporal_dfs():
    folder_name = inspect.currentframe().f_code.co_name
    for file_name in os.listdir(CITATION_DF_PATH):
        field_name = file_name.split('.parquet')[0]
        output_path = os.path.join(OUTPUT_PATH, folder_name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        df = pl.read_parquet(os.path.join(CITATION_DF_PATH, file_name))
        df = (
            df
            .groupby(pl.col('Date').dt.year().alias('Year'))
            .agg(pl.col('Doi'))
            .sort(pl.col('Year'))
            .groupby_rolling(index_column='Year', period=f'{df.height}i')
            .agg([
                pl.col('Doi'),
                pl.col('Doi').flatten().alias('CumDoi'),
            ])
            .with_columns([
                pl.col('Doi').arr.last(),
                pl.col('Doi').arr.last().arr.lengths().alias('DoiCount'),
                pl.col('CumDoi').arr.lengths().alias('CumDoiCount'),
            ])
        )
        df.write_parquet(os.path.join(output_path, field_name + '.parquet'))
        print(field_name, df.shape)

################
# Citation Graphs
################

def citation_graphs():
    output_path = os.path.join(OUTPUT_PATH, inspect.currentframe().f_code.co_name)
    if not os.path.exists(output_path): os.mkdir(output_path)
    Gs = defaultdict(dict)
    for file_name in os.listdir(CITATION_DF_PATH):
        field_name = file_name.split('.parquet')[0]
        df = pl.read_parquet(os.path.join(CITATION_DF_PATH, file_name))
        edges = (
            df
            .select([pl.col('Doi'), pl.col('References')])
            .explode('References')
            .drop_nulls()
            .to_numpy()
        )
        G = nx.DiGraph()
        G.add_edges_from(map(tuple, edges))
        G.remove_edges_from(map(lambda tup: (tup[-1], tup[0]), nx.simple_cycles(G)))
        G.remove_edges_from(nx.selfloop_edges(G))
        assert(nx.is_directed_acyclic_graph(G))
        Gs[field_name]['G'] = G
        Gs[field_name]['Df'] = df
        nx.write_weighted_edgelist(G, os.path.join(output_path, field_name + '.edgelist'))
        print(field_name, G)
    return Gs

################
# Co-Citation Graphs
################

def co_citation_graphs(n_edges=-1):
    output_path = os.path.join(OUTPUT_PATH, inspect.currentframe().f_code.co_name)
    if not os.path.exists(output_path): os.mkdir(output_path)
    Gs = defaultdict(dict)
    for file_name in os.listdir(CITATION_DF_PATH):
        field_name = file_name.split('.parquet')[0]
        df = pl.read_parquet(os.path.join(CITATION_DF_PATH, file_name))
        edges = (
            pl.read_parquet(os.path.join(OUTPUT_PATH, 'co_citation_dfs', file_name))
            .head(n_edges)
            .to_numpy()
        )
        G = nx.Graph()
        G.add_weighted_edges_from(map(tuple, edges))
        Gs[field_name]['G'] = G
        Gs[field_name]['Df'] = df
        nx.write_weighted_edgelist(G, os.path.join(OUTPUT_PATH, 'co_citation_graphs', field_name + '.edgelist'))
        print(field_name, G)
    return Gs

################
# Co-Occurrence Graphs
################

def co_occurence_graphs(n_edges=-1):
    output_path = os.path.join(OUTPUT_PATH, inspect.currentframe().f_code.co_name)
    if not os.path.exists(output_path): os.mkdir(output_path)
    Gs = defaultdict(dict)
    for file_name in os.listdir(CITATION_DF_PATH):
        field_name = file_name.split('.parquet')[0]
        df = pl.read_parquet(os.path.join(CITATION_DF_PATH, file_name))
        edges = (
            pl.read_parquet(os.path.join(OUTPUT_PATH, 'co_occurence_dfs', file_name))
            .head(n_edges)
            .to_numpy()
        )
        G = nx.Graph()
        G.add_weighted_edges_from(map(tuple, edges))
        Gs[field_name]['G'] = G
        Gs[field_name]['Df'] = df
        nx.write_weighted_edgelist(G, os.path.join(output_path, field_name + '.edgelist'))
        print(field_name, G)
    return Gs

################
# Get author names by DOI
################

def get_node_names(field_name, lst_doi):
    path = os.path.join(CITATION_DF_PATH, field_name + '.parquet')
    df = pl.read_parquet(path)
    return dict(
        df
        .select(
            pl.col('Doi'),
            pl.concat_str([
                pl.lit('('),
                pl.col('Authors').arr.first().str.split(', ').arr.first(),
                pl.lit(', '),
                pl.col('Date').dt.year(),
                pl.lit(')'),
            ])
        )
        .filter(pl.col('Doi').is_in(lst_doi))
        .to_numpy()
    )

################
# Compute Dendrogram Z & leaves
################

def compute_Z(levels):

    comm_ids = defaultdict(dict)
    id = 0
    for (l, level) in enumerate(levels):
        temp = dict()
        for (c, comm) in enumerate(level):
            sorted_comm = tuple(sorted(comm))
            temp[sorted_comm] = id
            id += 1
        comm_ids[l] = temp

    last = len(levels)-1
    edges = []
    leaf_map = {}

    for (l, level) in enumerate(levels[1:]):

        current_level = l + 1
        previous_level = l

        for child_comm in level:
            
            parent_comm = next(filter(lambda parent_comm: child_comm.issubset(parent_comm), levels[l]))
            sorted_parent_comm = tuple(sorted(parent_comm))
            sorted_child_comm = tuple(sorted(child_comm))

            parent_comm_id = comm_ids[previous_level][sorted_parent_comm]
            child_comm_id = comm_ids[current_level][sorted_child_comm]

            edges.append((parent_comm_id, child_comm_id))

            if current_level == last:
                leaf_map[child_comm_id] = next(iter(child_comm))

    G_dendo = nx.DiGraph()
    G_dendo.add_edges_from(edges)

    d = nx.to_dict_of_lists(G_dendo, nodelist=None)

    G_dendo     = nx.DiGraph(d)
    nodes       = G_dendo.nodes()
    leaves      = set( n for n in nodes if G_dendo.out_degree(n) == 0 )
    inner_nodes = [ n for n in nodes if G_dendo.out_degree(n) > 0 ]

    subtree = dict( (n, [n]) for n in leaves )
    for u in inner_nodes:
        children = set()
        node_list = list(d[u])
        while len(node_list) > 0:
            v = node_list.pop(0)
            children.add( v )
            node_list += d[v]

        subtree[u] = sorted(children & leaves)

    inner_nodes.sort(key=lambda n: len(subtree[n]))

    leaves = sorted(leaves)
    index  = dict( (tuple([n]), i) for i, n in enumerate(leaves) )
    Z = []
    k = len(leaves)
    for i, n in enumerate(inner_nodes):
        children = d[n]
        x = children[0]
        for y in children[1:]:
            z = tuple(sorted(subtree[x] + subtree[y]))
            i, j = index[tuple(subtree[x])], index[tuple(subtree[y])]
            Z.append([i, j, float(len(subtree[n])), len(z)])
            index[z] = k
            subtree[z] = list(z)
            x = z
            k += 1

    leaves = [leaf_map[l] for l in leaves]

    return (Z, leaves)


