
import polars as pl
import networkx as nx

from nltk.stem import WordNetLemmatizer

import os
import collections
import itertools

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

stopwords = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "year", "denmark", "shrm", 
"during", "out", "very", "from", "re", "edu", "use", "published", "a", "has", "elsevier", "may", "paper", "studi", "discuss",
"having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "new",
"itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", 
"below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", 
"this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", 
"when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", 
"then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", 
"has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", 
"theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]

abstracts = {
    # ORGANIZATIONS
    '10.1177/014920639101700108': 'Understanding sources of sustained competitive advantage has become a major area of research in strategic management. Building on the assumptions that strategic resources are heterogeneously distributed acrossfirms and that these differences are stable over time, this article examines the link betweenfirm resources and sustained competitive advantage. Four empirical indicators of the potential of firm resources to generate sustained competitive advantage-value, rareness, imitability, and substitutability-are discussed. The model is applied by analyzing the potential of severalfirm resourcesfor generating sustained competitive advantages. The article concludes by examining implications of this firm resource model of sustained competitive advantage for other business disciplines.',
    '10.2307/258557': 'This paper describes the process of inducting theory using case studies from specifying the research questions to reaching closure. Some features of the process, such as problem definition and construct validation, are similar to hypothesis-testing research. Others, such as within-case analysis and replication logic, are unique to the inductive, case-oriented process. Overall, the process described here is highly iterative and tightly linked to data. This research approach is especially appropriate in new topic areas. The resultant theory is often novel, testable, and empirically valid. Finally, framebreaking insights, the tests of good theory (e.g., parsimony, logical coherence), and convincing grounding in the evidence are the key criteria for evaluating this type of research',
    '10.2307/2095101': 'What makes organizations so similar? We contend that the engine of rationalization and bureaucratization has moved from the competitive marketplace to the state and the professions. Once a set of organizations emerges as a field, a paradox arises: rational actors make their organizations increasingly similar as they try to change them. We describe three isomorphic processes--coercive, mimetic, and normative--leading to this outcome. We then specify hypotheses about the impact of resource centralization and dependency, goal ambiguity and technical uncertainty, and professionalization and structuration on isomorphic change. Finally, we suggest implications for theories of organizations and social change.',
    '10.2307/2393553': 'In this paper, we argue that the ability of a firm to recognize the value of new, external information, assimilate it, and apply it to commercial ends is critical to its innovative capabilities. We label this capability a firms absorptive capacity and suggest that it is largely a function of the firms level of prior related knowledge. The discussion focuses first on the cognitive basis for an individuals absorptive capacity including, in particular, prior related knowledge and diversity of background. We then characterize the factors that influence absorptive capacity at the organizational level, how an organizations absorptive capacity differs from that of its individual members, and the role of diversity of expertise within an organization. We argue that the development of absorptive capacity, and, in turn, innovative performance are history- or path-dependent and argue how lack of investment in an area of expertise early on may foreclose the future development of a technical capability in that area. We formulate a model of firm investment in research and development (R&D), in which R&D contributes to a firms absorptive capacity, and test predictions relating a firms investment in R&D to the knowledge underlying technical change within an industry. Discussion focuses on the implications of absorptive capacity for the analysis of other related innovative activities, including basic research, the adoption and diffusion of innovations, and decisions to participate in cooperative R&D ventures.',
    '10.1086/226550': 'Many formal organizational structures arise as reflections of rationalized institutional rules. The elaboration of such rules in modern states and societies accounts in part for the expansion and increased complexity of formal organizational structures. Institutional rules function as myths which organizations incorporate, gaining legitimacy, resources, stability, and enhanced survival prospects. Organizations whose structures become isomorphic with the myths of the institutional environment-in contrast with those primarily structured by the demands of technical production and exchange-decrease internal coordination and control in order to maintain legitimacy. Structures are decoupled from each other and from ongoing activities. In place of coordination, inspection, and evaluation, a logic of confidence and good faith is employed.',
    '10.2307/258434': 'Theorists in various fields have discussed characteristics of top managers. This paper attempts to synthesize these previously fragmented literatures around a more general "upper echelons perspective." The theory states that organizational outcomes-strategic choices and performance levels-are partially predicted by managerial background characteristics. Propositions and methodological suggestions are included.',
    '10.1086/228311': 'How behavior and institutions are affected by social relations is one of the classic questions of social theory. This paper concerns the extent to which economic action is embedded in structures of social relations, in modern industrial society. Although the usual neoclasical accounts provide an "undersocialized" or atomized-actor explanation of such action, reformist economists who attempt to bring social structure back in do so in the "oversocialized" way criticized by Dennis Wrong. Under-and oversocialized accounts are paradoxically similar in their neglect of ongoing structures of social relations, and a sophisticated account of economic action must consider its embeddedness in such structures. The argument in illustrated by a critique of Oliver Williamsons markets and hierarchies research program.',
    '10.1086/225469': 'Analysis of social networks is suggested as a tool for linking micro and macro levels of sociological theory. The procedure is illustrated by elaboration of the macro implications of one aspect of small-scale interaction: the strength of dyadic ties. It is argued that the degree of overlap of two individuals friendship networks varies directly with the strength of their tie to one another. The impact of this principle on diffusion of influence and information, mobility opportunity, and community organization is explored. Stress is laid on the cohesive power of weak ties. Most network models deal, implicitly, with strong ties, thus confining their applicability to small, well-defined groups. Emphasis on weak ties lends itself to discussion of relations between groups and to analysis of segments of social structure not easily defined in terms of primary groups.',
    '10.1177/014920638601200408': 'Self-reports figure prominently in organizational and management research, but there are several problems associated with their use. This article identifies six categories of self-reports and discusses such problems as common method variance, the consistency motif, and social desirability. Statistical and post hoc remedies and some procedural methods for dealing with artifactual bias are presented and evaluated. Recommendations for future research are also offered.',
    '10.2307/2095567': 'Theory and research on organization-environment relations from a population ecology perspective have been based on the assumption that inertial pressures on structure are strong. This paper attempts to clarify the meaning of structural inertia and to derive propositions about structural inertia from an explicit evolutionary model. The proposed theory treats high levels of structural inertia as a consequence of a selection process rather than as a precondition for selection. It also considers how the strength of inertial forces varies with age, size, and complexity.',
    '10.1086/226424': 'A population ecology perspective on organization-environment relations is proposed as an alternative to the dominant adaptation perspective. The strength of inertial pressures on organizational structure suggests the application of models that depend on competition and selection in populations of organizations. Several such models as well as issues that arise in attempts to apply them to the organization-environment problem are discussed.',
    '10.1086/228943': 'In this paper, the concept of social capital is introduced and illustrated, its forms are described, the social structural conditions under which it arises are examined, and it is used in an analysis of dropouts from high school. Use of the concept of social capital is part of a general theoretical strategy discussed in the paper: taking rational action as a starting point but rejecting the extreme individualistic premises that often accompany it. The conception of social capital as a resource for action is one way of introducing social structure into the rational action paradigm. Three forms of social capital are examined: obligations and expectations, information channels, and social norms. The role of closure in the social structure in facilitating the first and third of these forms of social capital is described. An analysis of the effect of the lack of social capital available to high school sophomores on dropping out of school before graduation is carried out. The effect of social capital within the family and in the community outside the family is examined.',
    '10.2307/258610': 'This article applies the convergent insights of institutional and resource dependence perspectives to the prediction of strategic responses to institutional processes. The article offers a typology of strategic responses that vary in active organizational resistance from passive conformity to proactive manipulation. Ten institutional factors are hypothesized to predict the occurrence of the alternative proposed strategies and the degree of organizational conformity or resistance to institutional pressures.',
    '10.2307/2393549': 'This paper demonstrates that the traditional categorization of innovation as either incremental or radical is incomplete and potentially misleading and does not account for the sometimes disastrous effects on industry incumbents of seemingly minor improvements in technological products. We examine such innovations more closely and, distinguishing between the components of a product and the ways they are integrated into the system that is the product "architecture," define them as innovations that change the architecture of a product without changing its components. We show that architectural innovations destroy the usefulness of the architectural knowledge of established firms, and that since architectural knowledge tends to become embedded in the structure and information-processing procedures of established organizations, this destruction is difficult for firms to recognize and hard to correct. Architectural innovation therefore presents established organizations with subtle challenges that may have significant competitive implications. We illustrate the concepts explanatory force through an empirical study of the semiconductor photolithographic alignment equipment industry, which has experienced a number of architectural innovations.',
    '10.2307/2392832': 'This paper focuses on patterns of technological change and on the impact of technological breakthroughs on environmental conditions. Using data from the minicomputer, cement, and airline industries from their births through 1980, we demonstrate that technology evolves through periods of incremental change punctuated by technological break-throughs that either enhance or destroy the competence of firms in an industry. These breakthroughs, or technological discontinuities, significantly increase both environmental uncertainty and munificence. The study shows that while competence-destroying discontinuities are initiated by new firms and are associated with increased environmental turbulence, competence-enhancing discontinuities are initiated by existing firms and are associated with decreased environmental turbulence. These effects decrease over successive discontinuities. Those firms that initiate major technological changes grow more rapidly than other firms.',
    '10.2307/2393080': 'Industrial classifications were used as a basis for operational definitions of both industrial and organizational task environments. A codification of six environmental dimensions was reduced to three: munificence (capacity), complexity (homogeneity-heterogeneity, concentration-dispersion), and dynamism (stability-instability, turbulence). Interitem and factor analytic techniques were used to explore the viability of these environmental dimensions. Implications of the research for building both descriptive and normative theory about organization-environment relationships are advanced.',
    '10.1002/smj.4250100107': 'This paper reports the results of a study designed to investigate the effective strategic responses to environmental hostility among small manufacturing firms. Data on environmental hostility, organization structure, strategic posture, competitive tactics, and financial performance were collected from 161 small manufacturers. Findings indicate that performance among small firms in hostile environments was positively related to an organic structure, an entrepreneurial strategic posture, and a competitive profile characterized by a long-term orientation, high product prices, and a concern for predicting industry trends. In benign environments, on the other hand, performance was positively related to a mechanistic structure, a conservative strategic posture, and a competitive profile characterized by conservative financial management and a short-term financial orientation, an emphasis on product refinement, and a willingness to rely heavily on single customers.',
    '10.1002/smj.4250120908': 'Global competition highlights asymmetries in the skill endowments of firms. Collaboration may provide an opportunity for one partner to internalize the skills of the other, and thus improve its position both within and without the alliance. Detailed analysis of nine international alliances yielded a fine-grained understanding of the determinants of interpartner learning. The study suggests that not all partners are equally adept at learning; that asymmetries in learning alter the relative bargaining power of partners; that stability and longevity may be inappropriate metrics of partnership success; that partners may have competitive, as well as collaborative aims, vis-à-vis each other; and that process may be more important than structure in determining learning outcomes.',
    '10.2307/2393356': 'This paper combines institutional economics with aspects of contract law and organization theory to identify and explicate the key differences that distinguish three generic forms of economic organization-market, hybrid, and hierarchy. The analysis shows that the three generic forms are distinguished by different coordinating and control mechanisms and by different abilities to adapt to disturbances. Also, each generic form is supported and defined by a distinctive type of contract law. The cost-effective choice of organization form is shown to vary systematically with the attributes of transactions. The paper unifies two hitherto disjunct areas of institutional economics-the institutional environment and the institutions of governance-by treating the institutional environment as a locus of parameters, changes in which parameters bring about shifts in the comparative costs of governance. Changes in property rights, contract law, reputation effects, and uncertainty are investigated.',
    '10.2307/256434': 'Explored the speed of strategic decisions (STDs) in a high-velocity (i.e., rapidly changing) environment, using a multiple case design to study 8 microcomputer firms. Data on top management team, strategic decision, and firm performance were obtained from chief executive officer interviews, semistructured interviews with top management, questionnaires completed by each team member, and secondary sources (e.g., industry reports). Fast decision makers used more information, developed more alternatives, and used a 2-tiered advice process when compared with slow decision makers. Conflict resolution and integration among STDs and tactical plans were critical to the pace of decision making. Findings suggest that a configuration of cognitive, political, and emotional processes is associated with rapid closure on STDs.',
    '10.2307/258441': 'A comparative model of organizations as interpretation systems is proposed. The model describes four interpretation modes: enacting, discovering, undirected viewing, and conditioned viewing. Each mode is determined by (1) managements beliefs about the environment and (2) organizational intrusiveness. Interpretation modes are hypothesized to be associated with organizational differences in environmental scanning, equivocality reduction, strategy, and decision making.',
    '10.1002/smj.4250090403': 'This paper compares the perspectives of transaction costs and strategic behavior in explaining the motivation to joint venture. In addition, a theory of joint ventures as an instrument of organizational learning is proposed and developed. Existing studies of joint ventures are examined in light of these theories. Data on the sectoral distribution and stability of joint ventures are presented.',
    '10.2307/258189': 'It is argued that (a) social identification (SID) is a perception of oneness with a group of persons; (b) SID stems from the categorization of individuals, the distinctiveness and prestige of the group, the salience of outgroups, and the factors that traditionally are associated with group formation; and (c) SID leads to activities that are congruent with the identity, support for institutions that embody the identity, and stereotypical perceptions of self and others. SID also leads to outcomes that traditionally are associated with group formation and reinforces its own antecedents. This perspective is applied to organizational socialization, role conflict, and intergroup relations.',
    '10.1002/smj.4250120604': 'This paper reports an ethnographic study of the initiation of a strategic change effort in a large, public university. It develops a new framework for understanding the distinctive character of the beginning stages of strategic change by tracking the first year of the change through four phases (labeled as envisioning, signaling, re-visioning, and energizing). This interpretive approach suggests that the CEO primary role in instigating the strategic change process might best be understood in terms of the emergent concepts of ‘sensemaking’ and ‘sensegiving’. Relationships between these central concepts and other important theoretical domains are then drawn and implications for understanding strategic change initiation are discussed.',
    '10.1002/smj.4250060306': 'Deliberate and emergent strategies may be conceived as two ends of a continuum along which real-world strategies lie. This paper seeks to develop this notion, and some basic issues related to strategic choice, by elaborating along this continuum various types of strategies uncovered in research. These include strategies labelled planned, entrepreneurial, ideological, umbrella, process, unconnected, consensus and imposed.',

    # GENOMICS

    # ENGLISH

}

old_cols, new_cols = zip(*cols)
dtypes = [pl.Utf8] * len(cols)

class Corpora():

    data: dict = collections.defaultdict(dict)

    def __init__(self, data_path):

        for field_name in os.listdir(data_path):
            if not field_name.startswith('.'):
                
                field_dfs = []
                for journal_name in os.listdir(os.path.join(data_path, field_name)):
                    if not journal_name.startswith('.'):

                        df = pl.read_csv(
                            file=f"./data/Subcategory Data/{field_name}/{journal_name}/savedrecs*.txt",
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

                        field_dfs.append(df)

                df = pl.concat(field_dfs)
                
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

                df = (
                    df
                    .join(
                        pl.DataFrame([
                            pl.Series('Doi', list(abstracts.keys())),
                            pl.Series('Abstract', list(abstracts.values())),
                        ]), 
                        on='Doi', 
                        how='left'
                    )
                    .with_columns(pl.when(pl.col("Abstract").is_null()).then(pl.col('Abstract_right')).otherwise(pl.col('Abstract')).alias('Abstract'))
                    .drop(['Abstract_right'])
                )

                # Date & Text
                df = (
                    df
                    .with_columns([

                        pl.date(pl.col("Year"), pl.col("Month"), pl.lit(1)).alias("Date"),

                        pl.concat_str([
                            pl.col("Title"), 
                            pl.col("Abstract").fill_null(pl.lit("")), 
                            pl.col("Authors").arr.join(" ").fill_null(pl.lit("")),
                        ], " ")
                        .str.to_lowercase()
                        .str.replace_all(r"[^a-z\s]", " ")
                        .str.replace_all(r"\s+", " ")
                        .alias("Text")
                    ])
                    .drop(["Year", "Month"])
                    .drop_nulls(["Doi", "Authors", "Date", "Title", "References", "Text"]) # "Abstract"
                )

                # Clean Text
                lemmatizer = WordNetLemmatizer()
                df = (
                    df
                    .select([pl.all().exclude(["Text"])])
                    .join(
                        other=df
                        .select([
                            pl.col("Doi"),
                            pl.col("Text").str.split(" "),
                        ])
                        .explode("Text")
                        .with_column(pl.col("Text").apply(lambda s: lemmatizer.lemmatize(s)))
                        .filter(pl.col("Text").str.lengths() >= 3)
                        .filter(~pl.col("Text").is_in(stopwords))
                        .groupby("Doi")
                        .agg(pl.col("Text"))
                        .with_column(pl.col("Text").arr.join(" ")),
                        on="Doi",
                        how="inner",
                    )
                )

                # Prune dois, references
                df = (
                    df
                    .select([pl.all().exclude(["References"])])
                    .join(
                        other=df
                        .select([pl.col("Doi"), pl.col("Date"), pl.col("References")])
                        .explode(["References"])
                        .filter(pl.col("References").is_in(pl.col("Doi")))
                        .join(
                            other=df.select([pl.col("Doi"), pl.col("Date")]),
                            left_on=pl.col("References"),
                            right_on=pl.col("Doi"),
                            how="left",
                        )
                        .filter(pl.col("Date") >= pl.col("Date_right"))
                        .groupby([pl.col("Doi")])
                        .agg([pl.col("References").list()]),
                        on="Doi",
                        how="inner",
                    )
                )

                self.data[field_name]['citation_df'] = df

                dois, references, adjlist = (
                    df
                    .select([
                        pl.col('Doi'),
                        pl.col('References'),
                        pl.concat_list([
                            pl.col("Doi"),
                            pl.col("References"),
                        ])
                        .arr.join(" ")
                        .alias('AdjList'),
                    ])
                )

                citation_graph = nx.parse_adjlist(adjlist, nodetype=str, create_using=nx.DiGraph)

                nodes_not_in_dois = set(citation_graph.nodes()).difference(set(dois))
                citation_graph.remove_nodes_from(nodes_not_in_dois)

                large_component = max(nx.weakly_connected_components(citation_graph), key=len)
                citation_graph = nx.DiGraph(nx.subgraph(citation_graph, large_component))

                citation_graph.remove_nodes_from(nx.isolates(citation_graph))
                citation_graph.remove_edges_from(nx.selfloop_edges(citation_graph))
                citation_graph.remove_edges_from([(cycle[-1], cycle[0]) for cycle in nx.simple_cycles(citation_graph)])

                self.data[field_name]['citation_graph'] = citation_graph

                edges = [edge for reference in references for edge in itertools.combinations(reference, 2)]

                weighted_edges = [(u, v, weight) for ((u, v), weight) in collections.Counter(edges).items()]

                co_citation_graph = nx.Graph()
                co_citation_graph.add_weighted_edges_from(weighted_edges)

                large_component = max(nx.connected_components(co_citation_graph), key=len)
                co_citation_graph = nx.Graph(nx.subgraph(co_citation_graph, large_component))

                co_citation_graph.remove_nodes_from(nx.isolates(co_citation_graph))
                co_citation_graph.remove_edges_from(nx.selfloop_edges(co_citation_graph))

                self.data[field_name]['co_citation_graph'] = co_citation_graph

                print(
                    f'''
{field_name}
    DataFrame: {df.shape}
    Citation: {citation_graph}
    Co-Citation: {co_citation_graph}
                    '''
                )