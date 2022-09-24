mod clean;

use std::collections::HashMap;

use clean::utils::is_valid_doi;
use polars::{prelude::{Series, DataType, CsvReader, SerReader, UniqueKeepStrategy, NamedFrom, DataFrame, AnyValue}};

use chrono::NaiveDate;

use polars::prelude::*;

use crate::clean::{month::month_util, authors::authors_util, abs::abstract_util, journal::journal_util, title::title_util, keywords::keywords_util, refs::refs_util, date::date_util};

const STOP_WORDS: &[&str] = &["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", 
"during", "out", "very", "from", "re", "edu", "use", "published", "a", "has",
"having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", 
"itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", 
"below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", 
"this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", 
"when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", 
"then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", 
"has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", 
"theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"];

fn main() { 

    let now = std::time::Instant::now();
    
    // let mut data: Vec<HashMap<String, String>> = Vec::new();

    // let paths = std::fs::read_dir("./data/scm").unwrap();
    // for path in paths {

    //     let path = path.unwrap();
    //     if path.file_name().into_string().unwrap().starts_with("savedrecs") {
    //         let content = std::fs::read_to_string(path.path()).expect("Unable to read file");
    //         let mut rows = content.split("\r\nJ").map(|row| row.split("\t"));
    //         let cols = rows.next().unwrap().map(String::from);
    //         let rows = rows.map(|row| row.map(String::from));
    //         data = rows.map(|row| cols.clone().zip(row).collect()).collect();
    //     }
    // }

    // let records: Result<Vec<Record>, _> = data.into_iter().map(|row| Record::try_from(row)).collect();

    // match records {
    //     Ok(records) => {
    //         let file = std::fs::File::create("./dataframe.json").unwrap();
    //         serde_json::to_writer(&file, &records).unwrap();
    //     }
    //     Err(e) => { println!("{}", e); }
    // }

    let mut col_map = HashMap::new();
    col_map.insert("AU", "authors");
    col_map.insert("PY", "year");
    col_map.insert("TI", "title");
    col_map.insert("AB", "abstract");
    col_map.insert("SO", "journal");
    col_map.insert("DI", "doi");
    col_map.insert("DE", "kw_author");
    col_map.insert("ID", "kw_plus");
    col_map.insert("WC", "wos_cat");
    col_map.insert("SC", "areas");
    col_map.insert("CR", "refs");
    col_map.insert("NR", "num_refs");
    col_map.insert("PD", "month");
   
    let mut dtype_map = HashMap::new();
    dtype_map.insert("authors", DataType::Utf8);
    dtype_map.insert("year", DataType::Int64);
    dtype_map.insert("title", DataType::Utf8);
    dtype_map.insert("abstract", DataType::Utf8);
    dtype_map.insert("journal", DataType::Utf8);
    dtype_map.insert("doi", DataType::Utf8);
    dtype_map.insert("kw_author", DataType::Utf8);
    dtype_map.insert("kw_plus", DataType::Utf8);
    dtype_map.insert("wos_cat", DataType::Utf8);
    dtype_map.insert("areas", DataType::Utf8);
    dtype_map.insert("refs", DataType::Utf8);
    dtype_map.insert("num_refs", DataType::Int64);
    dtype_map.insert("month", DataType::Utf8);

    let path = "./data/organizations/savedrecs (1).txt";
    let df = CsvReader::from_path(path)
        .unwrap()
        .has_header(true)
        .with_delimiter(b'\t')
        .finish()
        .unwrap();

    let df = df.select(col_map.keys()).unwrap();
    let mut df = df.unique_stable(None, UniqueKeepStrategy::First).unwrap();
    for (old, new) in col_map { df.rename(old, new).unwrap(); };
    let mut df = df.drop_nulls(Some(&["authors", "year", "title", "abstract", "doi", "refs"].map(String::from))).unwrap();

    df.apply("month", month_util).unwrap();
    df.apply("authors", authors_util).unwrap();
    df.apply("kw_plus", keywords_util).unwrap();
    df.apply("kw_author", keywords_util).unwrap();
    df.apply("areas", keywords_util).unwrap();
    df.apply("wos_cat", keywords_util).unwrap();
    df.apply("abstract", abstract_util).unwrap();
    df.apply("journal", journal_util).unwrap();
    df.apply("title", title_util).unwrap();

    df.replace("year",
        Series::new("date", 
            df
                .column("year")
                .unwrap()
                .i64()
                .unwrap()
                .into_iter()
                .zip(df.column("month").unwrap().i64().unwrap().into_iter())
                .map(date_util)
                .collect::<Vec<Option<NaiveDate>>>()
        )
    ).unwrap();
    

    let parent_dois: Vec<String> = df
                        .column("doi")
                        .unwrap()
                        .utf8()
                        .unwrap()
                        .into_iter()
                        .filter_map(|doi| {
                            match doi {
                                Some(doi) => Some(doi.to_string()),
                                None => None,
                            }
                        })
                        .collect();

    df.apply("refs", refs_util).unwrap();
    df.replace("refs", 
        Series::new("refs", 
            df
                .column("refs")
                .unwrap()
                .list()
                .unwrap()
                .into_iter()
                .map(|refs| {
                    match refs {
                        Some(refs) => {
                            let refs = refs
                                .utf8()
                                .unwrap()
                                .into_iter()
                                .filter(|child_doi| 
                                    parent_dois.contains(&child_doi.unwrap().to_string())
                                )
                                .map(|s| match s {
                                    Some(s) => Some(s.to_string()),
                                    None => None,
                                })
                                .collect::<Vec<Option<String>>>();
                            Some(Series::new("", refs))
                        }
                        None => None
                    }
                })
                .collect::<Vec<Option<Series>>>()
            )
    ).unwrap();

    df.select([
        col("refs")
    ]);

    let df = df.drop("month").unwrap();

    // 

    

    // CYCLES
    
    // use polars::prelude::col;

    // let root_doi = "1";
    // let root_row = df.filter(col("doi") == root_doi).unwrap();
    // let root_refs = df.column("refs").unwrap();
    
    // for ref_doi in root_refs.utf8().unwrap() {
    //     let reference = df.filter(col("doi") == ref_doi).unwrap();
    //     let refs_of_ref = reference.column("refs").unwrap();
    //     let is_cycle = refs_of_ref.utf8().unwrap().into_iter().any(|ref_of_ref_doi| {
    //         ref_of_ref_doi.unwrap() == root_doi
    //     });
    // }

    // let dois: Vec<Option<&str>> = df.column("doi").unwrap().utf8().unwrap().into_iter().collect();
    // let dois = dois.into_iter();
    // let refs: Vec<Option<Series>> = df.column("refs").unwrap().list().unwrap().into_iter().collect();
    // let refs = refs.into_iter();
    // for tup in dois.zip(refs) {
    //     match tup {
    //         (Some(doi), Some(refs)) => {
    //             refs.utf8().unwrap().into_iter().filter(|r| {
    //                 match r {
    //                     Some(r) => {

    //                     }
    //                     None => None,
    //                 }
    //             })
    //         }
    //         _ => None,
    //     }
    // }
    


    // ISOLATES

    // let node = "doi1";

    // let all_refs: Vec<&str> = df.column("refs").unwrap().list().unwrap().into_iter().flat_map(|refs| {
    //     refs.unwrap().utf8().unwrap().into_iter().map(|doi| doi.unwrap())
    // }).collect();

    // let has_references = df.column("refs").unwrap().list().unwrap();


    // let nodes = df.filter(col("doi") == all_refs);

    // let df1 = df.select([
    //     col("doi").str().contains(root_doi)
    // ]).unwrap();

    // let v = df1.column("refs").unwrap().head(Some(1)).utf8().unwrap().into_iter().next().unwrap().unwrap();



    // SCHEMA

    // use polars::chunked_array::object::ArrowSchema;
    // use polars::chunked_array::object::ArrowField;
    // use polars::datatypes::ArrowDataType;
    // use std::collections::BTreeMap;

    // let metadata = BTreeMap::new();
    // let fields = vec![
    //     ArrowField::new("authors", ArrowDataType::List(Box::new(ArrowField::new("", ArrowDataType::Utf8, false))), false),
    //     ArrowField::new("doi", ArrowDataType::Utf8, false),
    //     ArrowField::new("refs", ArrowDataType::List(Box::new(ArrowField::new("", ArrowDataType::Utf8, false))), false),
    // ];
    // let schema = ArrowSchema {
    //     fields,
    //     metadata,
    // };

    // let df2 = DataFrame::from_rows_and_schema(
    //     rows,
    //     schema,
    // );


    // ADJACENCY LIST

    // let dois: Vec<Option<String>> = df.column("doi").unwrap().utf8().unwrap().into_iter().map(|s| {
    //     match s {
    //         Some(s) => Some(s.to_string()),
    //         None => None,
    //     }
    // }).collect();
    // let refs: Vec<Option<Vec<String>>> = df.column("refs").unwrap().list().unwrap().into_iter().map(|refs| {
    //     match refs {
    //         Some(refs) => {
    //             let v: Vec<String> = refs.utf8().unwrap().into_iter().filter_map(|s| s).map(String::from).collect();
    //             Some(v)
    //         }
    //         None => None
    //     }
    // }).collect();

    // for (doi, refs) in dois.into_iter().zip(refs.into_iter()) {
    //     match (doi, refs) {
    //         (Some(doi), Some(mut refs)) => {
    //             let row: Vec<String> = refs.splice(0..0, [doi]).collect();
    //         }
    //         _ => continue
    //     }

    // }

    println!("{}", now.elapsed().as_secs());
    println!("{}", df);
    println!("{:?}", df.column("refs").unwrap());  
    println!("{:?}", df.column("refs").unwrap().get(100));  
    
}

