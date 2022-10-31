use std::collections::{HashSet, HashMap};

use polars::prelude::*;

        // .with_columns([
        //     col("Text")
        //         .apply_many(
        //             tfidf_util,
        //             &[col("Doi")],
        //             GetOutput::from_type(
        //                 DataType::List(
        //                     Box::new(
        //                         DataType::Struct(vec![
        //                             Field::new("term", DataType::Utf8),
        //                             Field::new("tfidf", DataType::Float32),
        //                         ])
        //                     )
        //                 )
        //             )
        //         )
        //         .alias("Tfidf"),
        // ])

pub fn tfidf_util(series: &mut [polars::prelude::Series]) -> Result<Series, PolarsError> {

    let mut series = series.iter();
    let documents = series.next().unwrap().list()?;
    let dois = series.next().unwrap().utf8()?;


    let unique_terms = corpus_unique_terms(&documents);

    let tfs = tf(&dois, &documents)?;

    let idfs = idf(unique_terms, &documents)?;

    let tfidfs = tfidf(dois, tfs, idfs)?;

    let (terms, values): (Vec<Series>, Vec<Series>) = tfidfs
        .into_iter()
        .fold(
            (vec![], vec![]), 
            |(mut terms, mut values), tfidf_map| {
                let term: Utf8Chunked = tfidf_map.keys().map(|term| Some(term)).collect();
                let value: Float32Chunked = tfidf_map.values().map(|val| Some(*val)).collect();
                
                let term = term.into_series();
                let value = value.into_series();

                terms.push(term);
                values.push(value);

                (terms, values)
            }
        );

    let terms: ListChunked = terms.into_iter().collect();
    let values: ListChunked = values.into_iter().collect();

    let tfidfs = StructChunked::new("Tfidf", &[
        terms.into_series(),
        values.into_series(),
    ]).unwrap();
   
    let series = tfidfs.into_series();

    Ok(series)
}

fn corpus_unique_terms(documents: &ListChunked) -> HashSet<String> {
    documents
        .into_iter()
        .enumerate()
        .filter_map(|(i, opt_doc)| {
            print!("\rUnique: {i}");
            if let Some(doc) = opt_doc {
                let doc: Vec<String> = doc
                    .utf8()
                    .unwrap()
                    .into_iter()
                    .filter_map(|opt_word| {
                        if let Some(word) = opt_word {
                            Some(word.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                Some(doc)
            } else {
                None
            }
        })
        .flatten()
        .collect()
}

fn tf<'a>(dois: &'a Utf8Chunked, documents: &'a ListChunked) -> Result<HashMap<&'a str, HashMap<String, f32>>, PolarsError> {
    let mut tfs = HashMap::new();
    for (i, (doi, document)) in dois.into_iter().zip(documents).enumerate() {
        print!("\rTf: {i}");
        if let (Some(doi), Some(doc)) = (doi, document) {
            let doc_len = doc.len();
            let doc: Vec<String> = doc.utf8()?.into_iter().filter_map(|opt_word| {
                if let Some(word) = opt_word {
                    Some(word.to_string())
                } else {
                    None
                }
            }).collect();

            let mut temp_tfs = HashMap::new();
            for term in doc.clone() {
                    let term_count = doc.iter().filter(|t| **t == term).count();
                    let term_tfs = term_count / doc_len;
                    temp_tfs.insert(term, term_tfs as f32);
            }
            tfs.insert(doi, temp_tfs);
        } else {
            continue;
        }
    };
    Ok(tfs)
}

fn idf(unique_terms: HashSet<String>, documents: &ListChunked) -> Result<HashMap<String, f32>, PolarsError> {
    let mut idfs = HashMap::new();
    let num_docs = documents.len();
    for term in unique_terms {
        let mut count = 0;
        for opt_doc in documents {
            if let Some(doc) = opt_doc {
                let doc_all_words: HashSet<String> = doc.utf8()?.into_iter().filter_map(|opt_word| {
                    if let Some(word) = opt_word {
                        Some(word.to_string())
                    } else {
                        None
                    }
                }).collect();
                if doc_all_words.contains(&term) {
                    count += 1;
                }
            }
        }
        let term_idf = (num_docs as f32 / count as f32).ln();
        idfs.insert(term, term_idf);
    };
    Ok(idfs)
}

fn tfidf<'a>(dois: &Utf8Chunked, tfs: HashMap<&'a str, HashMap<String, f32>>, idfs: HashMap<String, f32>) -> Result<Vec<HashMap<String, f32>>, PolarsError> {
    let mut tfidfs = Vec::new();
    for opt_doi in dois {
        if let Some(doi) = opt_doi {
            let mut temp_tfidf = HashMap::new();
            for (term, tf) in tfs.get(doi).unwrap() {
                let term_tfidf = idfs.get(&term.to_string()).unwrap() * tf;
                temp_tfidf.insert(term.to_owned(), term_tfidf);
            }
            tfidfs.push(temp_tfidf);
        }
    }
    Ok(tfidfs)
}