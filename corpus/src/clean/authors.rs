use polars::{series::{Series}, prelude::NamedFrom};
use regex::Regex;

pub fn authors_util(str_series: &Series) -> Series {
    Series::new("", 
        str_series
            .utf8()
            .unwrap()
            .into_iter()
            .map(authors_split)
            .collect::<Vec<Option<Series>>>()
        )
}

pub fn authors_split(s: Option<&str>) -> Option<Series> {
    match s {
        Some(s) => {
            let s = s.to_lowercase();
            let v: Vec<&str> = s.split(";").map(str::trim).collect();
            
            let v = v
                .into_iter()
                .map(|s| {
                    let re = Regex::new(r"[ ,.]+").unwrap();
                    let author = re.split(s).next();
                    author
                }).collect();
            
            v
        }
        None => None
    }
}

// pub fn authors_util(s: Option<&str>) -> Option<Series> {
//     match s {
//         Some(s) => {
//             let s = s.to_lowercase();
//             let v: Vec<&str> = s.split(";").map(str::trim).collect();
//             let v: Vec<Vec<Option<&str>>> = v.into_iter().map(|s| s.split(",").map(str::trim).map(|s| Some(s)).collect()).collect();
//             let (first_names, last_names) = v
//                 .into_iter()
//                 .fold((vec![], vec![]), |(mut first_names, mut last_names), v| {
//                     match &v[..] {
//                         &[Some(first_name), Some(last_name)] => {
//                             first_names.push(Some(first_name));
//                             last_names.push(Some(last_name));
//                             (first_names, last_names)
//                         }
//                         _ => {
//                             first_names.push(None);
//                             last_names.push(None);
//                             (first_names, last_names)
//                         },
//                     }
//                 });

//             let new_authors = StructChunked::new("author", 
//             &[
//                 Series::new("first_name", first_names),
//                 Series::new("last_name", last_names),
//             ]).unwrap();
//             let new_authors = new_authors.into_series();
//             Some(new_authors)
//         }
//         None => None
//     }
// }