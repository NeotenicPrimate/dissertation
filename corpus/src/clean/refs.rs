use polars::{series::Series, prelude::NamedFrom};

use super::utils::parse_doi;

pub fn refs_util(str_series: &Series) -> Series {
    Series::new("",
        str_series.utf8()
            .unwrap()
            .into_iter()
            .map(refs_split)
            .collect::<Vec<Option<Series>>>()
    )
}

pub fn refs_split(s: Option<&str>) -> Option<Series> {
    match s {
        Some(s) => {
            let s = s.to_lowercase();
            let v: Vec<&str> = s.split(";").collect();
            let v: Vec<&str> = v.into_iter().map(str::trim).filter_map(parse_doi).collect();
            let s = Series::new("ref", v);
            Some(s)
        }
        None => None
    }
}



// pub fn refs_util(s: Option<&str>) -> Option<Series> {
//     match s {
//         Some(s) => {
//             let s = s.to_lowercase();
//             let v: Vec<&str> = s.split(";").collect();
//             let v: Vec<Vec<&str>> = v.into_iter().map(|s| s.split(",").map(str::trim).collect()).collect();
//             let (names, years, dois) = v
//                 .into_iter()
//                 .fold((vec![], vec![], vec![]), |(mut names, mut years, mut dois), v| {
//                     match &v[..] {
//                         &[name, year, .., doi] => {

//                             let re = Regex::new(r"[A-Za-z .]+").unwrap();
//                             match re.is_match(name) {
//                                 true => {
//                                     let name = name.split(" ").next().unwrap();
//                                     names.push(Some(name))
//                                 }
//                                 false => names.push(None),
//                             }

//                             let re = Regex::new(r"(\d{4})").unwrap();
//                             match re.captures(year) {
//                                 Some(caps) => {
//                                     let year = caps.get(2).map_or(None, |m| Some(m.as_str().parse::<i64>().unwrap()));
//                                     years.push(year);
//                                 }
//                                 None => years.push(None)
//                             };

//                             let re = Regex::new(r"(DOI|doi)? ?(10.\d{4}/.+)").unwrap();
//                             match re.captures(doi) {
//                                 Some(caps) => {
//                                     let doi = caps.get(2).map_or(None, |m| Some(m.as_str()));
//                                     dois.push(doi);
//                                 }
//                                 None => dois.push(None)
//                             };

//                             (names, years, dois)
//                         }
//                         _ => {
//                             names.push(None);
//                             years.push(None);
//                             dois.push(None);
//                             (names, years, dois)
//                         }
//                     }
//                 });

//             let new_refs = StructChunked::new("author", 
//             &[
//                 Series::new("name", names),
//                 Series::new("year", years),
//                 Series::new("doi", dois),
//             ]).unwrap();
//             let new_refs = new_refs.into_series();
//             Some(new_refs)
//         }
//         None => None
//     }
// }

