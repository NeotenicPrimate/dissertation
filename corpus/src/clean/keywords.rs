use polars::{series::Series, prelude::NamedFrom};
use regex::Regex;

pub fn keywords_util(str_series: &Series) -> Series {
    Series::new("",
        str_series.utf8()
            .unwrap()
            .into_iter()
            .map(keywords_split)
            .collect::<Vec<Option<Series>>>()
    )
}

pub fn keywords_split(s: Option<&str>) -> Option<Series> {
    match s {
        Some(s) => {
            let s = s.to_lowercase();
            let re = Regex::new(r"[ ,;&]+").unwrap();
            let s = re.split(s.as_str()).map(str::trim).collect();
            Some(s)
        }
        None => None
    }
}