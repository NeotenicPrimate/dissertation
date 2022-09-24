use polars::{series::Series, prelude::NamedFrom};

use super::utils::text_clean;



pub fn journal_util(str_series: &Series) -> Series {
    Series::new("",
        str_series.utf8()
            .unwrap()
            .into_iter()
            .map(text_clean)
            .collect::<Vec<Option<String>>>()
    )
}