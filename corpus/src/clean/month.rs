use polars::series::Series;



pub fn month_util(str_series: &Series) -> Series {
    str_series
        .utf8()
        .unwrap()
        .into_iter()
        .map(month_str_to_i64)
        .collect()
}

fn month_str_to_i64(s: Option<&str>) -> Option<i64> {
    match s {
        Some(s) => {
            let s = s.trim();
            match &s[..3] {
                "FAL" | "JAN" | "FEB" | "MAR" => Some(1),
                "SPR" | "APR" | "MAY" | "JUN" => Some(4),
                "SUM" | "JUL" | "AUG" | "SEP" => Some(8),
                "WIN" | "OCT" | "NOV" | "DEC" => Some(12),
                _ => None,
            }
        }
        None => None
    }
}