use chrono::NaiveDate;



pub fn date_util(date: (Option<i64>, Option<i64>)) -> Option<NaiveDate> {
    match date {
        (Some(year), Some(month)) => {
            let naivedate = NaiveDate::from_ymd(year as i32, month as u32, 1);
            Some(naivedate)
        },
        (Some(year), None) => {
            let naivedate = NaiveDate::from_ymd(year as i32, 1, 1);
            Some(naivedate)
        },
        _ => None,
    }
}