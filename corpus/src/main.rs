mod clean;
mod utils;

use std::fs::OpenOptions;

use polars::prelude::*;
use polars::prelude::UniqueKeepStrategy;
use polars::prelude::DataType;
use polars::prelude::when;

use crate::clean::text::text_util;
use crate::utils::abstract_lf;
use crate::utils::{drop_nulls, prune_references, import_files};

const OLD_COLS: [&str; 12] = ["DI","AU","PY","PD","TI","AB","SO","DE","ID","WC","SC","CR"];
const NEW_COLS: [&str; 12] = ["Doi","Authors","Year","Month","Title","Abstract","Journal","AuthorKeywords","WosKeywords","Category","Areas","References"];

#[tokio::main]
async fn main() -> Result<(), PolarsError> {

    let now = std::time::Instant::now();

    let df = import_files("./data/organizations").await?;
    println!("Import: {} secs", now.elapsed().as_secs());

    let df = df
        .lazy()
        .unique_stable(None, UniqueKeepStrategy::First)
        .rename(OLD_COLS, NEW_COLS)

        .join(abstract_lf()?, [col("Doi")], [col("Doi")], JoinType::Left)

        .with_columns([

            col("Doi")
                .str().strip(None),

            when(col("Abstract").is_null())
                .then(col("Abstract_right"))
                .otherwise(col("Abstract"))
                .str().to_lowercase()
                .str().strip(None)
                .alias("Abstract"),

            when(col("Month").str().starts_with("JAN")).then(lit(1))
            .when(col("Month").str().starts_with("FAL")).then(lit(1))
            .when(col("Month").str().starts_with("FEB")).then(lit(2))
            .when(col("Month").str().starts_with("MAR")).then(lit(3))
            .when(col("Month").str().starts_with("APR")).then(lit(4))
            .when(col("Month").str().starts_with("SPR")).then(lit(4))
            .when(col("Month").str().starts_with("MAY")).then(lit(5))
            .when(col("Month").str().starts_with("JUN")).then(lit(6))
            .when(col("Month").str().starts_with("JUL")).then(lit(7))
            .when(col("Month").str().starts_with("SUM")).then(lit(7))
            .when(col("Month").str().starts_with("AUG")).then(lit(8))
            .when(col("Month").str().starts_with("SEP")).then(lit(9))
            .when(col("Month").str().starts_with("WIN")).then(lit(10))
            .when(col("Month").str().starts_with("OCT")).then(lit(10))
            .when(col("Month").str().starts_with("NOV")).then(lit(11))
            .when(col("Month").str().starts_with("DEC")).then(lit(12))
            .otherwise(lit(1))
            .cast(DataType::UInt32)
            .alias("Month"),

            col("Authors").str().to_lowercase().str().split(";"), // TRIM
            col("WosKeywords").str().to_lowercase().str().split(";"), // TRIM
            col("AuthorKeywords").str().to_lowercase().str().split(";"), // TRIM
            col("Areas").str().to_lowercase().str().split(";"), // TRIM
            col("Category").str().to_lowercase().str().split(";"), // TRIM
            col("Journal").str().to_lowercase().str().strip(None),
            col("Title").str().to_lowercase().str().strip(None),

        ])
        .with_columns([

            datetime(
                DatetimeArgs {
                    year: col("Year"),
                    month: col("Month"),
                    day: lit(1),
                    hour: None,
                    minute: None,
                    second: None,
                    microsecond: None,
                })
                .alias("Date"),
            
            col("References").str().extract_all(r"10.\d{4,9}/[-._()/:a-zA-Z0-9]+"),
            
            concat_str(
                [
                    col("Title"), 
                    col("Abstract").fill_null(lit("")), 
                    col("AuthorKeywords").arr().join(" ").fill_null(lit(""))
                ], 
                " "
            )
                .map(text_util,
                    GetOutput::from_type(DataType::Utf8))
                .alias("Text")
        ])
        .drop_columns(["Year", "Month", "Abstract_right"])
        .unique_stable(Some(vec!["Doi".to_owned()]), UniqueKeepStrategy::First);

    let df = drop_nulls(df);
    let df = prune_references(df);
    let df = drop_nulls(df);
    let df = prune_references(df);
    // let df = drop_nulls(df);
    // let df = prune_references(df);
    // let df = drop_nulls(df);
    // let df = prune_references(df);

    let mut df = df.collect()?;

    println!("Collected: {} secs", now.elapsed().as_secs());

    ParquetWriter::new(
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("../output/df.parquet")?
    ).finish(&mut df)?;

    println!("Write to parquet: {} secs", now.elapsed().as_secs());

    println!("{:?}", df.get_column_names());
    println!("{:?}", df.null_count());
    println!("{}", df);
    
    Ok(())
}


