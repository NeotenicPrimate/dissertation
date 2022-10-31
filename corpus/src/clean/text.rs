use polars::prelude::{PolarsError, Utf8Chunked, Series, IntoSeries};

pub static STOPWORDS: [&str; 136] = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", 
"during", "out", "very", "from", "re", "edu", "use", "published", "a", "has", "elsevier", "may", 
"having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", 
"itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", 
"below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", 
"this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", 
"when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", 
"then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", 
"has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", 
"theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"];

pub fn text_util(series: Series) -> Result<Series, PolarsError> {

    let re_text = regex::Regex::new(r"[^a-zA-Z\s]")?;

    let c: Utf8Chunked = series
        .utf8()?
        .into_iter()
        .map(|opt_s| {
            if let Some(s) = opt_s {
                let s = s.to_lowercase();
                let s = re_text.replace_all(&s, "");
                let c: String = s
                    .split(" ")
                    .filter_map(|word| {
                        if (word.chars().count() > 2) & !STOPWORDS.contains(&word) {
                            let mut chars = word.chars();
                            if !(chars.next().unwrap() == chars.next().unwrap()) {
                                Some(word.trim())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<&str>>()
                    .join(" ");
                Some(c)
            } else {
                None
            }
        })
        .collect();
    Ok(c.into_series())
}