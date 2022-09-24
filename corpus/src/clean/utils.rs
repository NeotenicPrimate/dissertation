use regex::Regex;
use tokenizers::Tokenizer;

use crate::STOP_WORDS;

pub fn text_clean(s: Option<&str>) -> Option<String> {
    match s {
        Some(s) => {
            let s = s.to_lowercase();
            let s = remove_non_words(s);
            let s = remove_stop_words(s);
            Some(s)
        }
        None => None
    }
}

pub fn remove_stop_words(s: String) -> String {
    let pattern = format!("( {} )", STOP_WORDS.join(" | "));
    let re = Regex::new(pattern.as_str()).unwrap();
    let s = re.replace_all(s.as_str(), " ");
    s.into_owned()
}

pub fn remove_non_words(s: String) -> String {
    let re = Regex::new(r"[^\w\s]").unwrap();
    let s = re.replace_all(s.as_str(), " ");
    s.into_owned()
}

pub fn tokenize(text: &str) -> Vec<String> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    let tokens = tokenizer
        .encode(text, false)
        .unwrap()
        .get_tokens()
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();
    tokens
}

pub fn parse_doi(s: &str) -> Option<&str> {
    let re = Regex::new(r"(DOI|doi)? ?(10.\d{4}/.+)").unwrap();
    match re.captures(s) {
        Some(caps) => return caps.get(2).map_or(None, |m| Some(m.as_str())),
        None => return None,
    };
}

pub fn is_valid_doi(s: &str) -> bool {
    let re = Regex::new(r"(DOI|doi)? ?(10.\d{4}/.+)").unwrap();
    re.is_match(s)
}