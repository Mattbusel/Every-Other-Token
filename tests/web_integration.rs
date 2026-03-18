/// Integration tests for web module utilities.
///
/// These tests exercise `url_decode` and `parse_query`
/// without spinning up a real TCP socket.
use every_other_token::web::{parse_query, url_decode};

// ---------------------------------------------------------------------------
// url_decode
// ---------------------------------------------------------------------------

#[test]
fn test_url_decode_plain_text() {
    assert_eq!(url_decode("hello"), "hello");
}

#[test]
fn test_url_decode_plus_is_space() {
    assert_eq!(url_decode("hello+world"), "hello world");
}

#[test]
fn test_url_decode_percent_encoded_ascii() {
    assert_eq!(url_decode("hello%20world"), "hello world");
}

#[test]
fn test_url_decode_percent_encoded_utf8() {
    // "é" = %C3%A9
    assert_eq!(url_decode("%C3%A9"), "é");
}

#[test]
fn test_url_decode_mixed_encoding() {
    assert_eq!(url_decode("foo%20bar+baz"), "foo bar baz");
}

#[test]
fn test_url_decode_invalid_percent_sequence_passes_through() {
    // %ZZ is not valid hex — the decoder should not panic
    let result = url_decode("%ZZ");
    assert!(result.contains('%'));
}

#[test]
fn test_url_decode_empty_string() {
    assert_eq!(url_decode(""), "");
}

#[test]
fn test_url_decode_multiple_percent_sequences() {
    // "abc" encoded as %61%62%63
    assert_eq!(url_decode("%61%62%63"), "abc");
}

// ---------------------------------------------------------------------------
// parse_query
// ---------------------------------------------------------------------------

#[test]
fn test_parse_query_single_param() {
    let params = parse_query("key=value");
    assert_eq!(params.get("key").map(|s| s.as_str()), Some("value"));
}

#[test]
fn test_parse_query_multiple_params() {
    let params = parse_query("a=1&b=2&c=3");
    assert_eq!(params.get("a").map(|s| s.as_str()), Some("1"));
    assert_eq!(params.get("b").map(|s| s.as_str()), Some("2"));
    assert_eq!(params.get("c").map(|s| s.as_str()), Some("3"));
}

#[test]
fn test_parse_query_empty_string() {
    // parse_query("") should not panic; result may be empty or contain empty entry
    let _ = parse_query("");
}

#[test]
fn test_parse_query_url_decoded_values() {
    let params = parse_query("prompt=hello+world");
    assert_eq!(params.get("prompt").map(|s| s.as_str()), Some("hello world"));
}

#[test]
fn test_parse_query_missing_value_defaults_to_empty() {
    let params = parse_query("flag=");
    assert_eq!(params.get("flag").map(|s| s.as_str()), Some(""));
}

#[test]
fn test_parse_query_no_value_key_present() {
    let params = parse_query("standalone");
    // A key with no '=' may or may not be present — just must not panic
    let _ = params.get("standalone");
}
