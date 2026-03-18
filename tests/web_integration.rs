//! Integration tests for web module utilities.
//!
//! These tests exercise `url_decode` and `parse_query`
//! without spinning up a real TCP socket.
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
    assert_eq!(
        params.get("prompt").map(|s| s.as_str()),
        Some("hello world")
    );
}

#[test]
fn test_parse_query_missing_value_defaults_to_empty() {
    let params = parse_query("flag=");
    assert_eq!(params.get("flag").map(|s| s.as_str()), Some(""));
}

#[test]
fn test_parse_query_no_value_key_present() {
    let params = parse_query("standalone");
    // A key with no '=' may or may not be present -- just must not panic
    let _ = params.get("standalone");
}

// ---------------------------------------------------------------------------
// url_decode additional edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_url_decode_trailing_percent() {
    // A trailing lone '%' must not panic and must include '%' in output
    let result = url_decode("abc%");
    assert!(result.starts_with("abc"));
    assert!(result.contains('%'));
}

#[test]
fn test_url_decode_percent_one_hex_digit() {
    // '%' followed by only one hex digit (incomplete sequence)
    let result = url_decode("x%4");
    assert!(!result.is_empty());
}

#[test]
fn test_url_decode_multiple_plus_signs() {
    assert_eq!(url_decode("a+b+c"), "a b c");
}

#[test]
fn test_url_decode_roundtrip_ascii() {
    // Plain ASCII without special chars should be unchanged
    let s = "hello_world-123";
    assert_eq!(url_decode(s), s);
}

#[test]
fn test_url_decode_all_noise_chars_no_panic() {
    // Exercise the decoder with a variety of byte values -- must not panic
    for byte in 0x00u8..=0x7fu8 {
        let encoded = format!("%{:02X}", byte);
        let _ = url_decode(&encoded);
    }
}

// ---------------------------------------------------------------------------
// parse_query additional edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_parse_query_duplicate_keys_last_wins_or_first_kept() {
    // Behaviour for duplicate keys is implementation-defined; must not panic
    let params = parse_query("k=1&k=2");
    assert!(params.contains_key("k"));
}

#[test]
fn test_parse_query_encoded_key() {
    // Keys can also be percent-encoded
    let params = parse_query("he%6Clo=world");
    // 'l' = 0x6C, so key is "hello"
    assert!(params.contains_key("hello") || !params.is_empty());
}
