//! Crate-level error type for Every-Other-Token.
//!
//! All internal modules should return `EotError` (or a type convertible to it)
//! rather than `Box<dyn std::error::Error>`.  The top-level `main` converts to
//! `Box<dyn std::error::Error>` at the boundary so callers see a clean message.

use thiserror::Error;

/// Top-level error enum.  Each variant maps to a distinct failure domain so
/// that callers can match on the kind of error rather than string-comparing.
#[derive(Debug, Error)]
pub enum EotError {
    /// An expected API key environment variable was not set.
    #[error("API key not set: {0}")]
    ApiKeyMissing(String),

    /// The user supplied a transform name that could not be parsed.
    #[error("invalid transform '{0}': {1}")]
    InvalidTransform(String, String),

    /// A provider returned a non-success HTTP status.
    #[error("provider HTTP {status} from {url}")]
    ProviderHttp { status: u16, url: String },

    /// The provider response body could not be parsed.
    #[error("provider JSON parse error: {0}")]
    ProviderJson(String),

    /// An underlying HTTP client error (connection refused, timeout, etc.).
    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    /// A JSON serialization / deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A standard I/O error (file not found, permission denied, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// SQLite database error from the experiment store.
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// An arbitrary error from external code that has not yet been migrated to
    /// `EotError`.  Used as a migration shim — new code should not add these.
    #[error("{0}")]
    Other(String),
}

impl From<Box<dyn std::error::Error + Send + Sync>> for EotError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        EotError::Other(e.to_string())
    }
}

impl From<Box<dyn std::error::Error>> for EotError {
    fn from(e: Box<dyn std::error::Error>) -> Self {
        EotError::Other(e.to_string())
    }
}

impl From<String> for EotError {
    fn from(s: String) -> Self {
        EotError::Other(s)
    }
}

impl From<&str> for EotError {
    fn from(s: &str) -> Self {
        EotError::Other(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_missing_message() {
        let e = EotError::ApiKeyMissing("OPENAI_API_KEY".to_string());
        assert!(e.to_string().contains("OPENAI_API_KEY"));
    }

    #[test]
    fn test_invalid_transform_message() {
        let e = EotError::InvalidTransform("foo".to_string(), "unknown".to_string());
        assert!(e.to_string().contains("foo"));
        assert!(e.to_string().contains("unknown"));
    }

    #[test]
    fn test_provider_http_message() {
        let e = EotError::ProviderHttp {
            status: 429,
            url: "https://api.openai.com".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains("429"));
        assert!(msg.contains("openai.com"));
    }

    #[test]
    fn test_other_wraps_string() {
        let e: EotError = EotError::from("something went wrong");
        assert_eq!(e.to_string(), "something went wrong");
    }

    #[test]
    fn test_from_string() {
        let e: EotError = "my error".to_string().into();
        assert_eq!(e.to_string(), "my error");
    }

    #[test]
    fn test_io_error_wraps() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let e = EotError::Io(io);
        assert!(e.to_string().contains("file missing"));
    }

    #[test]
    fn test_provider_json_message() {
        let e = EotError::ProviderJson("unexpected field".to_string());
        assert!(e.to_string().contains("unexpected field"));
    }

    #[test]
    fn test_json_error_wraps() {
        let json_err = serde_json::from_str::<serde_json::Value>("{{invalid").unwrap_err();
        let e = EotError::Json(json_err);
        assert!(e.to_string().contains("JSON error"));
    }

    #[test]
    fn test_debug_format() {
        let e = EotError::ApiKeyMissing("TEST_KEY".to_string());
        let debug = format!("{:?}", e);
        assert!(debug.contains("ApiKeyMissing"));
    }
}
