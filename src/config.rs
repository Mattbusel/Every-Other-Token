//! Optional configuration file support (#16).
//!
//! EOT reads `~/.eot.toml` and then `.eot.toml` in the current directory.
//! Local config wins over the home-dir config. Missing files are silently
//! ignored so that users without a config file see no change in behaviour.
//!
//! Example `.eot.toml`:
//! ```toml
//! provider     = "anthropic"
//! model        = "claude-sonnet-4-6"
//! transform    = "reverse"
//! rate         = 0.5
//! port         = 8888
//! top_logprobs = 5
//! ```

use std::path::PathBuf;

#[derive(Debug, Default, serde::Deserialize)]
pub struct EotConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub transform: Option<String>,
    pub rate: Option<f64>,
    pub port: Option<u16>,
    pub top_logprobs: Option<u8>,
    pub system_a: Option<String>,
}

impl EotConfig {
    /// Load config by merging `~/.eot.toml` (base) and `./.eot.toml` (local,
    /// higher priority).  Silently ignores missing files or parse errors.
    pub fn load() -> Self {
        let mut cfg = Self::default();

        if let Some(home) = home_dir() {
            cfg.merge(load_file(&home.join(".eot.toml")));
        }

        cfg.merge(load_file(&PathBuf::from(".eot.toml")));
        cfg
    }

    /// Overwrite fields in `self` with non-`None` values from `other`.
    fn merge(&mut self, other: EotConfig) {
        if other.provider.is_some() {
            self.provider = other.provider;
        }
        if other.model.is_some() {
            self.model = other.model;
        }
        if other.transform.is_some() {
            self.transform = other.transform;
        }
        if other.rate.is_some() {
            self.rate = other.rate;
        }
        if other.port.is_some() {
            self.port = other.port;
        }
        if other.top_logprobs.is_some() {
            self.top_logprobs = other.top_logprobs;
        }
        if other.system_a.is_some() {
            self.system_a = other.system_a;
        }
    }
}

fn load_file(path: &PathBuf) -> EotConfig {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return EotConfig::default(),
    };
    match toml::from_str::<EotConfig>(&text) {
        Ok(mut cfg) => {
            // Validate rate is in [0.0, 1.0]; warn and clamp if out of range.
            if let Some(r) = cfg.rate {
                if !(0.0..=1.0).contains(&r) {
                    eprintln!(
                        "[config] warning: rate={} in {} is out of range [0.0, 1.0]; clamping",
                        r,
                        path.display()
                    );
                    cfg.rate = Some(r.clamp(0.0, 1.0));
                }
            }
            cfg
        }
        Err(e) => {
            eprintln!(
                "[config] warning: failed to parse {}: {}",
                path.display(),
                e
            );
            EotConfig::default()
        }
    }
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| std::env::var("USERPROFILE").ok().map(PathBuf::from))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_all_none() {
        let cfg = EotConfig::default();
        assert!(cfg.provider.is_none());
        assert!(cfg.model.is_none());
        assert!(cfg.transform.is_none());
        assert!(cfg.rate.is_none());
        assert!(cfg.port.is_none());
        assert!(cfg.top_logprobs.is_none());
        assert!(cfg.system_a.is_none());
    }

    #[test]
    fn test_merge_overwrites_none_fields() {
        let mut base = EotConfig::default();
        let other = EotConfig {
            provider: Some("anthropic".to_string()),
            model: Some("claude-sonnet-4-6".to_string()),
            ..Default::default()
        };
        base.merge(other);
        assert_eq!(base.provider.as_deref(), Some("anthropic"));
        assert_eq!(base.model.as_deref(), Some("claude-sonnet-4-6"));
        assert!(base.rate.is_none());
    }

    #[test]
    fn test_merge_does_not_overwrite_with_none() {
        let mut base = EotConfig {
            provider: Some("openai".to_string()),
            rate: Some(0.3),
            ..Default::default()
        };
        let other = EotConfig::default();
        base.merge(other);
        assert_eq!(base.provider.as_deref(), Some("openai"));
        assert_eq!(base.rate, Some(0.3));
    }

    #[test]
    fn test_merge_local_wins_over_home() {
        let mut base = EotConfig {
            model: Some("gpt-3.5-turbo".to_string()),
            ..Default::default()
        };
        let local = EotConfig {
            model: Some("gpt-4".to_string()),
            ..Default::default()
        };
        base.merge(local);
        assert_eq!(base.model.as_deref(), Some("gpt-4"));
    }

    #[test]
    fn test_load_file_missing_returns_default() {
        let cfg = load_file(&PathBuf::from("/nonexistent/path/.eot.toml"));
        assert!(cfg.provider.is_none());
    }

    #[test]
    fn test_load_file_invalid_toml_returns_default() {
        let tmp = std::env::temp_dir().join("eot_test_bad.toml");
        std::fs::write(&tmp, "not valid { toml }}}").ok();
        let cfg = load_file(&tmp);
        std::fs::remove_file(&tmp).ok();
        assert!(cfg.provider.is_none());
    }

    #[test]
    fn test_load_file_valid_toml_parses_fields() {
        let tmp = std::env::temp_dir().join("eot_test_valid.toml");
        std::fs::write(&tmp, "provider = \"anthropic\"\nrate = 0.3\nport = 9000\n").ok();
        let cfg = load_file(&tmp);
        std::fs::remove_file(&tmp).ok();
        assert_eq!(cfg.provider.as_deref(), Some("anthropic"));
        assert_eq!(cfg.rate, Some(0.3));
        assert_eq!(cfg.port, Some(9000));
    }

    #[test]
    fn test_load_does_not_panic_without_config_files() {
        let cfg = EotConfig::load();
        let _ = cfg;
    }

    #[test]
    fn test_home_dir_no_panic() {
        let _ = home_dir();
    }

    #[test]
    fn test_merge_all_fields() {
        let mut base = EotConfig::default();
        let other = EotConfig {
            provider: Some("openai".to_string()),
            model: Some("gpt-4".to_string()),
            transform: Some("uppercase".to_string()),
            rate: Some(0.7),
            port: Some(9999),
            top_logprobs: Some(10),
            system_a: Some("Be concise.".to_string()),
        };
        base.merge(other);
        assert_eq!(base.provider.as_deref(), Some("openai"));
        assert_eq!(base.model.as_deref(), Some("gpt-4"));
        assert_eq!(base.transform.as_deref(), Some("uppercase"));
        assert_eq!(base.rate, Some(0.7));
        assert_eq!(base.port, Some(9999));
        assert_eq!(base.top_logprobs, Some(10));
        assert_eq!(base.system_a.as_deref(), Some("Be concise."));
    }

    // -- Rate validation tests (#15) --

    #[test]
    fn test_load_file_rate_too_high_clamped_to_one() {
        let tmp = std::env::temp_dir().join("eot_test_rate_high.toml");
        std::fs::write(&tmp, "rate = 5.0\n").ok();
        let cfg = load_file(&tmp);
        std::fs::remove_file(&tmp).ok();
        assert_eq!(cfg.rate, Some(1.0));
    }

    #[test]
    fn test_load_file_rate_negative_clamped_to_zero() {
        let tmp = std::env::temp_dir().join("eot_test_rate_neg.toml");
        std::fs::write(&tmp, "rate = -0.5\n").ok();
        let cfg = load_file(&tmp);
        std::fs::remove_file(&tmp).ok();
        assert_eq!(cfg.rate, Some(0.0));
    }

    #[test]
    fn test_load_file_rate_valid_unchanged() {
        let tmp = std::env::temp_dir().join("eot_test_rate_ok.toml");
        std::fs::write(&tmp, "rate = 0.4\n").ok();
        let cfg = load_file(&tmp);
        std::fs::remove_file(&tmp).ok();
        assert!((cfg.rate.unwrap() - 0.4).abs() < 1e-9);
    }
}
