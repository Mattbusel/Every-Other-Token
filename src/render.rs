//! Terminal rendering helpers extracted from the core `TokenInterceptor`.
//!
//! This module owns all the "how does a token look in the terminal?" logic so
//! that `lib.rs` stays focused on streaming mechanics.  The web UI rendering
//! lives in the JavaScript in `src/ui/app.js`.

use colored::*;

use crate::providers::OpenAITopLogprob;
use crate::TokenEvent;

// ---------------------------------------------------------------------------
// Confidence band classification
// ---------------------------------------------------------------------------

/// Three-band classification of a model confidence value (0.0–1.0).
///
/// Used to pick terminal colours and emoji indicators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceBand {
    High,
    Mid,
    Low,
}

impl ConfidenceBand {
    /// Classify a raw confidence value in `[0.0, 1.0]`.
    ///
    /// # Examples
    /// ```
    /// use every_other_token::render::ConfidenceBand;
    /// assert_eq!(ConfidenceBand::from_confidence(0.9), ConfidenceBand::High);
    /// assert_eq!(ConfidenceBand::from_confidence(0.5), ConfidenceBand::Mid);
    /// assert_eq!(ConfidenceBand::from_confidence(0.1), ConfidenceBand::Low);
    /// ```
    pub fn from_confidence(c: f32) -> Self {
        if c >= 0.7 {
            ConfidenceBand::High
        } else if c >= 0.4 {
            ConfidenceBand::Mid
        } else {
            ConfidenceBand::Low
        }
    }

    /// Short emoji indicator, used in visual-mode output.
    pub fn indicator(self) -> &'static str {
        match self {
            ConfidenceBand::High => "●",
            ConfidenceBand::Mid  => "◑",
            ConfidenceBand::Low  => "○",
        }
    }
}

// ---------------------------------------------------------------------------
// Importance → heat level
// ---------------------------------------------------------------------------

/// Map a token importance score (0.0–1.0) to a discrete heat level 0–4.
///
/// Level 4 = hottest (most important), 0 = cold.
///
/// # Examples
/// ```
/// use every_other_token::render::importance_to_heat;
/// assert_eq!(importance_to_heat(1.0), 4);
/// assert_eq!(importance_to_heat(0.5), 2);
/// assert_eq!(importance_to_heat(0.0), 0);
/// ```
#[inline]
pub fn importance_to_heat(importance: f64) -> u8 {
    if importance >= 0.8 { 4 }
    else if importance >= 0.6 { 3 }
    else if importance >= 0.4 { 2 }
    else if importance >= 0.2 { 1 }
    else { 0 }
}

// ---------------------------------------------------------------------------
// Terminal colour helpers
// ---------------------------------------------------------------------------

/// Apply heatmap terminal colouring to a token string.
///
/// Returns an ANSI-coloured string based on the token's importance score.
pub fn heat_colorize(text: &str, importance: f64) -> ColoredString {
    match importance_to_heat(importance) {
        4 => text.on_red(),
        3 => text.red(),
        2 => text.yellow(),
        1 => text.blue(),
        _ => text.dimmed(),
    }
}

// ---------------------------------------------------------------------------
// Visual-mode line formatting
// ---------------------------------------------------------------------------

/// Format a single `TokenEvent` for visual-mode terminal output.
///
/// Includes confidence indicator and top-K alternatives when present.
pub fn format_visual_token(event: &TokenEvent, alternatives: &[OpenAITopLogprob]) -> String {
    let conf_str = match event.confidence {
        Some(c) => {
            let band = ConfidenceBand::from_confidence(c);
            format!(" {}({:.0}%)", band.indicator(), c * 100.0)
        }
        None => String::new(),
    };

    let perp_str = match event.perplexity {
        Some(p) if p > 5.0 => format!(" ⚡{:.1}", p),
        _ => String::new(),
    };

    let alts_str = if !alternatives.is_empty() {
        let top3: Vec<String> = alternatives
            .iter()
            .take(3)
            .map(|a| format!("{} ({:.0}%)", a.token, (a.logprob.exp() * 100.0)))
            .collect();
        format!(" [{}]", top3.join(", "))
    } else {
        String::new()
    };

    format!("{}{}{}{}", event.text, conf_str, perp_str, alts_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenAlternative;

    // ---- ConfidenceBand ----

    #[test]
    fn test_confidence_band_high() {
        assert_eq!(ConfidenceBand::from_confidence(0.9), ConfidenceBand::High);
        assert_eq!(ConfidenceBand::from_confidence(0.7), ConfidenceBand::High);
    }

    #[test]
    fn test_confidence_band_mid() {
        assert_eq!(ConfidenceBand::from_confidence(0.5), ConfidenceBand::Mid);
        assert_eq!(ConfidenceBand::from_confidence(0.4), ConfidenceBand::Mid);
    }

    #[test]
    fn test_confidence_band_low() {
        assert_eq!(ConfidenceBand::from_confidence(0.0), ConfidenceBand::Low);
        assert_eq!(ConfidenceBand::from_confidence(0.39), ConfidenceBand::Low);
    }

    #[test]
    fn test_confidence_band_boundary_exact_07() {
        assert_eq!(ConfidenceBand::from_confidence(0.7), ConfidenceBand::High);
    }

    #[test]
    fn test_confidence_band_boundary_exact_04() {
        assert_eq!(ConfidenceBand::from_confidence(0.4), ConfidenceBand::Mid);
    }

    #[test]
    fn test_confidence_band_indicators_distinct() {
        let indicators: Vec<&str> = [
            ConfidenceBand::High,
            ConfidenceBand::Mid,
            ConfidenceBand::Low,
        ]
        .iter()
        .map(|b| b.indicator())
        .collect();
        let unique: std::collections::HashSet<&&str> = indicators.iter().collect();
        assert_eq!(unique.len(), 3, "each band should have a unique indicator");
    }

    // ---- importance_to_heat ----

    #[test]
    fn test_heat_level_4() {
        assert_eq!(importance_to_heat(1.0), 4);
        assert_eq!(importance_to_heat(0.8), 4);
    }

    #[test]
    fn test_heat_level_3() {
        assert_eq!(importance_to_heat(0.7), 3);
        assert_eq!(importance_to_heat(0.6), 3);
    }

    #[test]
    fn test_heat_level_2() {
        assert_eq!(importance_to_heat(0.5), 2);
        assert_eq!(importance_to_heat(0.4), 2);
    }

    #[test]
    fn test_heat_level_1() {
        assert_eq!(importance_to_heat(0.3), 1);
        assert_eq!(importance_to_heat(0.2), 1);
    }

    #[test]
    fn test_heat_level_0() {
        assert_eq!(importance_to_heat(0.0), 0);
        assert_eq!(importance_to_heat(0.19), 0);
    }

    // ---- format_visual_token ----

    fn make_event(text: &str, confidence: Option<f32>, perplexity: Option<f32>) -> TokenEvent {
        TokenEvent {
            text: text.to_string(),
            original: text.to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: None,
            confidence,
            perplexity,
            alternatives: vec![],
            is_error: false,
        }
    }

    #[test]
    fn test_format_visual_no_extras() {
        let ev = make_event("hello", None, None);
        let out = format_visual_token(&ev, &[]);
        assert_eq!(out, "hello");
    }

    #[test]
    fn test_format_visual_with_confidence() {
        let ev = make_event("world", Some(0.85), None);
        let out = format_visual_token(&ev, &[]);
        assert!(out.contains("world"));
        assert!(out.contains("85%"));
        assert!(out.contains("●")); // high indicator
    }

    #[test]
    fn test_format_visual_high_perplexity_shown() {
        let ev = make_event("odd", None, Some(10.0));
        let out = format_visual_token(&ev, &[]);
        assert!(out.contains("⚡"));
        assert!(out.contains("10.0"));
    }

    #[test]
    fn test_format_visual_low_perplexity_hidden() {
        let ev = make_event("normal", None, Some(1.5));
        let out = format_visual_token(&ev, &[]);
        assert!(!out.contains("⚡"));
    }

    #[test]
    fn test_format_visual_with_alternatives() {
        let alts = vec![
            OpenAITopLogprob { token: "foo".to_string(), logprob: 0.0 },
            OpenAITopLogprob { token: "bar".to_string(), logprob: -1.0 },
        ];
        let ev = make_event("baz", None, None);
        let out = format_visual_token(&ev, &alts);
        assert!(out.contains("foo"));
        assert!(out.contains("bar"));
    }

    #[test]
    fn test_heat_colorize_returns_text() {
        // Just verify it runs without panic and the underlying text is preserved
        let s = heat_colorize("token", 0.9);
        assert!(s.to_string().contains("token"));
    }

    // ---- TokenAlternative round-trip (sanity check) ----

    #[test]
    fn test_token_alternative_fields() {
        let alt = TokenAlternative { token: "hello".to_string(), probability: 0.42 };
        assert_eq!(alt.token, "hello");
        assert!((alt.probability - 0.42).abs() < 1e-6);
    }
}
