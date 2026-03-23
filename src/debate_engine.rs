//! Multi-agent debate framework with position synthesis.

/// A position held by an agent in a debate.
#[derive(Debug, Clone)]
pub struct DebatePosition {
    pub agent_id: String,
    pub stance: String,
    pub argument: String,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

/// One round of a debate containing all agent positions and a moderator summary.
#[derive(Debug, Clone)]
pub struct DebateRound {
    pub round_number: u32,
    pub positions: Vec<DebatePosition>,
    pub moderator_summary: String,
}

/// The final outcome of a debate.
#[derive(Debug, Clone)]
pub struct DebateOutcome {
    pub consensus_reached: bool,
    pub winning_stance: Option<String>,
    pub confidence: f64,
    pub synthesis: String,
    pub dissent: Vec<String>,
}

/// A topic for a debate.
#[derive(Debug, Clone)]
pub struct DebateTopic {
    pub question: String,
    pub context: String,
    pub max_rounds: u32,
    pub consensus_threshold: f64,
}

/// Classifies text into stances based on keyword matching.
pub struct StanceClassifier {
    pub stances: Vec<String>,
}

impl StanceClassifier {
    pub fn new() -> Self {
        Self {
            stances: vec![
                "support".to_string(),
                "oppose".to_string(),
                "neutral".to_string(),
            ],
        }
    }

    pub fn classify(text: &str, _topic: &str) -> String {
        let lower = text.to_lowercase();
        let support_keywords = ["agree", "support", "yes", "benefit"];
        let oppose_keywords = ["disagree", "oppose", "no", "harm"];

        for kw in &support_keywords {
            if lower.contains(kw) {
                return "support".to_string();
            }
        }
        for kw in &oppose_keywords {
            if lower.contains(kw) {
                return "oppose".to_string();
            }
        }
        "neutral".to_string()
    }
}

impl Default for StanceClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Measures argument strength based on clarity, evidence, and length.
#[derive(Debug, Clone)]
pub struct ArgumentStrength {
    pub clarity: f64,
    pub evidence_count: usize,
    pub word_count: usize,
    pub strength: f64,
}

/// Evaluates the strength of an argument.
pub fn evaluate_argument(arg: &str, evidence: &[String]) -> ArgumentStrength {
    let words: Vec<&str> = arg.split_whitespace().collect();
    let word_count = words.len();
    let hedge_words = ["maybe", "perhaps", "might", "possibly"];
    let hedge_count = words
        .iter()
        .filter(|w| hedge_words.contains(&w.to_lowercase().as_str()))
        .count();

    let clarity = if word_count == 0 {
        1.0
    } else {
        1.0 - (hedge_count as f64 / word_count as f64)
    };

    let evidence_count = evidence.len();
    let strength = clarity * 0.4
        + (evidence_count.min(5) as f64 / 5.0) * 0.4
        + (word_count.min(100) as f64 / 100.0) * 0.2;

    ArgumentStrength {
        clarity,
        evidence_count,
        word_count,
        strength,
    }
}

/// The main debate engine that manages rounds and synthesizes outcomes.
pub struct DebateEngine {
    pub topic: DebateTopic,
    pub rounds: Vec<DebateRound>,
}

impl DebateEngine {
    pub fn new(topic: DebateTopic) -> Self {
        Self {
            topic,
            rounds: Vec::new(),
        }
    }

    pub fn add_position(&mut self, round: u32, position: DebatePosition) {
        if let Some(r) = self.rounds.iter_mut().find(|r| r.round_number == round) {
            r.positions.push(position);
        } else {
            let summary = String::new();
            self.rounds.push(DebateRound {
                round_number: round,
                positions: vec![position],
                moderator_summary: summary,
            });
        }
    }

    pub fn summarize_round(&self, round: u32) -> String {
        let r = match self.rounds.iter().find(|r| r.round_number == round) {
            Some(r) => r,
            None => return format!("Round {} not found.", round),
        };

        let mut stance_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for pos in &r.positions {
            *stance_counts.entry(pos.stance.as_str()).or_insert(0) += 1;
        }

        let top_arg = r
            .positions
            .iter()
            .max_by(|a, b| {
                let sa = evaluate_argument(&a.argument, &a.evidence).strength;
                let sb = evaluate_argument(&b.argument, &b.evidence).strength;
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.argument.as_str())
            .unwrap_or("No arguments");

        let counts_str: Vec<String> = stance_counts
            .iter()
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect();

        format!(
            "Round {}: [{}] | Top argument: {}",
            round,
            counts_str.join(", "),
            top_arg
        )
    }

    pub fn check_consensus(&self) -> Option<String> {
        let all_positions: Vec<&DebatePosition> =
            self.rounds.iter().flat_map(|r| r.positions.iter()).collect();

        let total = all_positions.len();
        if total == 0 {
            return None;
        }

        let mut stance_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for pos in &all_positions {
            *stance_counts.entry(pos.stance.as_str()).or_insert(0) += 1;
        }

        for (stance, count) in &stance_counts {
            if (*count as f64 / total as f64) > self.topic.consensus_threshold {
                return Some(stance.to_string());
            }
        }
        None
    }

    pub fn synthesize_outcome(&self) -> DebateOutcome {
        let consensus_stance = self.check_consensus();
        let consensus_reached = consensus_stance.is_some();

        let all_positions: Vec<&DebatePosition> =
            self.rounds.iter().flat_map(|r| r.positions.iter()).collect();

        let strongest = all_positions.iter().max_by(|a, b| {
            let sa = evaluate_argument(&a.argument, &a.evidence).strength;
            let sb = evaluate_argument(&b.argument, &b.evidence).strength;
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let winning_stance = consensus_stance.clone().or_else(|| {
            strongest.map(|p| p.stance.clone())
        });

        let avg_confidence = if all_positions.is_empty() {
            0.0
        } else {
            all_positions.iter().map(|p| p.confidence).sum::<f64>() / all_positions.len() as f64
        };

        let strongest_arg = strongest
            .map(|p| p.argument.as_str())
            .unwrap_or("No arguments provided");

        let synthesis = format!(
            "Topic: {}. Majority stance: {}. Strongest argument: {}",
            self.topic.question,
            winning_stance.as_deref().unwrap_or("none"),
            strongest_arg
        );

        let winning = winning_stance.as_deref().unwrap_or("");
        let dissent: Vec<String> = all_positions
            .iter()
            .filter(|p| p.stance != winning)
            .map(|p| format!("[{}] {}: {}", p.agent_id, p.stance, p.argument))
            .collect();

        DebateOutcome {
            consensus_reached,
            winning_stance,
            confidence: avg_confidence,
            synthesis,
            dissent,
        }
    }
}

/// Fluent builder for constructing `DebateTopic` instances.
pub struct DebateBuilder {
    question: String,
    context: String,
    max_rounds: u32,
    threshold: f64,
}

impl DebateBuilder {
    pub fn new(question: impl Into<String>, context: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            context: context.into(),
            max_rounds: 3,
            threshold: 0.6,
        }
    }

    pub fn max_rounds(mut self, n: u32) -> Self {
        self.max_rounds = n;
        self
    }

    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    pub fn build(self) -> DebateTopic {
        DebateTopic {
            question: self.question,
            context: self.context,
            max_rounds: self.max_rounds,
            consensus_threshold: self.threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_support() {
        assert_eq!(StanceClassifier::classify("I agree with this", "topic"), "support");
        assert_eq!(StanceClassifier::classify("This will benefit everyone", "topic"), "support");
    }

    #[test]
    fn classify_oppose() {
        assert_eq!(StanceClassifier::classify("I disagree strongly", "topic"), "oppose");
        assert_eq!(StanceClassifier::classify("This will harm the community", "topic"), "oppose");
    }

    #[test]
    fn classify_neutral() {
        assert_eq!(StanceClassifier::classify("This is interesting", "topic"), "neutral");
    }

    #[test]
    fn evaluate_argument_scores() {
        let evidence = vec!["Source A".to_string(), "Source B".to_string()];
        let result = evaluate_argument("This policy will clearly reduce costs significantly.", &evidence);
        assert!(result.clarity > 0.0);
        assert_eq!(result.evidence_count, 2);
        assert!(result.strength > 0.0 && result.strength <= 1.0);
    }

    #[test]
    fn evaluate_argument_hedges_reduce_clarity() {
        let no_hedge = evaluate_argument("This is definitely true.", &[]);
        let with_hedge = evaluate_argument("This might perhaps be true possibly.", &[]);
        assert!(no_hedge.clarity > with_hedge.clarity);
    }

    #[test]
    fn consensus_detection_at_threshold() {
        let topic = DebateBuilder::new("Test question", "context")
            .threshold(0.6)
            .build();
        let mut engine = DebateEngine::new(topic);

        for i in 0..7 {
            engine.add_position(1, DebatePosition {
                agent_id: format!("agent_{}", i),
                stance: "support".to_string(),
                argument: "Strong argument".to_string(),
                confidence: 0.8,
                evidence: vec![],
            });
        }
        for i in 0..3 {
            engine.add_position(1, DebatePosition {
                agent_id: format!("opponent_{}", i),
                stance: "oppose".to_string(),
                argument: "Counter argument".to_string(),
                confidence: 0.5,
                evidence: vec![],
            });
        }

        assert_eq!(engine.check_consensus(), Some("support".to_string()));
    }

    #[test]
    fn synthesis_includes_dissent() {
        let topic = DebateBuilder::new("Test", "ctx").threshold(0.9).build();
        let mut engine = DebateEngine::new(topic);

        engine.add_position(1, DebatePosition {
            agent_id: "a1".to_string(),
            stance: "support".to_string(),
            argument: "Pro argument".to_string(),
            confidence: 0.8,
            evidence: vec![],
        });
        engine.add_position(1, DebatePosition {
            agent_id: "a2".to_string(),
            stance: "oppose".to_string(),
            argument: "Con argument".to_string(),
            confidence: 0.6,
            evidence: vec![],
        });

        let outcome = engine.synthesize_outcome();
        assert!(!outcome.dissent.is_empty());
    }

    #[test]
    fn round_summary_contains_stance_counts() {
        let topic = DebateBuilder::new("Q", "ctx").build();
        let mut engine = DebateEngine::new(topic);
        engine.add_position(1, DebatePosition {
            agent_id: "a".to_string(),
            stance: "support".to_string(),
            argument: "Good point".to_string(),
            confidence: 0.7,
            evidence: vec![],
        });
        let summary = engine.summarize_round(1);
        assert!(summary.contains("support"));
    }
}
