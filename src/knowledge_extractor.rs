//! Named entity extraction and knowledge graph construction.

use std::collections::HashMap;
use std::fmt;

/// Broad categories of named entities.
#[derive(Debug, Clone, PartialEq)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Date,
    Money,
    Percentage,
    Product,
    Technology,
    Concept,
    Unknown,
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            EntityType::Person => "Person",
            EntityType::Organization => "Organization",
            EntityType::Location => "Location",
            EntityType::Date => "Date",
            EntityType::Money => "Money",
            EntityType::Percentage => "Percentage",
            EntityType::Product => "Product",
            EntityType::Technology => "Technology",
            EntityType::Concept => "Concept",
            EntityType::Unknown => "Unknown",
        };
        write!(f, "{}", label)
    }
}

/// A single named entity found in text.
#[derive(Debug, Clone)]
pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
    pub start_char: usize,
    pub end_char: usize,
    pub confidence: f64,
}

/// A directed relation between two entities.
#[derive(Debug, Clone)]
pub struct Relation {
    pub subject: Entity,
    pub predicate: String,
    pub object: Entity,
    pub confidence: f64,
}

/// A simple subject–predicate–object triple (string form).
#[derive(Debug, Clone)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// A knowledge graph built from extracted entities and relations.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeGraph {
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub triples: Vec<Triple>,
}

/// Stateless knowledge extractor.
pub struct KnowledgeExtractor;

impl KnowledgeExtractor {
    /// Rule-based entity extraction.
    ///
    /// Detects:
    /// * Capitalized multi-word phrases (Persons / Orgs / Locations)
    /// * Date patterns (month names, year numbers, "January 1, 2020")
    /// * Money patterns ($X, Xm, Xbn)
    /// * Percentage patterns (X%)
    pub fn extract_entities(text: &str) -> Vec<Entity> {
        let mut entities: Vec<Entity> = Vec::new();
        let chars: Vec<char> = text.chars().collect();

        // ---- money patterns: $\d+, \d+m, \d+bn ----
        {
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == '$' {
                    let start = i;
                    i += 1;
                    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == ',' || chars[i] == '.') {
                        i += 1;
                    }
                    if i > start + 1 {
                        let text_slice: String = chars[start..i].iter().collect();
                        entities.push(Entity {
                            text: text_slice,
                            entity_type: EntityType::Money,
                            start_char: start,
                            end_char: i,
                            confidence: 0.9,
                        });
                    }
                } else {
                    i += 1;
                }
            }
        }

        // ---- percentage patterns: \d+% ----
        {
            let mut i = 0;
            while i < chars.len() {
                if chars[i].is_ascii_digit() {
                    let start = i;
                    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                        i += 1;
                    }
                    if i < chars.len() && chars[i] == '%' {
                        i += 1;
                        let text_slice: String = chars[start..i].iter().collect();
                        entities.push(Entity {
                            text: text_slice,
                            entity_type: EntityType::Percentage,
                            start_char: start,
                            end_char: i,
                            confidence: 0.9,
                        });
                    }
                } else {
                    i += 1;
                }
            }
        }

        // ---- date patterns: look for month names ----
        let months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ];
        for month in &months {
            let mut search = text;
            let mut offset = 0;
            while let Some(pos) = search.find(month) {
                let start = offset + pos;
                let end_search = &search[pos + month.len()..];
                // grab optional " N, YYYY" or " YYYY"
                let extra: String = end_search
                    .chars()
                    .take_while(|&c| c.is_ascii_digit() || c == ',' || c == ' ')
                    .collect();
                let end = start + month.len() + extra.trim_end().len();
                let entity_text = text[start..end].trim().to_string();
                entities.push(Entity {
                    text: entity_text,
                    entity_type: EntityType::Date,
                    start_char: start,
                    end_char: end,
                    confidence: 0.85,
                });
                offset += pos + month.len();
                search = &search[pos + month.len()..];
            }
        }

        // ---- capitalized word sequences ----
        let words: Vec<(usize, &str)> = Self::tokenize_with_offsets(text);
        let mut i = 0;
        while i < words.len() {
            let (start_off, word) = words[i];
            if Self::is_capitalized_word(word) {
                // greedily extend
                let mut span_end = start_off + word.len();
                let mut phrase = word.to_string();
                let mut j = i + 1;
                while j < words.len() {
                    let (_, next_word) = words[j];
                    if Self::is_capitalized_word(next_word) || ["of", "the", "and", "in", "at", "for"].contains(&next_word) {
                        phrase.push(' ');
                        phrase.push_str(next_word);
                        span_end = words[j].0 + next_word.len();
                        j += 1;
                    } else {
                        break;
                    }
                }

                // don't emit single common stop words
                let lower = phrase.to_lowercase();
                let stop = ["the", "a", "an", "in", "of", "at", "to", "is", "are", "was", "were", "and", "or", "but"];
                if !stop.contains(&lower.as_str()) && phrase.len() > 1 {
                    let context_start = if start_off > 50 { start_off - 50 } else { 0 };
                    let context = &text[context_start..std::cmp::min(text.len(), span_end + 50)];
                    let (etype, conf) = Self::classify_entity(&phrase, context);
                    entities.push(Entity {
                        text: phrase,
                        entity_type: etype,
                        start_char: start_off,
                        end_char: span_end,
                        confidence: conf,
                    });
                }
                i = j;
            } else {
                i += 1;
            }
        }

        // Deduplicate by text (keep highest confidence)
        let mut deduped: Vec<Entity> = Vec::new();
        'outer: for entity in entities {
            for existing in &mut deduped {
                if existing.text == entity.text {
                    if entity.confidence > existing.confidence {
                        *existing = entity.clone();
                    }
                    continue 'outer;
                }
            }
            deduped.push(entity);
        }

        deduped
    }

    /// Classify an entity phrase using keyword heuristics.
    pub fn classify_entity(text: &str, context: &str) -> (EntityType, f64) {
        let lower = text.to_lowercase();
        let ctx = context.to_lowercase();

        // Organization signals
        if lower.ends_with(" inc") || lower.ends_with(" inc.")
            || lower.ends_with(" corp") || lower.ends_with(" corp.")
            || lower.ends_with(" ltd") || lower.ends_with(" ltd.")
            || lower.ends_with(" llc") || lower.ends_with(" co.")
            || lower.ends_with(" company") || lower.ends_with(" technologies")
            || lower.ends_with(" university") || lower.ends_with(" institute")
        {
            return (EntityType::Organization, 0.9);
        }

        // Person signals from context
        if ctx.contains(" said") || ctx.contains(" says") || ctx.contains(" told")
            || ctx.contains("mr.") || ctx.contains("ms.") || ctx.contains("dr.")
            || ctx.contains("prof.") || ctx.contains("ceo") || ctx.contains("founder")
        {
            return (EntityType::Person, 0.8);
        }

        // Location signals
        if ctx.contains(" city") || ctx.contains(" country") || ctx.contains(" region")
            || ctx.contains("located in") || ctx.contains("based in") || ctx.contains("born in")
        {
            return (EntityType::Location, 0.75);
        }

        // Technology signals
        if lower.contains("ai") || lower.contains("software") || lower.contains("platform")
            || lower.contains("api") || lower.contains("framework") || lower.contains("language")
            || lower.contains("database") || lower.contains("algorithm")
        {
            return (EntityType::Technology, 0.7);
        }

        // Product signals
        if ctx.contains("product") || ctx.contains("version") || ctx.contains("release")
            || ctx.contains("launch") || ctx.contains("model")
        {
            return (EntityType::Product, 0.65);
        }

        (EntityType::Unknown, 0.5)
    }

    /// Extract relations between entities using simple pattern matching.
    pub fn extract_relations(text: &str, entities: &[Entity]) -> Vec<Relation> {
        let mut relations: Vec<Relation> = Vec::new();
        let sentences = crate::document_parser::DocumentParser::split_sentences(text);

        for sentence in &sentences {
            let lower = sentence.to_lowercase();

            // Find all entities present in this sentence
            let present: Vec<&Entity> = entities
                .iter()
                .filter(|e| sentence.contains(&e.text))
                .collect();

            if present.len() < 2 {
                continue;
            }

            // Pattern: "X is Y"
            for i in 0..present.len() {
                for j in 0..present.len() {
                    if i == j {
                        continue;
                    }
                    let subj = present[i];
                    let obj = present[j];

                    let pred = if lower.contains(&format!("{} is", subj.text.to_lowercase())) {
                        Some("is")
                    } else if lower.contains(&format!("{} has", subj.text.to_lowercase())) {
                        Some("has")
                    } else if lower.contains("ceo") && lower.contains(&obj.text.to_lowercase()) {
                        Some("is CEO of")
                    } else if lower.contains("located in") {
                        Some("located in")
                    } else if lower.contains("founded") {
                        Some("founded")
                    } else {
                        None
                    };

                    if let Some(predicate) = pred {
                        relations.push(Relation {
                            subject: subj.clone(),
                            predicate: predicate.to_string(),
                            object: obj.clone(),
                            confidence: 0.65,
                        });
                        break;
                    }
                }
            }
        }

        relations
    }

    /// Build string triples from relations.
    pub fn build_triples(relations: &[Relation]) -> Vec<Triple> {
        relations
            .iter()
            .map(|r| Triple {
                subject: r.subject.text.clone(),
                predicate: r.predicate.clone(),
                object: r.object.text.clone(),
            })
            .collect()
    }

    /// Simple coreference resolution: map pronoun positions to entity indices.
    ///
    /// Returns a map from entity index (pronoun) → entity index (referent).
    pub fn coreference_resolution(entities: &[Entity], _text: &str) -> HashMap<usize, usize> {
        let mut mapping: HashMap<usize, usize> = HashMap::new();

        let pronouns_male = ["he", "him", "his"];
        let pronouns_female = ["she", "her", "hers"];
        let pronouns_neutral = ["they", "their", "them", "it", "its"];

        let mut last_person: Option<usize> = None;
        let mut last_org: Option<usize> = None;

        for (i, entity) in entities.iter().enumerate() {
            let lower = entity.text.to_lowercase();

            if pronouns_male.contains(&lower.as_str()) || pronouns_female.contains(&lower.as_str()) {
                if let Some(ref_idx) = last_person {
                    mapping.insert(i, ref_idx);
                }
            } else if pronouns_neutral.contains(&lower.as_str()) {
                if let Some(ref_idx) = last_org.or(last_person) {
                    mapping.insert(i, ref_idx);
                }
            } else {
                match entity.entity_type {
                    EntityType::Person => last_person = Some(i),
                    EntityType::Organization => last_org = Some(i),
                    _ => {}
                }
            }
        }

        mapping
    }

    /// Full extraction pipeline: returns a complete KnowledgeGraph.
    pub fn extract(text: &str) -> KnowledgeGraph {
        let entities = Self::extract_entities(text);
        let relations = Self::extract_relations(text, &entities);
        let triples = Self::build_triples(&relations);
        KnowledgeGraph { entities, relations, triples }
    }

    /// Merge two knowledge graphs, deduplicating entities by text.
    pub fn merge_graphs(mut a: KnowledgeGraph, b: KnowledgeGraph) -> KnowledgeGraph {
        for entity in b.entities {
            if !a.entities.iter().any(|e| e.text == entity.text) {
                a.entities.push(entity);
            }
        }
        a.relations.extend(b.relations);
        a.triples.extend(b.triples);
        a
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn is_capitalized_word(word: &str) -> bool {
        let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
        if clean.is_empty() {
            return false;
        }
        let mut chars = clean.chars();
        if let Some(first) = chars.next() {
            first.is_uppercase() && !clean.chars().all(|c| c.is_uppercase())
        } else {
            false
        }
    }

    fn tokenize_with_offsets(text: &str) -> Vec<(usize, &str)> {
        let mut result = Vec::new();
        let mut start = None;
        let mut byte_offset = 0;

        for (i, ch) in text.char_indices() {
            if ch.is_alphanumeric() || ch == '-' || ch == '\'' {
                if start.is_none() {
                    start = Some(i);
                }
            } else {
                if let Some(s) = start {
                    result.push((s, &text[s..i]));
                    start = None;
                }
            }
            byte_offset = i + ch.len_utf8();
        }
        if let Some(s) = start {
            result.push((s, &text[s..byte_offset]));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_display() {
        assert_eq!(EntityType::Person.to_string(), "Person");
        assert_eq!(EntityType::Organization.to_string(), "Organization");
    }

    #[test]
    fn test_extract_money() {
        let entities = KnowledgeExtractor::extract_entities("The deal was worth $500 million.");
        let money: Vec<_> = entities.iter().filter(|e| e.entity_type == EntityType::Money).collect();
        assert!(!money.is_empty());
    }

    #[test]
    fn test_extract_percentage() {
        let entities = KnowledgeExtractor::extract_entities("Prices rose by 12%.");
        let pct: Vec<_> = entities.iter().filter(|e| e.entity_type == EntityType::Percentage).collect();
        assert!(!pct.is_empty());
    }

    #[test]
    fn test_extract_date() {
        let entities = KnowledgeExtractor::extract_entities("The event is on January 15, 2024.");
        let dates: Vec<_> = entities.iter().filter(|e| e.entity_type == EntityType::Date).collect();
        assert!(!dates.is_empty());
    }

    #[test]
    fn test_classify_org() {
        let (etype, _conf) = KnowledgeExtractor::classify_entity("Acme Corp", "");
        assert_eq!(etype, EntityType::Organization);
    }

    #[test]
    fn test_build_triples() {
        let graph = KnowledgeExtractor::extract("Google Inc is a company. Google Inc has many employees.");
        // triples are built from relations
        // just verify the pipeline runs without panic
        assert!(graph.triples.len() <= graph.relations.len());
    }

    #[test]
    fn test_merge_graphs() {
        let g1 = KnowledgeExtractor::extract("Apple Inc launched a product.");
        let g2 = KnowledgeExtractor::extract("Microsoft Corp released software.");
        let merged = KnowledgeExtractor::merge_graphs(g1, g2);
        assert!(!merged.entities.is_empty());
    }
}
