//! Reusable prompt template library with categorization and search.

use std::collections::HashMap;

/// Category for a prompt template.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PromptCategory {
    Coding,
    Writing,
    Analysis,
    Summarization,
    QA,
    Translation,
    Creative,
    System,
}

impl PromptCategory {
    pub fn label(&self) -> &str {
        match self {
            PromptCategory::Coding => "Coding",
            PromptCategory::Writing => "Writing",
            PromptCategory::Analysis => "Analysis",
            PromptCategory::Summarization => "Summarization",
            PromptCategory::QA => "Q&A",
            PromptCategory::Translation => "Translation",
            PromptCategory::Creative => "Creative",
            PromptCategory::System => "System",
        }
    }
}

/// A single reusable prompt template.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    pub id: String,
    pub name: String,
    pub category: PromptCategory,
    pub template: String,
    pub variables: Vec<String>,
    pub tags: Vec<String>,
    pub usage_count: u32,
    pub avg_quality: f64,
}

/// Library that stores, indexes, and retrieves prompt templates.
#[derive(Debug, Default)]
pub struct PromptLibrary {
    templates: HashMap<String, PromptTemplate>,
}

impl PromptLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a template to the library (overwrites if same id).
    pub fn add(&mut self, template: PromptTemplate) {
        self.templates.insert(template.id.clone(), template);
    }

    /// Retrieve a template by id.
    pub fn get(&self, id: &str) -> Option<&PromptTemplate> {
        self.templates.get(id)
    }

    /// Search templates whose name, tags, or template text contain `query`
    /// (case-insensitive).
    pub fn search_by_keyword(&self, query: &str) -> Vec<&PromptTemplate> {
        let q = query.to_lowercase();
        let mut results: Vec<&PromptTemplate> = self
            .templates
            .values()
            .filter(|t| {
                t.name.to_lowercase().contains(&q)
                    || t.template.to_lowercase().contains(&q)
                    || t.tags.iter().any(|tag| tag.to_lowercase().contains(&q))
            })
            .collect();
        results.sort_by(|a, b| a.id.cmp(&b.id));
        results
    }

    /// Return all templates in the given category.
    pub fn search_by_category(&self, cat: PromptCategory) -> Vec<&PromptTemplate> {
        let mut results: Vec<&PromptTemplate> = self
            .templates
            .values()
            .filter(|t| t.category == cat)
            .collect();
        results.sort_by(|a, b| a.id.cmp(&b.id));
        results
    }

    /// Return the top `n` templates by usage count (descending).
    pub fn popular(&self, n: usize) -> Vec<&PromptTemplate> {
        let mut v: Vec<&PromptTemplate> = self.templates.values().collect();
        v.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
        v.truncate(n);
        v
    }

    /// Return the top `n` templates by average quality score (descending).
    pub fn best_rated(&self, n: usize) -> Vec<&PromptTemplate> {
        let mut v: Vec<&PromptTemplate> = self.templates.values().collect();
        v.sort_by(|a, b| b.avg_quality.partial_cmp(&a.avg_quality).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(n);
        v
    }

    /// Record a usage event for `id`, updating usage_count and a running
    /// average of quality scores.
    pub fn record_usage(&mut self, id: &str, quality_score: f64) {
        if let Some(t) = self.templates.get_mut(id) {
            let n = t.usage_count as f64;
            t.avg_quality = (t.avg_quality * n + quality_score) / (n + 1.0);
            t.usage_count += 1;
        }
    }

    /// Return templates that share the same category *and* have at least one
    /// overlapping tag with the template identified by `id`.
    pub fn similar_templates(&self, id: &str) -> Vec<&PromptTemplate> {
        let base = match self.templates.get(id) {
            Some(t) => t,
            None => return vec![],
        };
        let base_cat = base.category.clone();
        let base_tags: std::collections::HashSet<&String> = base.tags.iter().collect();

        let mut results: Vec<&PromptTemplate> = self
            .templates
            .values()
            .filter(|t| {
                t.id != id
                    && t.category == base_cat
                    && t.tags.iter().any(|tag| base_tags.contains(tag))
            })
            .collect();
        results.sort_by(|a, b| a.id.cmp(&b.id));
        results
    }
}

/// Build a library pre-populated with five example templates.
pub fn default_library() -> PromptLibrary {
    let mut lib = PromptLibrary::new();

    lib.add(PromptTemplate {
        id: "code-review".to_string(),
        name: "Code Review".to_string(),
        category: PromptCategory::Coding,
        template: "Review the following {{language}} code for bugs, style issues, and improvements:\n\n```{{language}}\n{{code}}\n```".to_string(),
        variables: vec!["language".to_string(), "code".to_string()],
        tags: vec!["review".to_string(), "quality".to_string(), "coding".to_string()],
        usage_count: 0,
        avg_quality: 0.0,
    });

    lib.add(PromptTemplate {
        id: "summarize-article".to_string(),
        name: "Summarize Article".to_string(),
        category: PromptCategory::Summarization,
        template: "Summarize the following article in {{num_sentences}} sentences, focusing on the key points:\n\n{{article}}".to_string(),
        variables: vec!["num_sentences".to_string(), "article".to_string()],
        tags: vec!["summary".to_string(), "article".to_string(), "condensed".to_string()],
        usage_count: 0,
        avg_quality: 0.0,
    });

    lib.add(PromptTemplate {
        id: "translate-text".to_string(),
        name: "Translate Text".to_string(),
        category: PromptCategory::Translation,
        template: "Translate the following text from {{source_lang}} to {{target_lang}}:\n\n{{text}}".to_string(),
        variables: vec!["source_lang".to_string(), "target_lang".to_string(), "text".to_string()],
        tags: vec!["translate".to_string(), "language".to_string(), "localization".to_string()],
        usage_count: 0,
        avg_quality: 0.0,
    });

    lib.add(PromptTemplate {
        id: "creative-story".to_string(),
        name: "Creative Story Starter".to_string(),
        category: PromptCategory::Creative,
        template: "Write the opening paragraph of a {{genre}} story set in {{setting}} featuring a character named {{protagonist}}.".to_string(),
        variables: vec!["genre".to_string(), "setting".to_string(), "protagonist".to_string()],
        tags: vec!["story".to_string(), "fiction".to_string(), "creative".to_string()],
        usage_count: 0,
        avg_quality: 0.0,
    });

    lib.add(PromptTemplate {
        id: "data-analysis".to_string(),
        name: "Data Analysis Request".to_string(),
        category: PromptCategory::Analysis,
        template: "Analyze the following dataset and provide insights about {{metric}}. Highlight trends, outliers, and recommendations:\n\n{{data}}".to_string(),
        variables: vec!["metric".to_string(), "data".to_string()],
        tags: vec!["analysis".to_string(), "data".to_string(), "insights".to_string()],
        usage_count: 0,
        avg_quality: 0.0,
    });

    lib
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lib() -> PromptLibrary {
        default_library()
    }

    #[test]
    fn test_add_and_get() {
        let mut lib = PromptLibrary::new();
        let t = PromptTemplate {
            id: "t1".to_string(),
            name: "Test".to_string(),
            category: PromptCategory::QA,
            template: "Answer: {{question}}".to_string(),
            variables: vec!["question".to_string()],
            tags: vec!["qa".to_string()],
            usage_count: 0,
            avg_quality: 0.0,
        };
        lib.add(t);
        assert!(lib.get("t1").is_some());
        assert!(lib.get("missing").is_none());
    }

    #[test]
    fn test_search_by_keyword() {
        let lib = sample_lib();
        let results = lib.search_by_keyword("translate");
        assert!(!results.is_empty());
        assert!(results.iter().any(|t| t.id == "translate-text"));
    }

    #[test]
    fn test_search_by_category() {
        let lib = sample_lib();
        let results = lib.search_by_category(PromptCategory::Coding);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "code-review");
    }

    #[test]
    fn test_popular() {
        let mut lib = sample_lib();
        lib.record_usage("code-review", 0.9);
        lib.record_usage("code-review", 0.8);
        lib.record_usage("translate-text", 0.7);
        let pop = lib.popular(1);
        assert_eq!(pop[0].id, "code-review");
    }

    #[test]
    fn test_best_rated() {
        let mut lib = sample_lib();
        lib.record_usage("summarize-article", 1.0);
        lib.record_usage("data-analysis", 0.5);
        let best = lib.best_rated(1);
        assert_eq!(best[0].id, "summarize-article");
    }

    #[test]
    fn test_record_usage_updates_average() {
        let mut lib = sample_lib();
        lib.record_usage("code-review", 0.8);
        lib.record_usage("code-review", 1.0);
        let t = lib.get("code-review").unwrap();
        assert_eq!(t.usage_count, 2);
        let expected_avg = (0.8 + 1.0) / 2.0;
        assert!((t.avg_quality - expected_avg).abs() < 1e-9);
    }

    #[test]
    fn test_similar_templates_same_category_overlapping_tags() {
        let mut lib = PromptLibrary::new();
        lib.add(PromptTemplate {
            id: "a".to_string(),
            name: "A".to_string(),
            category: PromptCategory::Creative,
            template: "...".to_string(),
            variables: vec![],
            tags: vec!["fiction".to_string(), "story".to_string()],
            usage_count: 0,
            avg_quality: 0.0,
        });
        lib.add(PromptTemplate {
            id: "b".to_string(),
            name: "B".to_string(),
            category: PromptCategory::Creative,
            template: "...".to_string(),
            variables: vec![],
            tags: vec!["story".to_string()],
            usage_count: 0,
            avg_quality: 0.0,
        });
        lib.add(PromptTemplate {
            id: "c".to_string(),
            name: "C".to_string(),
            category: PromptCategory::Coding,
            template: "...".to_string(),
            variables: vec![],
            tags: vec!["story".to_string()],
            usage_count: 0,
            avg_quality: 0.0,
        });
        let similar = lib.similar_templates("a");
        assert_eq!(similar.len(), 1);
        assert_eq!(similar[0].id, "b");
    }

    #[test]
    fn test_similar_templates_unknown_id() {
        let lib = sample_lib();
        assert!(lib.similar_templates("nonexistent").is_empty());
    }

    #[test]
    fn test_category_label() {
        assert_eq!(PromptCategory::Coding.label(), "Coding");
        assert_eq!(PromptCategory::QA.label(), "Q&A");
        assert_eq!(PromptCategory::Translation.label(), "Translation");
    }
}
