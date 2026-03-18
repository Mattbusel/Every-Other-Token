use crate::TokenEvent;
use std::io::Write;

/// Accumulates per-position confidence data across runs.
pub struct HeatmapExporter {
    /// data[position][run] = confidence value (if available)
    data: Vec<Vec<Option<f32>>>,
    run_count: usize,
}

impl HeatmapExporter {
    pub fn new() -> Self {
        HeatmapExporter {
            data: Vec::new(),
            run_count: 0,
        }
    }

    pub fn record_run(&mut self, events: &[TokenEvent]) {
        let run_idx = self.run_count;
        self.run_count += 1;

        // Ensure data has enough positions
        if events.len() > self.data.len() {
            self.data.resize(events.len(), Vec::new());
        }

        for (pos, event) in events.iter().enumerate() {
            // Ensure this position's vec has enough entries for prior runs
            while self.data[pos].len() < run_idx {
                self.data[pos].push(None);
            }
            self.data[pos].push(event.confidence);
        }
    }

    pub fn export_csv(&self, path: &str, min_confidence: f32, sort_by: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = std::fs::File::create(path)?;

        // Header: position, run_0, run_1, ...
        let header_cols: Vec<String> = (0..self.run_count).map(|i| format!("run_{}", i)).collect();
        writeln!(file, "position,{}", header_cols.join(","))?;

        // Compute (position, mean_confidence, row_string) tuples
        let mut rows: Vec<(usize, f32, String)> = self.data.iter().enumerate()
            .map(|(pos, runs)| {
                let vals: Vec<f32> = runs.iter().filter_map(|v| *v).collect();
                let mean = if vals.is_empty() { 0.0f32 } else { vals.iter().sum::<f32>() / vals.len() as f32 };
                let cols: Vec<String> = (0..self.run_count)
                    .map(|r| runs.get(r).and_then(|v| *v).map(|v| v.to_string()).unwrap_or_default())
                    .collect();
                (pos, mean, format!("{},{}", pos, cols.join(",")))
            })
            .filter(|(_, mean, _)| *mean >= min_confidence)
            .collect();

        if sort_by == "confidence" {
            rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        for (_, _, row) in &rows {
            writeln!(file, "{}", row)?;
        }

        Ok(())
    }
}

impl Default for HeatmapExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenEvent;

    fn make_event(idx: usize, confidence: Option<f32>) -> TokenEvent {
        TokenEvent {
            text: "tok".to_string(),
            original: "tok".to_string(),
            index: idx,
            transformed: false,
            importance: 0.0,
            chaos_label: None,
            provider: None,
            confidence,
            perplexity: None,
            alternatives: vec![],
            is_error: false,
        }
    }

    #[test]
    fn test_record_and_export() {
        let mut exporter = HeatmapExporter::new();
        let events = vec![
            make_event(0, Some(0.9)),
            make_event(1, Some(0.8)),
        ];
        exporter.record_run(&events);

        let tmp = std::env::temp_dir().join("heatmap_test.csv");
        exporter.export_csv(tmp.to_str().unwrap(), 0.0, "position").expect("export");
        let content = std::fs::read_to_string(&tmp).expect("read");
        assert!(content.contains("position,run_0"));
        assert!(content.contains("0,0.9"));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_empty_exporter() {
        let exporter = HeatmapExporter::new();
        let tmp = std::env::temp_dir().join("heatmap_empty.csv");
        exporter.export_csv(tmp.to_str().unwrap(), 0.0, "position").expect("export empty");
        std::fs::remove_file(&tmp).ok();
    }
}
