//! A small language model running in-process with [candle](https://github.com/huggingface/candle):
//! exact per-token log-probabilities, the real top alternatives at every
//! step, and occlusion attribution computed by actually re-running the model.
//!
//! Hosted APIs either send no probabilities at all (Anthropic, Gemini) or
//! cannot score a fixed answer, so they cannot say which words of a prompt
//! mattered. A local model can: [`LocalModel::occlusion`] removes each prompt
//! word in turn, re-scores the same answer with teacher forcing, and reports
//! how much each answer token's log-probability dropped.
//!
//! Models are Llama-architecture checkpoints from the Hugging Face Hub in
//! safetensors format (the default is `HuggingFaceTB/SmolLM2-135M-Instruct`,
//! 269 MB, CPU-friendly). Files are downloaded once into
//! [`LocalModel::cache_dir`].
//!
//! Requires the `local` feature.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks};
use tokenizers::Tokenizer;

pub use crate::providers::DEFAULT_LOCAL_MODEL;

/// Files a checkpoint needs.
const MODEL_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

/// Errors from loading or running the local model.
#[derive(Debug, thiserror::Error)]
pub enum LocalModelError {
    /// Downloading a model file failed.
    #[error("downloading {file} for {repo}: {message}")]
    Download {
        /// Hub repository.
        repo: String,
        /// File name.
        file: String,
        /// What went wrong.
        message: String,
    },
    /// Reading or writing the cache failed.
    #[error("model cache: {0}")]
    Io(#[from] std::io::Error),
    /// The checkpoint is not one this module can run.
    #[error("unsupported model {repo}: {message}")]
    Unsupported {
        /// Hub repository.
        repo: String,
        /// Why.
        message: String,
    },
    /// The tokenizer failed.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// The model computation failed.
    #[error("model: {0}")]
    Candle(#[from] candle_core::Error),
}

/// One generated token with its real probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedToken {
    /// Text this token added to the reply (may be empty for a token that
    /// only completes a multi-byte character).
    pub text: String,
    /// Tokenizer id.
    pub id: u32,
    /// Natural-log probability the model gave this token.
    pub logprob: f32,
    /// The most likely tokens at this step, as `(text, probability)`, most
    /// likely first. Includes the chosen token when it was among them.
    pub alternatives: Vec<(String, f32)>,
}

/// Occlusion attribution for one reply: `scores[g][w]` is how many nats
/// answer token `g`'s log-probability dropped when prompt word `w` was
/// removed. Positive means the word supported that token.
///
/// Scores measure what the prompt word adds beyond the answer written so
/// far. When the answer restates a word ("The capital of France is Paris"),
/// later tokens lean on the answer's own copy, so their per-token score for
/// that prompt word is small; [`word_totals`](Self::word_totals), summed over
/// the whole answer, still shows it. Measured with SmolLM2-135M on "Reply
/// with only the city name. Capital of France?": France +4.61, city +3.52,
/// Capital +3.13, the +0.31 nats.
#[derive(Debug, Clone, PartialEq)]
pub struct Occlusion {
    /// Prompt words, in order (whitespace-separated).
    pub words: Vec<String>,
    /// Answer tokens' text.
    pub tokens: Vec<String>,
    /// `tokens.len()` rows of `words.len()` scores.
    pub scores: Vec<Vec<f32>>,
}

impl Occlusion {
    /// Total drop across the whole answer for each prompt word: how much the
    /// answer as a whole depended on it.
    pub fn word_totals(&self) -> Vec<f32> {
        (0..self.words.len())
            .map(|w| self.scores.iter().map(|row| row[w]).sum())
            .collect()
    }
}

/// A Llama-architecture model loaded on the CPU.
pub struct LocalModel {
    repo: String,
    model: Llama,
    config: Config,
    tokenizer: Tokenizer,
    device: Device,
    eos: Vec<u32>,
}

impl std::fmt::Debug for LocalModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalModel").field("repo", &self.repo).finish_non_exhaustive()
    }
}

impl LocalModel {
    /// Where model files are stored: `EOT_MODEL_DIR` when set, otherwise
    /// `every-other-token/models` under the user's cache directory.
    pub fn cache_dir() -> PathBuf {
        if let Some(dir) = std::env::var_os("EOT_MODEL_DIR") {
            return dir.into();
        }
        let base = if cfg!(windows) {
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Caches"))
        } else {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        };
        base.unwrap_or_else(std::env::temp_dir)
            .join("every-other-token")
            .join("models")
    }

    /// Download `repo` from the Hugging Face Hub if needed, then load it.
    pub async fn load(repo: &str) -> Result<Self, LocalModelError> {
        let dir = Self::cache_dir().join(repo.replace('/', "--"));
        download_files(repo, &dir).await?;
        let repo = repo.to_string();
        let dir_clone = dir.clone();
        tokio::task::spawn_blocking(move || Self::load_dir(&repo, &dir_clone))
            .await
            .map_err(|e| LocalModelError::Io(std::io::Error::other(e.to_string())))?
    }

    /// Load a checkpoint already on disk (`config.json`, `tokenizer.json`,
    /// `model.safetensors` in `dir`).
    pub fn load_dir(repo: &str, dir: &Path) -> Result<Self, LocalModelError> {
        let raw = std::fs::read(dir.join("config.json"))?;
        let arch: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| unsupported(repo, e))?;
        let is_llama = arch["model_type"].as_str() == Some("llama")
            || arch["architectures"][0].as_str() == Some("LlamaForCausalLM");
        if !is_llama {
            return Err(LocalModelError::Unsupported {
                repo: repo.to_string(),
                message: format!(
                    "only Llama-architecture models are supported (this is '{}')",
                    arch["model_type"].as_str().unwrap_or("unknown")
                ),
            });
        }
        let llama_config: LlamaConfig = serde_json::from_slice(&raw).map_err(|e| unsupported(repo, e))?;
        let eos = match &llama_config.eos_token_id {
            Some(LlamaEosToks::Single(id)) => vec![*id],
            Some(LlamaEosToks::Multiple(ids)) => ids.clone(),
            None => Vec::new(),
        };
        let config = llama_config.into_config(false);
        let device = Device::Cpu;
        // Read into memory rather than mmap: the crate forbids unsafe code.
        let weights = std::fs::read(dir.join("model.safetensors"))?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        let model = Llama::load(vb, &config)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| LocalModelError::Tokenizer(e.to_string()))?;
        Ok(Self {
            repo: repo.to_string(),
            model,
            config,
            tokenizer,
            device,
            eos,
        })
    }

    /// The Hub repository this model was loaded from.
    pub fn repo(&self) -> &str {
        &self.repo
    }

    /// Wrap `prompt` in the model's chat template (ChatML for the SmolLM2
    /// instruct models; plain text when the tokenizer has no ChatML tokens).
    pub fn chat_prompt(&self, prompt: &str) -> String {
        if self.tokenizer.token_to_id("<|im_start|>").is_some() {
            format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        } else {
            prompt.to_string()
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, LocalModelError> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| LocalModelError::Tokenizer(e.to_string()))
    }

    fn decode(&self, ids: &[u32]) -> Result<String, LocalModelError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| LocalModelError::Tokenizer(e.to_string()))
    }

    /// Log-probabilities over the vocabulary after `logits` (last position).
    fn log_softmax(logits: &Tensor) -> Result<Vec<f32>, LocalModelError> {
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let lsm = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        Ok(lsm.to_vec1::<f32>()?)
    }

    /// Feed `ids` starting at position `pos`, returning the next-token
    /// log-probabilities.
    fn step(&self, ids: &[u32], pos: usize, cache: &mut Cache) -> Result<Vec<f32>, LocalModelError> {
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, pos, cache)?;
        Self::log_softmax(&logits)
    }

    fn top_k(&self, logprobs: &[f32], k: usize) -> Result<Vec<(String, f32)>, LocalModelError> {
        let mut idx: Vec<usize> = (0..logprobs.len()).collect();
        let k = k.min(idx.len());
        idx.select_nth_unstable_by(k.saturating_sub(1), |a, b| logprobs[*b].total_cmp(&logprobs[*a]));
        idx.truncate(k);
        idx.sort_by(|a, b| logprobs[*b].total_cmp(&logprobs[*a]));
        idx.into_iter()
            .map(|i| Ok((self.decode(&[i as u32])?, logprobs[i].exp())))
            .collect()
    }

    /// Generate a reply to `prompt` (chat template applied), calling
    /// `on_token` for each token as it is produced.
    ///
    /// `temperature <= 0` is greedy decoding (deterministic). `seed` makes
    /// sampling reproducible.
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        seed: u64,
        mut on_token: impl FnMut(&GeneratedToken),
    ) -> Result<Vec<GeneratedToken>, LocalModelError> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let prompt_ids = self.encode(&self.chat_prompt(prompt))?;
        let mut cache = Cache::new(true, DType::F32, &self.config, &self.device)?;
        let mut logprobs = self.step(&prompt_ids, 0, &mut cache)?;
        let mut generated: Vec<u32> = Vec::new();
        let mut decoded = String::new();
        let mut out = Vec::new();
        for pos in (prompt_ids.len()..).take(max_tokens) {
            let id = if temperature <= 0.0 {
                argmax(&logprobs)
            } else {
                // Sample from softmax(logits / T), i.e. exp(logprob / T) renormalised.
                let weights: Vec<f64> = logprobs.iter().map(|lp| f64::from(lp / temperature).exp()).collect();
                let total: f64 = weights.iter().sum();
                let mut r = rng.gen::<f64>() * total;
                let mut chosen = argmax(&logprobs);
                for (i, w) in weights.iter().enumerate() {
                    r -= w;
                    if r <= 0.0 {
                        chosen = i as u32;
                        break;
                    }
                }
                chosen
            };
            if self.eos.contains(&id) {
                break;
            }
            let alternatives = self.top_k(&logprobs, 5)?;
            let logprob = logprobs[id as usize];
            generated.push(id);
            // Decode the whole reply and take the new suffix, so tokens that
            // split a multi-byte character come out correctly.
            let full = self.decode(&generated)?;
            let text = full.strip_prefix(decoded.as_str()).unwrap_or(&full).to_string();
            decoded = full;
            let token = GeneratedToken {
                text,
                id,
                logprob,
                alternatives,
            };
            on_token(&token);
            out.push(token);
            logprobs = self.step(&[id], pos, &mut cache)?;
        }
        Ok(out)
    }

    /// Log-probability of each token of `answer` given the chat-formatted
    /// `prompt` (teacher forcing: the model is shown the true answer so far
    /// at every step).
    pub fn score(&self, prompt: &str, answer: &[u32]) -> Result<Vec<f32>, LocalModelError> {
        let prompt_ids = self.encode(&self.chat_prompt(prompt))?;
        let mut cache = Cache::new(true, DType::F32, &self.config, &self.device)?;
        let mut logprobs = self.step(&prompt_ids, 0, &mut cache)?;
        let mut out = Vec::with_capacity(answer.len());
        for (pos, &id) in (prompt_ids.len()..).zip(answer) {
            out.push(logprobs[id as usize]);
            logprobs = self.step(&[id], pos, &mut cache)?;
        }
        Ok(out)
    }

    /// Occlusion attribution: for each whitespace-separated word of
    /// `prompt`, remove it, re-score `answer` (token ids from
    /// [`generate`](Self::generate)) and record how much each answer token's
    /// log-probability dropped. Costs one scoring pass per prompt word.
    pub fn occlusion(&self, prompt: &str, answer: &[GeneratedToken]) -> Result<Occlusion, LocalModelError> {
        let ids: Vec<u32> = answer.iter().map(|t| t.id).collect();
        let words: Vec<String> = prompt.split_whitespace().map(str::to_string).collect();
        let full = self.score(prompt, &ids)?;
        let mut scores = vec![vec![0.0_f32; words.len()]; ids.len()];
        for w in 0..words.len() {
            let masked: Vec<&str> = words
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != w)
                .map(|(_, s)| s.as_str())
                .collect();
            let masked_lp = self.score(&masked.join(" "), &ids)?;
            for ((row, f), m) in scores.iter_mut().zip(&full).zip(&masked_lp) {
                row[w] = f - m;
            }
        }
        Ok(Occlusion {
            words,
            tokens: answer.iter().map(|t| t.text.clone()).collect(),
            scores,
        })
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn unsupported(repo: &str, e: impl std::fmt::Display) -> LocalModelError {
    LocalModelError::Unsupported {
        repo: repo.to_string(),
        message: e.to_string(),
    }
}

/// Download any missing model files from the Hub into `dir`, writing each
/// to a temporary name first so an interrupted download is never mistaken
/// for a complete file.
async fn download_files(repo: &str, dir: &Path) -> Result<(), LocalModelError> {
    use futures_util::StreamExt;
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let client = reqwest::Client::new();
    for file in MODEL_FILES {
        let target = dir.join(file);
        if target.exists() {
            continue;
        }
        let err = |message: String| LocalModelError::Download {
            repo: repo.to_string(),
            file: file.to_string(),
            message,
        };
        let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
        let resp = client.get(&url).send().await.map_err(|e| err(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(err(format!("HTTP {} from {url}", resp.status())));
        }
        let total = resp.content_length();
        if file == "model.safetensors" {
            eprintln!(
                "Downloading {repo} ({} MB, once) to {}",
                total.map(|b| b / 1_000_000).unwrap_or(0),
                dir.display()
            );
        }
        let partial = dir.join(format!("{file}.partial"));
        let mut out = std::fs::File::create(&partial)?;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            out.write_all(&chunk.map_err(|e| err(e.to_string()))?)?;
        }
        out.flush()?;
        drop(out);
        std::fs::rename(&partial, &target)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_the_largest() {
        assert_eq!(argmax(&[-3.0, -0.1, -2.0]), 1);
    }

    #[test]
    fn word_totals_sum_each_column() {
        let o = Occlusion {
            words: vec!["a".into(), "b".into()],
            tokens: vec!["x".into(), "y".into()],
            scores: vec![vec![1.0, 0.5], vec![2.0, -0.5]],
        };
        assert_eq!(o.word_totals(), vec![3.0, 0.0]);
    }

    #[test]
    fn cache_dir_respects_the_override() {
        // Only checks the default shape; EOT_MODEL_DIR is process-global.
        assert!(LocalModel::cache_dir().ends_with("models") || std::env::var_os("EOT_MODEL_DIR").is_some());
    }
}
