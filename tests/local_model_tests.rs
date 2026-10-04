//! Tests for `--provider local` (feature `local`) against the real default
//! model, SmolLM2-135M-Instruct.
//!
//! Ignored by default: the first run downloads the model (269 MB) into
//! `LocalModel::cache_dir()`. Run with:
//!
//! ```text
//! cargo test --features local --test local_model_tests -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "local")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use every_other_token::local_model::{LocalModel, DEFAULT_LOCAL_MODEL};

async fn model() -> LocalModel {
    LocalModel::load(DEFAULT_LOCAL_MODEL).await.expect("load the default local model")
}

#[tokio::test]
#[ignore = "downloads a 269 MB model on first run"]
async fn greedy_answer_is_sensible_and_probabilities_are_valid() {
    let m = model().await;
    let tokens = m
        .generate("What is the capital of France? Answer in one word.", 12, 0.0, 0, |_| {})
        .expect("generate");
    let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
    println!("answer: {text:?}");
    assert!(text.contains("Paris"), "{text}");
    for t in &tokens {
        assert!(t.logprob <= 0.0, "a log-probability is never positive: {t:?}");
        let mass: f32 = t.alternatives.iter().map(|(_, p)| p).sum();
        assert!(mass <= 1.0001, "top-5 probabilities cannot exceed 1: {mass}");
        // Greedy decoding picks the most likely token, so it is the first alternative.
        assert!((t.alternatives[0].1 - t.logprob.exp()).abs() < 1e-4, "{t:?}");
    }
}

#[tokio::test]
#[ignore = "downloads a 269 MB model on first run"]
async fn teacher_forced_scores_match_generation_logprobs() {
    // Scoring the model's own answer must reproduce the log-probabilities it
    // reported while generating it. If not, scoring (and so attribution) is wrong.
    let m = model().await;
    let prompt = "Name a primary color.";
    let tokens = m.generate(prompt, 10, 0.0, 0, |_| {}).expect("generate");
    let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
    let scored = m.score(prompt, &ids).expect("score");
    for (t, s) in tokens.iter().zip(&scored) {
        assert!((t.logprob - s).abs() < 1e-3, "generation {} vs scoring {s} for {:?}", t.logprob, t.text);
    }
}

#[tokio::test]
#[ignore = "downloads a 269 MB model on first run"]
async fn occlusion_finds_the_word_the_answer_depends_on() {
    // The answer must name the city without restating the question: if it
    // said "The capital of France is Paris", the answer's own "France" would
    // carry the information and removing it from the prompt would not matter.
    let m = model().await;
    let prompt = "Reply with only the city name. Capital of France?";
    let tokens = m.generate(prompt, 6, 0.0, 0, |_| {}).expect("generate");
    let answer: String = tokens.iter().map(|t| t.text.as_str()).collect();
    println!("answer: {answer:?}");
    let occ = m.occlusion(prompt, &tokens).expect("occlusion");
    for (w, total) in occ.words.iter().zip(occ.word_totals()) {
        println!("  {w:>10}  {total:+.3}");
    }
    // Per answer token, a word the answer itself restates ("...of France is
    // Paris") gets little credit: the answer's own copy carries it. Over the
    // whole answer, the content words must dominate the filler.
    let totals = occ.word_totals();
    let total = |word: &str| {
        let w = occ
            .words
            .iter()
            .position(|x| x.trim_end_matches(['?', '.']) == word)
            .expect(word);
        totals[w]
    };
    let max = totals.iter().cloned().fold(f32::MIN, f32::max);
    assert_eq!(total("France"), max, "France is the word the answer depends on most");
    assert!(total("France") > 5.0 * total("the"), "France {} vs the {}", total("France"), total("the"));
    assert!(total("France") > 2.0, "removing France changes the answer a lot");
}

#[tokio::test]
#[ignore = "downloads a 269 MB model on first run"]
async fn sampling_is_reproducible_with_a_seed() {
    let m = model().await;
    let a = m.generate("Tell me a fruit.", 8, 0.8, 42, |_| {}).expect("a");
    let b = m.generate("Tell me a fruit.", 8, 0.8, 42, |_| {}).expect("b");
    assert_eq!(
        a.iter().map(|t| t.id).collect::<Vec<_>>(),
        b.iter().map(|t| t.id).collect::<Vec<_>>()
    );
}
