//! Full-screen terminal view (`--tui`), built on `ratatui`.
//!
//! The stream runs in a background task and sends [`TokenEvent`]s over the
//! same channel the web UI uses. The view shows:
//!
//! - the reply as it arrives, each token colored by the model's confidence
//!   (green sure, yellow unsure, red guessing) and underlined when the
//!   transform rewrote it;
//! - running stats: token count, rewritten count, mean confidence and
//!   perplexity, and the token the model was least sure about;
//! - a sparkline of confidence per token;
//! - the latest token's top alternatives as bars.
//!
//! Keys: `q` or `Esc` quits (also `Ctrl+C`), arrow keys and Page Up/Down
//! scroll, `End` follows the stream again.
//!
//! Drawing is split from terminal handling so tests can render into
//! `ratatui`'s in-memory `TestBackend`.

use crate::{TokenEvent, TokenInterceptor};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Sparkline, Wrap};
use ratatui::{Frame, Terminal};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Where the stream is.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamStatus {
    /// Tokens are still arriving.
    Streaming,
    /// The provider finished the reply.
    Done,
    /// The stream ended with an error (message shown in the header).
    Failed(String),
}

/// Everything the view shows. Pure data, so it can be built in tests.
pub struct TuiState {
    /// Provider name shown in the header.
    pub provider: String,
    /// Model name shown in the header.
    pub model: String,
    /// Transform name shown in the header.
    pub transform: String,
    /// The prompt that was sent.
    pub prompt: String,
    /// Token events received so far.
    pub events: Vec<TokenEvent>,
    /// Stream state.
    pub status: StreamStatus,
    started: Instant,
    finished_after: Option<Duration>,
    /// Lines scrolled up from the bottom; `None` follows the newest token.
    scroll_back: Option<u16>,
}

/// Summary numbers shown in the stats panel.
#[derive(Debug, Clone, PartialEq)]
pub struct TuiStats {
    /// Tokens received.
    pub tokens: usize,
    /// Tokens the transform rewrote.
    pub rewritten: usize,
    /// Mean confidence over tokens that had one.
    pub mean_confidence: Option<f32>,
    /// Mean perplexity over tokens that had one.
    pub mean_perplexity: Option<f32>,
    /// The token with the lowest confidence and that confidence.
    pub least_sure: Option<(String, f32)>,
}

impl TuiState {
    /// New empty state for a run.
    pub fn new(provider: &str, model: &str, transform: &str, prompt: &str) -> Self {
        TuiState {
            provider: provider.to_string(),
            model: model.to_string(),
            transform: transform.to_string(),
            prompt: prompt.to_string(),
            events: Vec::new(),
            status: StreamStatus::Streaming,
            started: Instant::now(),
            finished_after: None,
            scroll_back: None,
        }
    }

    /// Record one token event. Error events end the stream as failed.
    pub fn push(&mut self, ev: TokenEvent) {
        if ev.is_error {
            self.finish(Err(ev.text));
        } else {
            self.events.push(ev);
        }
    }

    /// Mark the stream finished, successfully or not.
    pub fn finish(&mut self, result: Result<(), String>) {
        if self.status != StreamStatus::Streaming {
            return;
        }
        self.finished_after = Some(self.started.elapsed());
        self.status = match result {
            Ok(()) => StreamStatus::Done,
            Err(e) => StreamStatus::Failed(e),
        };
    }

    /// Compute the stats panel numbers.
    pub fn stats(&self) -> TuiStats {
        let confs: Vec<f32> = self.events.iter().filter_map(|e| e.confidence).collect();
        let perps: Vec<f32> = self.events.iter().filter_map(|e| e.perplexity).collect();
        let mean = |v: &[f32]| (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32);
        let least_sure = self
            .events
            .iter()
            .filter_map(|e| e.confidence.map(|c| (e.original.trim().to_string(), c)))
            .filter(|(t, _)| !t.is_empty())
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        TuiStats {
            tokens: self.events.len(),
            rewritten: self.events.iter().filter(|e| e.transformed).count(),
            mean_confidence: mean(&confs),
            mean_perplexity: mean(&perps),
            least_sure,
        }
    }

    /// The reply as displayed (after transforms), for printing on exit.
    pub fn text(&self) -> String {
        self.events.iter().map(|e| e.text.as_str()).collect()
    }

    /// Handle a key press. Returns `true` when the user asked to quit.
    pub fn on_key(&mut self, key: KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        let back = self.scroll_back.unwrap_or(0);
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => return true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
            KeyCode::Up => self.scroll_back = Some(back.saturating_add(1)),
            KeyCode::PageUp => self.scroll_back = Some(back.saturating_add(10)),
            KeyCode::Down => self.scroll_back = back.checked_sub(1).filter(|b| *b > 0),
            KeyCode::PageDown => self.scroll_back = back.checked_sub(10).filter(|b| *b > 0),
            KeyCode::End => self.scroll_back = None,
            _ => {}
        }
        false
    }
}

/// Color for a confidence value: green when sure, yellow when unsure, red
/// when guessing, gray when the provider gave no confidence.
pub fn confidence_color(conf: Option<f32>) -> Color {
    match conf {
        Some(c) if c >= 0.8 => Color::Green,
        Some(c) if c >= 0.5 => Color::Yellow,
        Some(_) => Color::Red,
        None => Color::Gray,
    }
}

fn fmt_opt(v: Option<f32>, digits: usize) -> String {
    v.map(|x| format!("{x:.digits$}")).unwrap_or_else(|| "n/a".to_string())
}

/// Draw the whole view into `frame`.
pub fn draw(frame: &mut Frame, state: &TuiState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(4),
            Constraint::Length(7),
            Constraint::Length(1),
        ])
        .split(frame.area());

    draw_header(frame, rows[0], state);

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(30), Constraint::Length(30)])
        .split(rows[1]);
    draw_stream(frame, middle[0], state);
    draw_stats(frame, middle[1], state);
    draw_sparkline(frame, rows[2], state);
    draw_alternatives(frame, rows[3], state);

    let help = Line::from(vec![
        Span::styled(" q ", Style::default().add_modifier(Modifier::REVERSED)),
        Span::raw(" quit  "),
        Span::styled(" \u{2191}\u{2193} PgUp PgDn ", Style::default().add_modifier(Modifier::REVERSED)),
        Span::raw(" scroll  "),
        Span::styled(" End ", Style::default().add_modifier(Modifier::REVERSED)),
        Span::raw(" follow"),
    ]);
    frame.render_widget(Paragraph::new(help), rows[4]);
}

fn draw_header(frame: &mut Frame, area: Rect, state: &TuiState) {
    let elapsed = state
        .finished_after
        .unwrap_or_else(|| state.started.elapsed())
        .as_secs_f32();
    let (status, color) = match &state.status {
        StreamStatus::Streaming => (format!("streaming {elapsed:.1}s"), Color::Cyan),
        StreamStatus::Done => (format!("done in {elapsed:.1}s"), Color::Green),
        StreamStatus::Failed(e) => (format!("failed: {e}"), Color::Red),
    };
    let title = format!(
        " every-other-token  {} \u{00b7} {} \u{00b7} {} ",
        state.provider, state.model, state.transform
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_bottom(Line::from(Span::styled(format!(" {status} "), Style::default().fg(color))).right_aligned());
    let prompt = Paragraph::new(Line::from(vec![
        Span::styled("Prompt: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(state.prompt.replace('\n', " ")),
    ]))
    .block(block);
    frame.render_widget(prompt, area);
}

fn draw_stream(frame: &mut Frame, area: Rect, state: &TuiState) {
    let spans: Vec<Span> = state
        .events
        .iter()
        .map(|e| {
            let mut style = Style::default().fg(confidence_color(e.confidence));
            if e.transformed {
                style = style.add_modifier(Modifier::UNDERLINED | Modifier::BOLD);
            }
            Span::styled(e.text.clone(), style)
        })
        .collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Reply (color = confidence, underlined = rewritten) ");
    let para = Paragraph::new(Line::from(spans)).wrap(Wrap { trim: false });
    // Follow the newest token unless the user scrolled back.
    let inner_w = area.width.saturating_sub(2);
    let inner_h = area.height.saturating_sub(2);
    let total = para.line_count(inner_w) as u16;
    let bottom = total.saturating_sub(inner_h);
    let offset = bottom.saturating_sub(state.scroll_back.unwrap_or(0));
    frame.render_widget(para.block(block).scroll((offset, 0)), area);
}

fn draw_stats(frame: &mut Frame, area: Rect, state: &TuiState) {
    let s = state.stats();
    let label = |t: &str| Span::styled(format!("{t:<13}"), Style::default().fg(Color::DarkGray));
    let mut lines = vec![
        Line::from(vec![label("Tokens"), Span::raw(s.tokens.to_string())]),
        Line::from(vec![label("Rewritten"), Span::raw(s.rewritten.to_string())]),
        Line::from(vec![
            label("Confidence"),
            Span::styled(fmt_opt(s.mean_confidence, 3), Style::default().fg(confidence_color(s.mean_confidence))),
        ]),
        Line::from(vec![label("Perplexity"), Span::raw(fmt_opt(s.mean_perplexity, 2))]),
    ];
    if let Some((tok, c)) = &s.least_sure {
        lines.push(Line::from(""));
        lines.push(Line::from(label("Least sure")));
        lines.push(Line::from(vec![
            Span::styled(format!("{tok:?}"), Style::default().fg(confidence_color(Some(*c)))),
            Span::raw(format!(" {c:.2}")),
        ]));
    }
    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Stats "));
    frame.render_widget(para, area);
}

fn draw_sparkline(frame: &mut Frame, area: Rect, state: &TuiState) {
    let width = area.width.saturating_sub(2) as usize;
    let data: Vec<u64> = state
        .events
        .iter()
        .map(|e| e.confidence.map(|c| (c.clamp(0.0, 1.0) * 100.0).round() as u64).unwrap_or(0))
        .collect();
    let tail = &data[data.len().saturating_sub(width)..];
    let spark = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(" Confidence per token "))
        .data(tail)
        .max(100)
        .style(Style::default().fg(Color::Cyan));
    frame.render_widget(spark, area);
}

fn draw_alternatives(frame: &mut Frame, area: Rect, state: &TuiState) {
    // The newest token that came with alternatives (OpenAI-style logprobs).
    let latest = state.events.iter().rev().find(|e| !e.alternatives.is_empty());
    let title = match latest {
        Some(e) => format!(" What else the model considered for {:?} ", e.original.trim()),
        None => " Alternatives (shown when the provider returns logprobs) ".to_string(),
    };
    let bar_w = area.width.saturating_sub(30).max(5) as f32;
    let lines: Vec<Line> = latest
        .map(|e| {
            e.alternatives
                .iter()
                .take(5)
                .map(|a| {
                    let n = (a.probability.clamp(0.0, 1.0) * bar_w).round() as usize;
                    Line::from(vec![
                        Span::raw(format!("{:<16}", format!("{:?}", a.token))),
                        Span::styled("\u{2588}".repeat(n.max(1)), Style::default().fg(confidence_color(Some(a.probability)))),
                        Span::raw(format!(" {:.3}", a.probability)),
                    ])
                })
                .collect()
        })
        .unwrap_or_default();
    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(para, area);
}

/// Drive the view until the user quits.
///
/// `rx` delivers token events and `stream` is the task producing them.
/// `next_key` waits up to the given time for a key press; the real terminal
/// passes a `crossterm` poll, tests pass a scripted sequence.
pub async fn event_loop<B, K>(
    terminal: &mut Terminal<B>,
    state: &mut TuiState,
    rx: &mut mpsc::UnboundedReceiver<TokenEvent>,
    mut stream: tokio::task::JoinHandle<Result<(), String>>,
    mut next_key: K,
) -> Result<(), Box<dyn std::error::Error>>
where
    B: ratatui::backend::Backend,
    B::Error: 'static,
    K: FnMut(Duration) -> std::io::Result<Option<KeyEvent>>,
{
    let mut stream_done = false;
    loop {
        while let Ok(ev) = rx.try_recv() {
            state.push(ev);
        }
        if !stream_done && stream.is_finished() {
            stream_done = true;
            // Pick up anything sent just before the task ended.
            while let Ok(ev) = rx.try_recv() {
                state.push(ev);
            }
            let result = match (&mut stream).await {
                Ok(r) => r,
                Err(e) => Err(format!("stream task failed: {e}")),
            };
            state.finish(result);
        }
        terminal.draw(|f| draw(f, state))?;
        if let Some(key) = next_key(Duration::from_millis(33))? {
            if state.on_key(key) {
                stream.abort();
                return Ok(());
            }
        }
    }
}

/// Run `interceptor` on `prompt` in the full-screen view.
///
/// After the user quits, the reply and a one-line summary are printed to
/// stdout so they stay in the terminal's scrollback.
///
/// # Errors
/// Returns an error when stdout is not an interactive terminal.
pub async fn run(mut interceptor: TokenInterceptor, prompt: String) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        return Err("--tui needs an interactive terminal; use --json-stream when piping".into());
    }
    let mut state = TuiState::new(
        &interceptor.provider.to_string(),
        &interceptor.model,
        &format!("{:?}", interceptor.transform).to_lowercase(),
        &prompt,
    );
    let (tx, mut rx) = mpsc::unbounded_channel();
    interceptor.web_tx = Some(tx);
    let stream = tokio::spawn(async move {
        interceptor.intercept_stream(&prompt).await.map_err(|e| e.to_string())
    });

    let mut terminal = ratatui::try_init()?;
    let result = event_loop(&mut terminal, &mut state, &mut rx, stream, |timeout| {
        tokio::task::block_in_place(|| {
            if event::poll(timeout)? {
                if let Event::Key(k) = event::read()? {
                    return Ok(Some(k));
                }
            }
            Ok(None)
        })
    })
    .await;
    ratatui::restore();
    result?;

    let s = state.stats();
    println!("{}", state.text());
    println!(
        "{} tokens, {} rewritten, mean confidence {}",
        s.tokens,
        s.rewritten,
        fmt_opt(s.mean_confidence, 3)
    );
    if let StreamStatus::Failed(e) = &state.status {
        return Err(e.clone().into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenAlternative;
    use ratatui::backend::TestBackend;

    fn ev(text: &str, original: &str, conf: Option<f32>, transformed: bool) -> TokenEvent {
        TokenEvent {
            text: text.into(),
            original: original.into(),
            index: 0,
            transformed,
            importance: 0.5,
            chaos_label: None,
            provider: None,
            confidence: conf,
            perplexity: conf.map(|c| 1.0 / c),
            alternatives: vec![],
            is_error: false,
            arrival_ms: None,
        }
    }

    fn buffer_text(t: &Terminal<TestBackend>) -> String {
        let buf = t.backend().buffer();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn stats_count_and_average() {
        let mut s = TuiState::new("mock", "m", "reverse", "p");
        s.push(ev("The", "The", Some(0.9), false));
        s.push(ev(" kciuq", " quick", Some(0.5), true));
        s.push(ev(" fox", " fox", None, false));
        let st = s.stats();
        assert_eq!(st.tokens, 3);
        assert_eq!(st.rewritten, 1);
        assert!((st.mean_confidence.unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(st.least_sure, Some(("quick".to_string(), 0.5)));
        assert_eq!(s.text(), "The kciuq fox");
    }

    #[test]
    fn error_event_marks_stream_failed() {
        let mut s = TuiState::new("openai", "m", "reverse", "p");
        let mut e = ev("[orchestrator error] down", "", None, false);
        e.is_error = true;
        s.push(e);
        assert_eq!(s.status, StreamStatus::Failed("[orchestrator error] down".into()));
        assert!(s.events.is_empty());
    }

    #[test]
    fn keys_scroll_and_quit() {
        let mut s = TuiState::new("mock", "m", "reverse", "p");
        let key = |c| KeyEvent::new(c, KeyModifiers::NONE);
        assert!(!s.on_key(key(KeyCode::Up)));
        assert!(!s.on_key(key(KeyCode::PageUp)));
        assert_eq!(s.scroll_back, Some(11));
        s.on_key(key(KeyCode::End));
        assert_eq!(s.scroll_back, None);
        assert!(s.on_key(key(KeyCode::Char('q'))));
        assert!(s.on_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)));
    }

    #[test]
    fn confidence_colors() {
        assert_eq!(confidence_color(Some(0.95)), Color::Green);
        assert_eq!(confidence_color(Some(0.6)), Color::Yellow);
        assert_eq!(confidence_color(Some(0.2)), Color::Red);
        assert_eq!(confidence_color(None), Color::Gray);
    }

    #[test]
    fn draw_renders_reply_stats_and_alternatives() {
        let mut s = TuiState::new("openai", "gpt-4o", "reverse", "Why is the sky blue?");
        s.push(ev("The", "The", Some(0.9), false));
        let mut e = ev(" kciuq", " quick", Some(0.41), true);
        e.alternatives = vec![
            TokenAlternative { token: " quick".into(), probability: 0.41 },
            TokenAlternative { token: " slow".into(), probability: 0.2 },
        ];
        s.push(e);
        s.finish(Ok(()));
        let mut t = Terminal::new(TestBackend::new(100, 30)).unwrap();
        t.draw(|f| draw(f, &s)).unwrap();
        let text = buffer_text(&t);
        assert!(text.contains("The kciuq"), "{text}");
        assert!(text.contains("Why is the sky blue?"));
        assert!(text.contains("openai"));
        assert!(text.contains("Rewritten"));
        assert!(text.contains("\" slow\""), "{text}");
        assert!(text.contains("done in"));

        // The rewritten token is underlined and colored by its confidence.
        let buf = t.backend().buffer();
        let (x, y) = (0..buf.area.height)
            .flat_map(|y| (0..buf.area.width).map(move |x| (x, y)))
            .find(|&(x, y)| {
                x + 5 <= buf.area.width
                    && (0..5).map(|i| buf[(x + i, y)].symbol()).collect::<String>() == "kciuq"
            })
            .expect("kciuq on screen");
        assert!(buf[(x, y)].modifier.contains(Modifier::UNDERLINED));
        assert_eq!(buf[(x, y)].fg, Color::Red);
    }

    #[test]
    fn long_reply_follows_the_newest_token() {
        let mut s = TuiState::new("mock", "m", "none", "p");
        for i in 0..400 {
            s.push(ev(&format!(" w{i}"), &format!(" w{i}"), Some(0.9), false));
        }
        let mut t = Terminal::new(TestBackend::new(80, 24)).unwrap();
        t.draw(|f| draw(f, &s)).unwrap();
        let text = buffer_text(&t);
        assert!(text.contains("w399"), "newest token must be visible");
        assert!(!text.contains(" w1 "), "oldest tokens scrolled off");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn event_loop_runs_mock_stream_to_completion() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut interceptor = TokenInterceptor::new(
            crate::providers::Provider::Mock,
            crate::transforms::Transform::Reverse,
            "mock-fixture-v1".into(),
            false,
            false,
            false,
        )
        .unwrap()
        .with_web_tx(tx);
        let stream = tokio::spawn(async move {
            interceptor.intercept_stream("Why is the sky blue?").await.map_err(|e| e.to_string())
        });
        let mut state = TuiState::new("mock", "mock-fixture-v1", "reverse", "Why is the sky blue?");
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        // Press nothing until the stream is done and drawn, then quit.
        let mut ticks = 0;
        let started = Instant::now();
        event_loop(&mut terminal, &mut state, &mut rx, stream, |timeout| {
            std::thread::sleep(timeout);
            ticks += 1;
            assert!(started.elapsed() < Duration::from_secs(10), "stream never finished");
            Ok((ticks > 3).then(|| KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE)))
        })
        .await
        .unwrap();
        // Ticks run quickly; if the stream had not finished by tick 4 the
        // assert below fails rather than hanging.
        assert_eq!(state.status, StreamStatus::Done);
        assert!(state.text().contains("kciuq"));
        assert!(buffer_text(&terminal).contains("done in"));
    }
}
