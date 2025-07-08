use clap::{Arg, Command};
use colored::*;
use rand::Rng;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{self, Write};
use tokio;
use tokio_stream::StreamExt;

#[derive(Debug, Clone)]
pub enum Transform {
    Reverse,
    Uppercase,
    Mock,
    Noise,
}

impl Transform {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "reverse" => Ok(Transform::Reverse),
            "uppercase" => Ok(Transform::Uppercase),
            "mock" => Ok(Transform::Mock),
            "noise" => Ok(Transform::Noise),
            _ => Err(format!("Unknown transform: {}", s)),
        }
    }

    fn apply(&self, token: &str) -> String {
        match self {
            Transform::Reverse => token.chars().rev().collect(),
            Transform::Uppercase => token.to_uppercase(),
            Transform::Mock => {
                token
                    .chars()
                    .enumerate()
                    .map(|(i, c)| {
                        if i % 2 == 0 {
                            c.to_lowercase().next().unwrap_or(c)
                        } else {
                            c.to_uppercase().next().unwrap_or(c)
                        }
                    })
                    .collect()
            }
            Transform::Noise => {
                let mut rng = rand::thread_rng();
                let noise_chars = ['*', '+', '~', '@', '#', '$', '%'];
                let noise_char = noise_chars[rng.gen_range(0..noise_chars.len())];
                format!("{}{}", token, noise_char)
            }
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct ChatResponseDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponseChoice {
    delta: ChatResponseDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatResponseChoice>,
}

pub struct TokenInterceptor {
    client: Client,
    api_key: String,
    transform: Transform,
    model: String,
    token_count: usize,
    transformed_count: usize,
}

impl TokenInterceptor {
    pub fn new(transform: Transform, model: String) -> Result<Self, Box<dyn std::error::Error>> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| "OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")?;

        let client = Client::new();

        Ok(TokenInterceptor {
            client,
            api_key,
            transform,
            model,
            token_count: 0,
            transformed_count: 0,
        })
    }

    pub async fn intercept_stream(&mut self, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.print_header(prompt);

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            stream: true,
            temperature: 0.7,
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("API Error: {}", error_text).into());
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str);

            // Process complete lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer.drain(..=line_end);

                if line.starts_with("data: ") && line != "data: [DONE]" {
                    let json_str = &line[6..]; // Remove "data: " prefix
                    
                    if let Ok(response) = serde_json::from_str::<ChatResponse>(json_str) {
                        if let Some(choice) = response.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                self.process_content(content);
                            }
                        }
                    }
                }
            }
        }

        self.print_footer();
        Ok(())
    }

    fn process_content(&mut self, content: &str) {
        // Simple tokenization - split by whitespace and punctuation
        let tokens = self.tokenize(content);
        
        for token in tokens {
            if !token.trim().is_empty() {
                let processed_token = if self.token_count % 2 == 0 {
                    // Even tokens - pass through unchanged
                    token.clone()
                } else {
                    // Odd tokens - apply transformation
                    self.transformed_count += 1;
                    self.transform.apply(&token)
                };

                print!("{}", processed_token);
                io::stdout().flush().unwrap();
                self.token_count += 1;
            }
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() || ch.is_ascii_punctuation() {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                if !ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
                if ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
            } else {
                current_token.push(ch);
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        tokens
    }

    fn print_header(&self, prompt: &str) {
        println!("{}", "🔮 EVERY OTHER TOKEN INTERCEPTOR".bright_cyan().bold());
        println!("{}: {:?}", "Transform".bright_yellow(), self.transform);
        println!("{}: {}", "Model".bright_yellow(), self.model);
        println!("{}: {}", "Prompt".bright_yellow(), prompt);
        println!("{}", "=".repeat(50).bright_blue());
        println!("{}", "Response (with transformations):".bright_green());
        println!();
    }

    fn print_footer(&self) {
        println!("\n{}", "=".repeat(50).bright_blue());
        println!("{} Complete! Processed {} tokens.", "✅".bright_green(), self.token_count);
        println!("{} Transform applied to {} tokens.", "🔄".bright_yellow(), self.transformed_count);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("API Key from env: {:?}", std::env::var("OPENAI_API_KEY"));
    
    let matches = Command::new("every-other-token")
        .version("1.0.0")
        .author("AI Research Tools")
        .about("A real-time LLM stream interceptor for token-level interaction research")
        .arg(
            Arg::new("prompt")
                .help("Your input prompt")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("transform")
                .help("Transformation type (reverse, uppercase, mock, noise)")
                .default_value("reverse")
                .index(2),
        )
        .arg(
            Arg::new("model")
                .help("OpenAI model to use")
                .default_value("gpt-3.5-turbo")
                .index(3),
        )
        .get_matches();

    let prompt = matches.get_one::<String>("prompt").unwrap();
    let transform_str = matches.get_one::<String>("transform").unwrap();
    let model = matches.get_one::<String>("model").unwrap();

    let transform = Transform::from_str(transform_str)
        .map_err(|e| format!("Invalid transform: {}", e))?;

    let mut interceptor = TokenInterceptor::new(transform, model.clone())?;
    interceptor.intercept_stream(prompt).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_reverse() {
        let transform = Transform::Reverse;
        assert_eq!(transform.apply("hello"), "olleh");
        assert_eq!(transform.apply("world"), "dlrow");
    }

    #[test]
    fn test_transform_uppercase() {
        let transform = Transform::Uppercase;
        assert_eq!(transform.apply("hello"), "HELLO");
        assert_eq!(transform.apply("world"), "WORLD");
    }

    #[test]
    fn test_transform_mock() {
        let transform = Transform::Mock;
        assert_eq!(transform.apply("hello"), "hElLo");
        assert_eq!(transform.apply("world"), "wOrLd");
    }

    #[test]
    fn test_transform_noise() {
        let transform = Transform::Noise;
        let result = transform.apply("hello");
        assert!(result.starts_with("hello"));
        assert!(result.len() > 5);
    }

    #[test]
    fn test_transform_from_str() {
        assert!(matches!(Transform::from_str("reverse"), Ok(Transform::Reverse)));
        assert!(matches!(Transform::from_str("uppercase"), Ok(Transform::Uppercase)));
        assert!(matches!(Transform::from_str("mock"), Ok(Transform::Mock)));
        assert!(matches!(Transform::from_str("noise"), Ok(Transform::Noise)));
        assert!(Transform::from_str("invalid").is_err());
    }
}
