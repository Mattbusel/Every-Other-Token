use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{console, HtmlElement, HtmlInputElement, HtmlSelectElement, HtmlTextAreaElement};
use serde::{Deserialize, Serialize};
use js_sys::Promise;
use std::collections::HashMap;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformResult {
    pub original_token: String,
    pub transformed_token: String,
    pub position: usize,
    pub transform_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub transformed_text: String,
    pub total_tokens: usize,
    pub transformed_tokens: usize,
    pub transform_details: Vec<TransformResult>,
    pub processing_time_ms: f64,
}

#[wasm_bindgen]
pub struct WebTokenInterceptor {
    transforms: HashMap<String, Box<dyn Fn(&str) -> String>>,
    current_transform: String,
    processing_results: Vec<ProcessingResult>,
}

#[wasm_bindgen]
impl WebTokenInterceptor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebTokenInterceptor {
        console_log!("Initializing WebTokenInterceptor");
        
        let mut transforms: HashMap<String, Box<dyn Fn(&str) -> String>> = HashMap::new();
        
        // Add built-in transforms
        transforms.insert("reverse".to_string(), Box::new(|s: &str| {
            s.chars().rev().collect()
        }));
        
        transforms.insert("uppercase".to_string(), Box::new(|s: &str| {
            s.to_uppercase()
        }));
        
        transforms.insert("mock".to_string(), Box::new(|s: &str| {
            s.chars()
                .enumerate()
                .map(|(i, c)| {
                    if i % 2 == 0 {
                        c.to_lowercase().next().unwrap_or(c)
                    } else {
                        c.to_uppercase().next().unwrap_or(c)
                    }
                })
                .collect()
        }));
        
        transforms.insert("noise".to_string(), Box::new(|s: &str| {
            let noise_chars = ['*', '+', '~', '@', '#', '$', '%'];
            let noise_char = noise_chars[js_sys::Math::random() as usize % noise_chars.len()];
            format!("{}{}", s, noise_char)
        }));
        
        WebTokenInterceptor {
            transforms,
            current_transform: "reverse".to_string(),
            processing_results: Vec::new(),
        }
    }
    
    #[wasm_bindgen]
    pub fn set_transform(&mut self, transform_name: &str) {
        if self.transforms.contains_key(transform_name) {
            self.current_transform = transform_name.to_string();
            console_log!("Transform set to: {}", transform_name);
        } else {
            console_log!("Unknown transform: {}", transform_name);
        }
    }
    
    #[wasm_bindgen]
    pub fn add_custom_transform(&mut self, name: &str, js_function: &js_sys::Function) {
        let name = name.to_string();
        let func = js_function.clone();
        
        self.transforms.insert(name.clone(), Box::new(move |input: &str| {
            let js_input = JsValue::from_str(input);
            match func.call1(&JsValue::NULL, &js_input) {
                Ok(result) => result.as_string().unwrap_or_else(|| input.to_string()),
                Err(_) => input.to_string(),
            }
        }));
        
        console_log!("Added custom transform: {}", name);
    }
    
    #[wasm_bindgen]
    pub fn process_text(&mut self, input: &str) -> JsValue {
        let start_time = js_sys::Date::now();
        
        let tokens = self.tokenize(input);
        let mut transformed_text = String::new();
        let mut transform_details = Vec::new();
        let mut transformed_count = 0;
        
        for (i, token) in tokens.iter().enumerate() {
            if token.trim().is_empty() {
                transformed_text.push_str(token);
                continue;
            }
            
            let processed_token = if i % 2 == 0 {
                // Even tokens - pass through unchanged
                token.clone()
            } else {
                // Odd tokens - apply transformation
                transformed_count += 1;
                if let Some(transform_fn) = self.transforms.get(&self.current_transform) {
                    let transformed = transform_fn(token);
                    transform_details.push(TransformResult {
                        original_token: token.clone(),
                        transformed_token: transformed.clone(),
                        position: i,
                        transform_type: self.current_transform.clone(),
                    });
                    transformed
                } else {
                    token.clone()
                }
            };
            
            transformed_text.push_str(&processed_token);
        }
        
        let processing_time = js_sys::Date::now() - start_time;
        
        let result = ProcessingResult {
            transformed_text,
            total_tokens: tokens.len(),
            transformed_tokens: transformed_count,
            transform_details,
            processing_time_ms: processing_time,
        };
        
        self.processing_results.push(result.clone());
        
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }
    
    #[wasm_bindgen]
    pub fn get_available_transforms(&self) -> JsValue {
        let transforms: Vec<String> = self.transforms.keys().cloned().collect();
        serde_wasm_bindgen::to_value(&transforms).unwrap_or(JsValue::NULL)
    }
    
    #[wasm_bindgen]
    pub fn get_processing_history(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.processing_results).unwrap_or(JsValue::NULL)
    }
    
    #[wasm_bindgen]
    pub fn export_results(&self, format: &str) -> String {
        match format {
            "json" => serde_json::to_string_pretty(&self.processing_results).unwrap_or_default(),
            "csv" => self.export_as_csv(),
            _ => String::new(),
        }
    }
    
    #[wasm_bindgen]
    pub fn clear_history(&mut self) {
        self.processing_results.clear();
        console_log!("Processing history cleared");
    }
    
    // Real-time processing with streaming simulation
    #[wasm_bindgen]
    pub fn process_stream(&self, input: &str, callback: &js_sys::Function) {
        let tokens = self.tokenize(input);
        let transform_name = self.current_transform.clone();
        let transform_fn = self.transforms.get(&transform_name).cloned();
        
        spawn_local(async move {
            for (i, token) in tokens.iter().enumerate() {
                if token.trim().is_empty() {
                    continue;
                }
                
                let processed_token = if i % 2 == 0 {
                    token.clone()
                } else if let Some(ref transform) = transform_fn {
                    transform(token)
                } else {
                    token.clone()
                };
                
                let token_result = serde_json::json!({
                    "token": processed_token,
                    "position": i,
                    "is_transformed": i % 2 == 1,
                    "original": token
                });
                
                let js_result = JsValue::from_str(&token_result.to_string());
                let _ = callback.call1(&JsValue::NULL, &js_result);
                
                // Simulate streaming delay
                let promise = Promise::new(&mut |resolve, _| {
                    web_sys::window()
                        .unwrap()
                        .set_timeout_with_callback_and_timeout_and_arguments_0(
                            &resolve,
                            50 + (js_sys::Math::random() * 100.0) as i32,
                        )
                        .unwrap();
                });
                
                let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
            }
        });
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
    
    fn export_as_csv(&self) -> String {
        let mut csv = String::from("timestamp,total_tokens,transformed_tokens,processing_time_ms,transform_type\n");
        
        for (i, result) in self.processing_results.iter().enumerate() {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                i,
                result.total_tokens,
                result.transformed_tokens,
                result.processing_time_ms,
                result.transform_details.first()
                    .map(|d| d.transform_type.as_str())
                    .unwrap_or("unknown")
            ));
        }
        
        csv
    }
}

// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn main() {
    console_log!("Every Other Token WASM module initialized");
}
