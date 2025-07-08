use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use serde_json;
use regex::Regex;

// Plugin trait for custom transformations
pub trait TransformPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn version(&self) -> &str;
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>>;
    fn configure(&mut self, config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub name: String,
    pub description: String,
    pub version: String,
    pub author: String,
    pub dependencies: Vec<String>,
    pub config_schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    pub enabled: bool,
    pub parameters: serde_json::Value,
}

pub struct CustomTransformManager {
    plugins: HashMap<String, Box<dyn TransformPlugin>>,
    configs: HashMap<String, TransformConfig>,
    plugin_directory: String,
}

impl CustomTransformManager {
    pub fn new(plugin_directory: String) -> Self {
        let mut manager = CustomTransformManager {
            plugins: HashMap::new(),
            configs: HashMap::new(),
            plugin_directory,
        };
        
        // Register built-in transforms
        manager.register_builtin_transforms();
        
        // Load custom plugins
        if let Err(e) = manager.load_plugins() {
            eprintln!("Warning: Failed to load some plugins: {}", e);
        }
        
        manager
    }
    
    fn register_builtin_transforms(&mut self) {
        self.register_plugin(Box::new(ReverseTransform::new()));
        self.register_plugin(Box::new(UppercaseTransform::new()));
        self.register_plugin(Box::new(MockTransform::new()));
        self.register_plugin(Box::new(NoiseTransform::new()));
        self.register_plugin(Box::new(LeetSpeakTransform::new()));
        self.register_plugin(Box::new(RandomCaseTransform::new()));
        self.register_plugin(Box::new(WordShuffleTransform::new()));
        self.register_plugin(Box::new(CharacterSubstitutionTransform::new()));
        self.register_plugin(Box::new(MorseCodeTransform::new()));
        self.register_plugin(Box::new(Base64Transform::new()));
    }
    
    pub fn register_plugin(&mut self, plugin: Box<dyn TransformPlugin>) {
        let name = plugin.name().to_string();
        self.plugins.insert(name.clone(), plugin);
        
        // Set default config
        self.configs.insert(name, TransformConfig {
            enabled: true,
            parameters: serde_json::json!({}),
        });
    }
    
    pub fn load_plugins(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let plugin_dir = Path::new(&self.plugin_directory);
        if !plugin_dir.exists() {
            fs::create_dir_all(plugin_dir)?;
            return Ok(());
        }
        
        for entry in fs::read_dir(plugin_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                match self.load_json_plugin(&path) {
                    Ok(_) => println!("✅ Loaded plugin: {:?}", path.file_name()),
                    Err(e) => eprintln!("❌ Failed to load plugin {:?}: {}", path.file_name(), e),
                }
            }
        }
        
        Ok(())
    }
    
    fn load_json_plugin(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let plugin_def: JsonPluginDefinition = serde_json::from_str(&content)?;
        
        let plugin = JsonPlugin::new(plugin_def)?;
        self.register_plugin(Box::new(plugin));
        
        Ok(())
    }
    
    pub fn transform(&self, input: &str, transform_name: &str) -> Result<String, Box<dyn std::error::Error>> {
        if let Some(config) = self.configs.get(transform_name) {
            if !config.enabled {
                return Err(format!("Transform '{}' is disabled", transform_name).into());
            }
        }
        
        if let Some(plugin) = self.plugins.get(transform_name) {
            plugin.transform(input)
        } else {
            Err(format!("Transform '{}' not found", transform_name).into())
        }
    }
    
    pub fn list_transforms(&self) -> Vec<PluginMetadata> {
        self.plugins.values().map(|plugin| {
            PluginMetadata {
                name: plugin.name().to_string(),
                description: plugin.description().to_string(),
                version: plugin.version().to_string(),
                author: "System".to_string(),
                dependencies: vec![],
                config_schema: None,
            }
        }).collect()
    }
    
    pub fn configure_transform(&mut self, name: &str, config: TransformConfig) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(plugin) = self.plugins.get_mut(name) {
            plugin.configure(config.parameters.clone())?;
            self.configs.insert(name.to_string(), config);
            Ok(())
        } else {
            Err(format!("Transform '{}' not found", name).into())
        }
    }
    
    pub fn create_plugin_template(&self, name: &str) -> String {
        serde_json::to_string_pretty(&JsonPluginDefinition {
            metadata: PluginMetadata {
                name: name.to_string(),
                description: "Custom transformation plugin".to_string(),
                version: "1.0.0".to_string(),
                author: "User".to_string(),
                dependencies: vec![],
                config_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "parameter1": {"type": "string", "default": "value1"}
                    }
                })),
            },
            transform_rules: vec![
                TransformRule {
                    rule_type: "regex".to_string(),
                    pattern: r"(\w+)".to_string(),
                    replacement: "$1_transformed".to_string(),
                    flags: vec!["global".to_string()],
                }
            ],
            config: serde_json::json!({}),
        }).unwrap_or_default()
    }
}

// Built-in transform implementations
pub struct ReverseTransform;
impl ReverseTransform {
    pub fn new() -> Self { ReverseTransform }
}
impl TransformPlugin for ReverseTransform {
    fn name(&self) -> &str { "reverse" }
    fn description(&self) -> &str { "Reverses the input string" }
    fn version(&self) -> &str { "1.0.0" }
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(input.chars().rev().collect())
    }
    fn configure(&mut self, _config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
}

pub struct UppercaseTransform;
impl UppercaseTransform {
    pub fn new() -> Self { UppercaseTransform }
}
impl TransformPlugin for UppercaseTransform {
    fn name(&self) -> &str { "uppercase" }
    fn description(&self) -> &str { "Converts input to uppercase" }
    fn version(&self) -> &str { "1.0.0" }
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(input.to_uppercase())
    }
    fn configure(&mut self, _config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
}

pub struct MockTransform;
impl MockTransform {
    pub fn new() -> Self { MockTransform }
}
impl TransformPlugin for MockTransform {
    fn name(&self) -> &str { "mock" }
    fn description(&self) -> &str { "Alternating case for mocking text" }
    fn version(&self) -> &str { "1.0.0" }
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(input.chars()
            .enumerate()
            .map(|(i, c)| {
                if i % 2 == 0 {
                    c.to_lowercase().next().unwrap_or(c)
                } else {
                    c.to_uppercase().next().unwrap_or(c)
                }
            })
            .collect())
    }
    fn configure(&mut self, _config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
}

pub struct NoiseTransform {
    noise_chars: Vec<char>,
}
impl NoiseTransform {
    pub fn new() -> Self {
        NoiseTransform {
            noise_chars: vec!['*', '+', '~', '@', '#', '$', '%'],
        }
    }
}
impl TransformPlugin for NoiseTransform {
    fn name(&self) -> &str { "noise" }
    fn description(&self) -> &str { "Adds random noise characters" }
    fn version(&self) -> &str { "1.0.0" }
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        let noise_char = self.noise_chars[rand::random::<usize>() % self.noise_chars.len()];
        Ok(format!("{}{}", input, noise_char))
    }
    fn configure(&mut self, config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(chars) = config.get("noise_chars").and_then(|v| v.as_str()) {
            self.noise_chars = chars.chars().collect();
        }
        Ok(())
    }
}

pub struct LeetSpeakTransform {
    substitutions: HashMap<char, char>,
}
impl LeetSpeakTransform {
    pub fn new() -> Self {
        let mut substitutions = HashMap::new();
        substitutions.insert('a', '@');
        substitutions.insert('e', '3');
        substitutions.insert('i', '1');
        substitutions.insert('o', '0');
        substitutions.insert('s', '$');
        substitutions.insert('t', '7');
        
        LeetSpeakTransform { substitutions }
    }
}
impl TransformPlugin for LeetSpeakTransform {
    fn name(&self) -> &str { "leet" }
    fn description(&self) -> &str { "Converts text to leet speak" }
    fn version(&self) -> &str { "1.0.0" }
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(input.chars()
            .map(|c| self.substitutions.get(&c.to_lowercase().next().unwrap_or(c)).unwrap_or(&c).clone())
            .collect())
    }
    fn configure(&mut self, config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(subs) = config.get("substitutions").and_then(|v| v.as_object()) {
            for (key, value) in subs {
                if let (Some(k), Some(v)) = (key.chars().next(), value.as_str().and_then(|s| s.chars().next())) {
                    self.substitutions.insert(k, v);
                }
            }
        }
        Ok(())
    }
}

pub struct RandomCaseTransform {
    probability: f64,
}
impl RandomCaseTransform {
    pub fn new() -> Self {
        RandomCaseTransform { probability: 0.5 }
    }
}
impl TransformPlugin for RandomCaseTransform {
    fn name(&self) -> &str { "random_case" }
    fn description(&self) -> &str { "Randomly changes case of characters" }
    fn version(&self) -> &str { "1.0.0" }
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(input.chars()
            .map(|c| {
                if rand::random::<f64>() < self.probability {
                    c.to_uppercase().next().unwrap_or(c)
                } else {
                    c.to_lowercase().next().unwrap_or(c)
                }
            })
            .collect())
    }
    fn configure(&mut self, config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(prob) = config.get("probability").and_then(|v| v.as_f64()) {
            self.probability = prob.clamp(0.0, 1.0);
        }
        Ok(())
    }
}

pub struct WordShuffleTransform;
impl WordShuffleTransform {
    pub fn new() -> Self { WordShuffleTransform }
}
impl TransformPlugin for WordShuffleTransform {
    fn name(&self) -> &str { "word_shuffle" }
    fn description(&self) -> &str { "Shuffles characters within words" }
    fn version(&self) -> &str { "1.0.0" }
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        let words: Vec<&str> = input.split_whitespace().collect();
        let transformed_words: Vec<String> = words.into_iter()
            .map(|word| {
                if word.len() <= 2 {
                    word.to_string()
                } else {
                    let mut chars: Vec<char> = word.chars().collect();
                    let middle = &mut chars[1..chars.len()-1];
                    middle.shuffle(&mut thread_rng());
                    chars.into_iter().collect()
                }
            })
            .collect();
        
        Ok(transformed_words.join(" "))
    }
    fn configure(&mut self, _config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
}

pub struct CharacterSubstitutionTransform {
    substitution_map: HashMap<char, String>,
}
impl CharacterSubstitutionTransform {
    pub fn new() -> Self {
        let mut substitution_map = HashMap::new();
        substitution_map.insert('a', "α".to_string());
        substitution_map.insert('b', "β".to_string());
        substitution_map.insert('g', "γ".to_string());
        substitution_map.insert('d', "δ".to_string());
        
        CharacterSubstitutionTransform { substitution_map }
    }
}
impl TransformPlugin for CharacterSubstitutionTransform {
    fn name(&self) -> &str { "char_sub" }
    fn description(&self) -> &str { "Substitutes characters with unicode equivalents" }
    fn version(&self) -> &str { "1.0.0" }
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(input.chars()
            .map(|c| {
                self.substitution_map.get(&c.to_lowercase().next().unwrap_or(c))
                    .cloned()
                    .unwrap_or_else(|| c.to_string())
            })
            .collect())
    }
    fn configure(&mut self, config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(map) = config.get("substitutions").and_then(|v| v.as_object()) {
            for (key, value) in map {
                if let (Some(k), Some(v)) = (key.chars().next(), value.as_str()) {
                    self.substitution_map.insert(k, v.to_string());
                }
            }
        }
        Ok(())
    }
}

pub struct MorseCodeTransform {
    morse_map: HashMap<char, &'static str>,
}
impl MorseCodeTransform {
    pub fn new() -> Self {
        let mut morse_map = HashMap::new();
        morse_map.insert('a', ".-");
        morse_map.insert('b', "-...");
        morse_map.insert('c', "-.-.");
        morse_map.insert('d', "-..");
        morse_map.insert('e', ".");
        morse_map.insert('f', "..-.");
        morse_map.insert('g', "--.");
        morse_map.insert('h', "....");
        morse_map.insert('i', "..");
        morse_map.insert('j', ".---");
        morse_map.insert('k', "-.-");
        morse_map.insert('l', ".-..");
        morse_map.insert('m', "--");
        morse_map.insert('n', "-.");
        morse_map.insert('o', "---");
        morse_map.insert('p', ".--.");
        morse_map.insert('q', "--.-");
        morse_map.insert('r', ".-.");
        morse_map.insert('s', "...");
        morse_map.insert('t', "-");
        morse_map.insert('u', "..-");
        morse_map.insert('v', "...-");
        morse_map.insert('w', ".--");
        morse_map.insert('x', "-..-");
        morse_map.insert('y', "-.--");
        morse_map.insert('z', "--..");
        morse_map.insert(' ', "/");
        
        MorseCodeTransform { morse_map }
    }
}
impl TransformPlugin for MorseCodeTransform {
    fn name(&self) -> &str { "morse" }
    fn description(&self) -> &str { "Converts text to morse code" }
    fn version(&self) -> &str { "1.0.0" }
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        let morse_result: Vec<String> = input.to_lowercase()
            .chars()
            .map(|c| {
                self.morse_map.get(&c)
                    .unwrap_or(&"?")
                    .to_string()
            })
            .collect();
        
        Ok(morse_result.join(" "))
    }
    fn configure(&mut self, _config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
}

pub struct Base64Transform;
impl Base64Transform {
    pub fn new() -> Self { Base64Transform }
}
impl TransformPlugin for Base64Transform {
    fn name(&self) -> &str { "base64" }
    fn description(&self) -> &str { "Encodes text to base64" }
    fn version(&self) -> &str { "1.0.0" }
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        use base64::{Engine as _, engine::general_purpose};
        Ok(general_purpose::STANDARD.encode(input))
    }
    fn configure(&mut self, _config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
}

// JSON-based plugin system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonPluginDefinition {
    pub metadata: PluginMetadata,
    pub transform_rules: Vec<TransformRule>,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformRule {
    pub rule_type: String, // "regex", "replace", "function"
    pub pattern: String,
    pub replacement: String,
    pub flags: Vec<String>,
}

pub struct JsonPlugin {
    metadata: PluginMetadata,
    rules: Vec<TransformRule>,
    config: serde_json::Value,
}

impl JsonPlugin {
    pub fn new(definition: JsonPluginDefinition) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(JsonPlugin {
            metadata: definition.metadata,
            rules: definition.transform_rules,
            config: definition.config,
        })
    }
}

impl TransformPlugin for JsonPlugin {
    fn name(&self) -> &str {
        &self.metadata.name
    }
    
    fn description(&self) -> &str {
        &self.metadata.description
    }
    
    fn version(&self) -> &str {
        &self.metadata.version
    }
    
    fn transform(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        let mut result = input.to_string();
        
        for rule in &self.rules {
            result = match rule.rule_type.as_str() {
                "regex" => {
                    let re = Regex::new(&rule.pattern)?;
                    re.replace_all(&result, &rule.replacement).to_string()
                }
                "replace" => {
                    result.replace(&rule.pattern, &rule.replacement)
                }
                "function" => {
                    // Simple function evaluation for basic operations
                    self.evaluate_function(&rule.pattern, &result)?
                }
                _ => result,
            };
        }
        
        Ok(result)
    }
    
    fn configure(&mut self, config: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        self.config = config;
        Ok(())
    }
}

impl JsonPlugin {
    fn evaluate_function(&self, function: &str, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        match function {
            "reverse" => Ok(input.chars().rev().collect()),
            "uppercase" => Ok(input.to_uppercase()),
            "lowercase" => Ok(input.to_lowercase()),
            "length" => Ok(input.len().to_string()),
            "rot13" => Ok(self.rot13(input)),
            _ => Ok(input.to_string()),
        }
    }
    
    fn rot13(&self, input: &str) -> String {
        input.chars()
            .map(|c| {
                match c {
                    'a'..='z' => ((c as u8 - b'a' + 13) % 26 + b'a') as char,
                    'A'..='Z' => ((c as u8 - b'A' + 13) % 26 + b'A') as char,
                    _ => c,
                }
            })
            .collect()
    }
}

// CLI for managing custom transforms
pub async fn run_transform_manager_cli() -> Result<(), Box<dyn std::error::Error>> {
    let matches = clap::Command::new("transform-manager")
        .version("1.0.0")
        .about("Manage custom transformations for Every Other Token")
        .subcommand(
            clap::Command::new("list")
                .about("List all available transforms"))
        .subcommand(
            clap::Command::new("test")
                .about("Test a transformation")
                .arg(clap::Arg::new("transform")
                    .help("Transform name")
                    .required(true))
                .arg(clap::Arg::new("input")
                    .help("Input text")
                    .required(true)))
        .subcommand(
            clap::Command::new("create")
                .about("Create a new plugin template")
                .arg(clap::Arg::new("name")
                    .help("Plugin name")
                    .required(true)))
        .subcommand(
            clap::Command::new("config")
                .about("Configure a transform")
                .arg(clap::Arg::new("transform")
                    .help("Transform name")
                    .required(true))
                .arg(clap::Arg::new("config")
                    .help("JSON configuration")
                    .required(true)))
        .get_matches();
    
    let mut manager = CustomTransformManager::new("./plugins".to_string());
    
    match matches.subcommand() {
        Some(("list", _)) => {
            let transforms = manager.list_transforms();
            println!("🔧 Available Transformations:");
            for transform in transforms {
                println!("  📦 {} v{} - {}", transform.name, transform.version, transform.description);
            }
        }
        Some(("test", sub_matches)) => {
            let transform_name = sub_matches.get_one::<String>("transform").unwrap();
            let input = sub_matches.get_one::<String>("input").unwrap();
            
            match manager.transform(input, transform_name) {
                Ok(result) => {
                    println!("📥 Input:  {}", input);
                    println!("📤 Output: {}", result);
                }
                Err(e) => println!("❌ Error: {}", e),
            }
        }
        Some(("create", sub_matches)) => {
            let name = sub_matches.get_one::<String>("name").unwrap();
            let template = manager.create_plugin_template(name);
            let filename = format!("./plugins/{}.json", name);
            
            std::fs::create_dir_all("./plugins")?;
            std::fs::write(&filename, template)?;
            println!("✅ Created plugin template: {}", filename);
        }
        Some(("config", sub_matches)) => {
            let transform_name = sub_matches.get_one::<String>("transform").unwrap();
            let config_str = sub_matches.get_one::<String>("config").unwrap();
            
            let config_json: serde_json::Value = serde_json::from_str(config_str)?;
            let transform_config = TransformConfig {
                enabled: true,
                parameters: config_json,
            };
            
            match manager.configure_transform(transform_name, transform_config) {
                Ok(_) => println!("✅ Configured transform: {}", transform_name),
                Err(e) => println!("❌ Error: {}", e),
            }
        }
        _ => {
            println!("Use --help for available commands");
        }
    }
    
    Ok(())
}

// Integration with main token interceptor
impl crate::TokenInterceptor {
    pub fn with_custom_transforms(mut self, plugin_directory: String) -> Self {
        let transform_manager = CustomTransformManager::new(plugin_directory);
        // Store the transform manager in the interceptor
        // This would require modifying the main TokenInterceptor struct
        self
    }
    
    pub fn apply_custom_transform(&self, token: &str, transform_name: &str) -> Result<String, Box<dyn std::error::Error>> {
        // This would use the stored transform manager
        // For now, we'll create a temporary one
        let manager = CustomTransformManager::new("./plugins".to_string());
        manager.transform(token, transform_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reverse_transform() {
        let transform = ReverseTransform::new();
        assert_eq!(transform.transform("hello").unwrap(), "olleh");
    }
    
    #[test]
    fn test_leet_transform() {
        let transform = LeetSpeakTransform::new();
        let result = transform.transform("hello").unwrap();
        assert!(result.contains("3")); // 'e' -> '3'
    }
    
    #[test]
    fn test_morse_transform() {
        let transform = MorseCodeTransform::new();
        let result = transform.transform("hi").unwrap();
        assert_eq!(result, ".... ..");
    }
    
    #[tokio::test]
    async fn test_custom_transform_manager() {
        let mut manager = CustomTransformManager::new("./test_plugins".to_string());
        
        // Test built-in transforms
        assert!(manager.transform("hello", "reverse").is_ok());
        assert!(manager.transform("hello", "uppercase").is_ok());
        
        // Test invalid transform
        assert!(manager.transform("hello", "nonexistent").is_err());
    }
    
    #[test]
    fn test_json_plugin() {
        let plugin_def = JsonPluginDefinition {
            metadata: PluginMetadata {
                name: "test_plugin".to_string(),
                description: "Test plugin".to_string(),
                version: "1.0.0".to_string(),
                author: "Test".to_string(),
                dependencies: vec![],
                config_schema: None,
            },
            transform_rules: vec![
                TransformRule {
                    rule_type: "replace".to_string(),
                    pattern: "hello".to_string(),
                    replacement: "hi".to_string(),
                    flags: vec![],
                }
            ],
            config: serde_json::json!({}),
        };
        
        let plugin = JsonPlugin::new(plugin_def).unwrap();
        assert_eq!(plugin.transform("hello world").unwrap(), "hi world");
    }
}
