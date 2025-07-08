use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};
use tokio::time::sleep;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    pub id: String,
    pub prompts: Vec<String>,
    pub transform: String,
    pub model: String,
    pub max_concurrent: usize,
    pub delay_between_requests: Duration,
    pub status: JobStatus,
    pub created_at: Instant,
    pub completed_at: Option<Instant>,
    pub results: Vec<BatchResult>,
    pub errors: Vec<BatchError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub prompt_index: usize,
    pub original_prompt: String,
    pub transformed_text: String,
    pub token_count: usize,
    pub transformed_token_count: usize,
    pub processing_time_ms: u64,
    pub model_used: String,
    pub transform_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchError {
    pub prompt_index: usize,
    pub error_message: String,
    pub error_type: String,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    pub max_concurrent_jobs: usize,
    pub max_concurrent_requests_per_job: usize,
    pub default_delay_ms: u64,
    pub retry_attempts: usize,
    pub timeout_seconds: u64,
    pub output_directory: String,
    pub auto_export: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            max_concurrent_jobs: 5,
            max_concurrent_requests_per_job: 10,
            default_delay_ms: 100,
            retry_attempts: 3,
            timeout_seconds: 300,
            output_directory: "./batch_results".to_string(),
            auto_export: true,
        }
    }
}

pub struct BatchProcessor {
    config: BatchConfig,
    jobs: HashMap<String, BatchJob>,
    job_queue: mpsc::UnboundedSender<String>,
    job_receiver: mpsc::UnboundedReceiver<String>,
    semaphore: Arc<Semaphore>,
    interceptor: Arc<crate::TokenInterceptor>,
}

impl BatchProcessor {
    pub fn new(config: BatchConfig, interceptor: crate::TokenInterceptor) -> Self {
        let (job_queue, job_receiver) = mpsc::unbounded_channel();
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_jobs));
        
        BatchProcessor {
            config,
            jobs: HashMap::new(),
            job_queue,
            job_receiver,
            semaphore,
            interceptor: Arc::new(interceptor),
        }
    }
    
    pub async fn submit_batch_job(
        &mut self,
        prompts: Vec<String>,
        transform: String,
        model: String,
        max_concurrent: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let job_id = Uuid::new_v4().to_string();
        
        let job = BatchJob {
            id: job_id.clone(),
            prompts,
            transform,
            model,
            max_concurrent: max_concurrent.unwrap_or(self.config.max_concurrent_requests_per_job),
            delay_between_requests: Duration::from_millis(self.config.default_delay_ms),
            status: JobStatus::Pending,
            created_at: Instant::now(),
            completed_at: None,
            results: Vec::new(),
            errors: Vec::new(),
        };
        
        self.jobs.insert(job_id.clone(), job);
        self.job_queue.send(job_id.clone())?;
        
        println!("✅ Batch job {} submitted with {} prompts", job_id, self.jobs[&job_id].prompts.len());
        Ok(job_id)
    }
    
    pub async fn submit_from_file(
        &mut self,
        file_path: &str,
        transform: String,
        model: String,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let prompts: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;
        
        self.submit_batch_job(prompts, transform, model, None).await
    }
    
    pub async fn process_jobs(&mut self) {
        while let Some(job_id) = self.job_receiver.recv().await {
            let permit = self.semaphore.clone().acquire_owned().await.unwrap();
            let job = self.jobs.get_mut(&job_id).unwrap();
            job.status = JobStatus::Running;
            
            let job_clone = job.clone();
            let interceptor = self.interceptor.clone();
            let config = self.config.clone();
            
            tokio::spawn(async move {
                let result = Self::execute_batch_job(job_clone, interceptor, config).await;
                drop(permit); // Release semaphore permit
                
                match result {
                    Ok(completed_job) => {
                        println!("✅ Batch job {} completed successfully", completed_job.id);
                    }
                    Err(e) => {
                        eprintln!("❌ Batch job {} failed: {}", job_id, e);
                    }
                }
            });
        }
    }
    
    async fn execute_batch_job(
        mut job: BatchJob,
        interceptor: Arc<crate::TokenInterceptor>,
        config: BatchConfig,
    ) -> Result<BatchJob, Box<dyn std::error::Error>> {
        let semaphore = Arc::new(Semaphore::new(job.max_concurrent));
        let mut tasks = Vec::new();
        
        for (index, prompt) in job.prompts.iter().enumerate() {
            let permit = semaphore.clone().acquire_owned().await?;
            let prompt = prompt.clone();
            let transform = job.transform.clone();
            let model = job.model.clone();
            let job_id = job.id.clone();
            let delay = job.delay_between_requests;
            
            let task = tokio::spawn(async move {
                let start_time = Instant::now();
                
                // Simulate API call with retry logic
                let mut attempts = 0;
                let result = loop {
                    attempts += 1;
                    
                    match Self::process_single_prompt(&prompt, &transform, &model).await {
                        Ok(result) => break Ok(result),
                        Err(e) if attempts < config.retry_attempts => {
                            eprintln!("Retry {} for prompt {}: {}", attempts, index, e);
                            sleep(Duration::from_millis(500 * attempts as u64)).await;
                            continue;
                        }
                        Err(e) => break Err(e),
                    }
                };
                
                sleep(delay).await; // Rate limiting
                drop(permit); // Release semaphore permit
                
                let processing_time = start_time.elapsed().as_millis() as u64;
                
                match result {
                    Ok((transformed_text, token_count, transformed_token_count)) => {
                        Ok(BatchResult {
                            prompt_index: index,
                            original_prompt: prompt,
                            transformed_text,
                            token_count,
                            transformed_token_count,
                            processing_time_ms: processing_time,
                            model_used: model,
                            transform_used: transform,
                        })
                    }
                    Err(e) => Err(BatchError {
                        prompt_index: index,
                        error_message: e.to_string(),
                        error_type: "ProcessingError".to_string(),
                        timestamp: Instant::now(),
                    })
                }
            });
            
            tasks.push(task);
        }
        
        // Collect results
        for task in tasks {
            match task.await? {
                Ok(result) => job.results.push(result),
                Err(error) => job.errors.push(error),
            }
        }
        
        job.status = if job.errors.is_empty() {
            JobStatus::Completed
        } else if job.results.is_empty() {
            JobStatus::Failed
        } else {
            JobStatus::Completed // Partial success
        };
        
        job.completed_at = Some(Instant::now());
        
        // Auto-export if enabled
        if config.auto_export {
            Self::export_job_results(&job, &config.output_directory).await?;
        }
        
        Ok(job)
    }
    
    async fn process_single_prompt(
        prompt: &str,
        transform: &str,
        model: &str,
    ) -> Result<(String, usize, usize), Box<dyn std::error::Error>> {
        // Simulate token processing (replace with actual API call)
        let tokens = Self::simple_tokenize(prompt);
        let mut transformed_text = String::new();
        let mut transformed_count = 0;
        
        for (i, token) in tokens.iter().enumerate() {
            let processed_token = if i % 2 == 0 {
                token.clone()
            } else {
                transformed_count += 1;
                Self::apply_transform(token, transform)
            };
            transformed_text.push_str(&processed_token);
        }
        
        // Simulate network delay
        sleep(Duration::from_millis(100 + rand::random::<u64>() % 200)).await;
        
        Ok((transformed_text, tokens.len(), transformed_count))
    }
    
    fn simple_tokenize(text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_string()).collect()
    }
    
    fn apply_transform(token: &str, transform: &str) -> String {
        match transform {
            "reverse" => token.chars().rev().collect(),
            "uppercase" => token.to_uppercase(),
            "mock" => token.chars()
                .enumerate()
                .map(|(i, c)| {
                    if i % 2 == 0 {
                        c.to_lowercase().next().unwrap_or(c)
                    } else {
                        c.to_uppercase().next().unwrap_or(c)
                    }
                })
                .collect(),
            "noise" => format!("{}*", token),
            _ => token.to_string(),
        }
    }
    
    async fn export_job_results(
        job: &BatchJob,
        output_dir: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(output_dir)?;
        
        // Export as JSON
        let json_path = format!("{}/batch_job_{}.json", output_dir, job.id);
        let json_content = serde_json::to_string_pretty(job)?;
        std::fs::write(json_path, json_content)?;
        
        // Export as CSV
        let csv_path = format!("{}/batch_job_{}.csv", output_dir, job.id);
        let mut csv_content = String::from("prompt_index,original_prompt,transformed_text,token_count,transformed_token_count,processing_time_ms,model,transform\n");
        
        for result in &job.results {
            csv_content.push_str(&format!(
                "{},{:?},{:?},{},{},{},{},{}\n",
                result.prompt_index,
                result.original_prompt,
                result.transformed_text,
                result.token_count,
                result.transformed_token_count,
                result.processing_time_ms,
                result.model_used,
                result.transform_used
            ));
        }
        
        std::fs::write(csv_path, csv_content)?;
        
        // Export errors if any
        if !job.errors.is_empty() {
            let errors_path = format!("{}/batch_job_{}_errors.json", output_dir, job.id);
            let errors_content = serde_json::to_string_pretty(&job.errors)?;
            std::fs::write(errors_path, errors_content)?;
        }
        
        Ok(())
    }
    
    pub fn get_job_status(&self, job_id: &str) -> Option<&BatchJob> {
        self.jobs.get(job_id)
    }
    
    pub fn list_jobs(&self) -> Vec<&BatchJob> {
        self.jobs.values().collect()
    }
    
    pub fn cancel_job(&mut self, job_id: &str) -> Result<(), String> {
        if let Some(job) = self.jobs.get_mut(job_id) {
            match job.status {
                JobStatus::Pending | JobStatus::Running => {
                    job.status = JobStatus::Cancelled;
                    Ok(())
                }
                _ => Err("Job cannot be cancelled in current state".to_string()),
            }
        } else {
            Err("Job not found".to_string())
        }
    }
    
    pub async fn wait_for_completion(&self, job_id: &str, timeout: Option<Duration>) -> Result<(), String> {
        let start = Instant::now();
        let timeout = timeout.unwrap_or(Duration::from_secs(self.config.timeout_seconds));
        
        loop {
            if let Some(job) = self.jobs.get(job_id) {
                match job.status {
                    JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled => {
                        return Ok(());
                    }
                    _ => {}
                }
            } else {
                return Err("Job not found".to_string());
            }
            
            if start.elapsed() > timeout {
                return Err("Timeout waiting for job completion".to_string());
            }
            
            sleep(Duration::from_millis(100)).await;
        }
    }
    
    pub fn get_batch_analytics(&self) -> BatchAnalytics {
        let mut analytics = BatchAnalytics::default();
        
        for job in self.jobs.values() {
            analytics.total_jobs += 1;
            analytics.total_prompts += job.prompts.len();
            analytics.total_results += job.results.len();
            analytics.total_errors += job.errors.len();
            
            match job.status {
                JobStatus::Completed => analytics.completed_jobs += 1,
                JobStatus::Failed => analytics.failed_jobs += 1,
                JobStatus::Running => analytics.running_jobs += 1,
                JobStatus::Cancelled => analytics.cancelled_jobs += 1,
                JobStatus::Pending => analytics.pending_jobs += 1,
            }
            
            if let Some(completed_at) = job.completed_at {
                let duration = completed_at.duration_since(job.created_at);
                analytics.total_processing_time += duration;
                analytics.average_processing_time = analytics.total_processing_time / analytics.completed_jobs as u32;
            }
        }
        
        analytics
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BatchAnalytics {
    pub total_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub running_jobs: usize,
    pub cancelled_jobs: usize,
    pub pending_jobs: usize,
    pub total_prompts: usize,
    pub total_results: usize,
    pub total_errors: usize,
    pub total_processing_time: Duration,
    pub average_processing_time: Duration,
}

// CLI integration for batch processing
pub async fn run_batch_cli() -> Result<(), Box<dyn std::error::Error>> {
    let matches = clap::Command::new("batch-processor")
        .version("1.0.0")
        .about("Batch processing for Every Other Token")
        .subcommand(
            clap::Command::new("submit")
                .about("Submit a batch job")
                .arg(clap::Arg::new("file")
                    .short('f')
                    .long("file")
                    .value_name("FILE")
                    .help("File containing prompts (one per line)")
                    .required(true))
                .arg(clap::Arg::new("transform")
                    .short('t')
                    .long("transform")
                    .default_value("reverse")
                    .help("Transform type"))
                .arg(clap::Arg::new("model")
                    .short('m')
                    .long("model")
                    .default_value("gpt-3.5-turbo")
                    .help("Model to use"))
                .arg(clap::Arg::new("concurrent")
                    .short('c')
                    .long("concurrent")
                    .value_name("NUM")
                    .help("Max concurrent requests")))
        .subcommand(
            clap::Command::new("status")
                .about("Check job status")
                .arg(clap::Arg::new("job-id")
                    .help("Job ID to check")
                    .required(true)))
        .subcommand(
            clap::Command::new("list")
                .about("List all jobs"))
        .subcommand(
            clap::Command::new("analytics")
                .about("Show batch processing analytics"))
        .get_matches();
    
    let config = BatchConfig::default();
    let interceptor = crate::TokenInterceptor::new(
        crate::Transform::Reverse,
        "gpt-3.5-turbo".to_string()
    )?;
    
    let mut processor = BatchProcessor::new(config, interceptor);
    
    // Start job processing in background
    tokio::spawn(async move {
        processor.process_jobs().await;
    });
    
    match matches.subcommand() {
        Some(("submit", sub_matches)) => {
            let file_path = sub_matches.get_one::<String>("file").unwrap();
            let transform = sub_matches.get_one::<String>("transform").unwrap().clone();
            let model = sub_matches.get_one::<String>("model").unwrap().clone();
            
            let job_id = processor.submit_from_file(file_path, transform, model).await?;
            println!("📋 Submitted batch job: {}", job_id);
        }
        Some(("status", sub_matches)) => {
            let job_id = sub_matches.get_one::<String>("job-id").unwrap();
            if let Some(job) = processor.get_job_status(job_id) {
                println!("📊 Job Status: {:?}", job.status);
                println!("📈 Progress: {}/{} completed", job.results.len(), job.prompts.len());
                if !job.errors.is_empty() {
                    println!("⚠️  Errors: {}", job.errors.len());
                }
            } else {
                println!("❌ Job not found: {}", job_id);
            }
        }
        Some(("list", _)) => {
            let jobs = processor.list_jobs();
            for job in jobs {
                println!("📋 {} - {:?} - {}/{} completed", 
                    job.id, job.status, job.results.len(), job.prompts.len());
            }
        }
        Some(("analytics", _)) => {
            let analytics = processor.get_batch_analytics();
            println!("📊 Batch Processing Analytics:");
            println!("   Total Jobs: {}", analytics.total_jobs);
            println!("   Completed: {}", analytics.completed_jobs);
            println!("   Failed: {}", analytics.failed_jobs);
            println!("   Running: {}", analytics.running_jobs);
            println!("   Total Prompts: {}", analytics.total_prompts);
            println!("   Average Processing Time: {:?}", analytics.average_processing_time);
        }
        _ => {
            println!("Use --help for available commands");
        }
    }
    
    Ok(())
}
