use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::time::interval;
use warp::Filter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEvent {
    pub timestamp: u64,
    pub event_type: String,
    pub value: f64,
    pub tags: HashMap<String, String>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    pub name: String,
    pub total_events: usize,
    pub average_value: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub last_event_timestamp: u64,
    pub events_per_minute: f64,
    pub percentiles: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_connections: usize,
    pub total_requests: usize,
    pub error_rate: f64,
    pub average_response_time: f64,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformMetrics {
    pub transform_name: String,
    pub usage_count: usize,
    pub success_rate: f64,
    pub average_processing_time: f64,
    pub error_count: usize,
    pub token_throughput: f64,
    pub popular_input_lengths: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub system_metrics: SystemMetrics,
    pub transform_metrics: Vec<TransformMetrics>,
    pub recent_events: Vec<MetricEvent>,
    pub time_series_data: HashMap<String, Vec<(u64, f64)>>,
    pub alerts: Vec<Alert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub level: AlertLevel,
    pub message: String,
    pub timestamp: u64,
    pub resolved: bool,
    pub metric_name: String,
    pub threshold_value: f64,
    pub actual_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

pub struct MetricsCollector {
    events: Arc<Mutex<Vec<MetricEvent>>>,
    metrics_summaries: Arc<Mutex<HashMap<String, MetricSummary>>>,
    system_metrics: Arc<Mutex<SystemMetrics>>,
    transform_metrics: Arc<Mutex<HashMap<String, TransformMetrics>>>,
    alerts: Arc<Mutex<Vec<Alert>>>,
    start_time: Instant,
    alert_thresholds: HashMap<String, f64>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert("cpu_usage".to_string(), 80.0);
        alert_thresholds.insert("memory_usage".to_string(), 85.0);
        alert_thresholds.insert("error_rate".to_string(), 5.0);
        alert_thresholds.insert("response_time".to_string(), 1000.0);
        
        MetricsCollector {
            events: Arc::new(Mutex::new(Vec::new())),
            metrics_summaries: Arc::new(Mutex::new(HashMap::new())),
            system_metrics: Arc::new(Mutex::new(SystemMetrics {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                active_connections: 0,
                total_requests: 0,
                error_rate: 0.0,
                average_response_time: 0.0,
                uptime_seconds: 0,
            })),
            transform_metrics: Arc::new(Mutex::new(HashMap::new())),
            alerts: Arc::new(Mutex::new(Vec::new())),
            start_time: Instant::now(),
            alert_thresholds,
        }
    }
    
    pub fn record_event(&self, event_type: String, value: f64, tags: HashMap<String, String>) {
        let event = MetricEvent {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            event_type: event_type.clone(),
            value,
            tags,
            metadata: serde_json::json!({}),
        };
        
        // Store event
        if let Ok(mut events) = self.events.lock() {
            events.push(event.clone());
            
            // Keep only last 1000 events
            if events.len() > 1000 {
                events.drain(0..events.len() - 1000);
            }
        }
        
        // Update metrics summary
        self.update_metric_summary(&event_type, value, event.timestamp);
        
        // Check for alerts
        self.check_alerts(&event_type, value);
    }
    
    fn update_metric_summary(&self, metric_name: &str, value: f64, timestamp: u64) {
        if let Ok(mut summaries) = self.metrics_summaries.lock() {
            let summary = summaries.entry(metric_name.to_string()).or_insert(MetricSummary {
                name: metric_name.to_string(),
                total_events: 0,
                average_value: 0.0,
                min_value: value,
                max_value: value,
                last_event_timestamp: timestamp,
                events_per_minute: 0.0,
                percentiles: HashMap::new(),
            });
            
            // Update basic stats
            summary.total_events += 1;
            summary.average_value = (summary.average_value * (summary.total_events - 1) as f64 + value) / summary.total_events as f64;
            summary.min_value = summary.min_value.min(value);
            summary.max_value = summary.max_value.max(value);
            summary.last_event_timestamp = timestamp;
            
            // Calculate events per minute (simplified)
            let time_diff = timestamp.saturating_sub(summary.last_event_timestamp) as f64 / 60000.0; // Convert to minutes
            if time_diff > 0.0 {
                summary.events_per_minute = 1.0 / time_diff;
            }
        }
    }
    
    fn check_alerts(&self, metric_name: &str, value: f64) {
        if let Some(&threshold) = self.alert_thresholds.get(metric_name) {
            if value > threshold {
                let alert = Alert {
                    id: uuid::Uuid::new_v4().to_string(),
                    level: if value > threshold * 1.5 {
                        AlertLevel::Critical
                    } else if value > threshold * 1.2 {
                        AlertLevel::Error
                    } else {
                        AlertLevel::Warning
                    },
                    message: format!("{} is above threshold: {:.2} > {:.2}", metric_name, value, threshold),
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                    resolved: false,
                    metric_name: metric_name.to_string(),
                    threshold_value: threshold,
                    actual_value: value,
                };
                
                if let Ok(mut alerts) = self.alerts.lock() {
                    alerts.push(alert);
                    
                    // Keep only last 100 alerts
                    if alerts.len() > 100 {
                        alerts.drain(0..alerts.len() - 100);
                    }
                }
            }
        }
    }
    
    pub fn record_transform_usage(&self, transform_name: &str, processing_time: f64, success: bool, token_count: usize) {
        if let Ok(mut metrics) = self.transform_metrics.lock() {
            let transform_metric = metrics.entry(transform_name.to_string()).or_insert(TransformMetrics {
                transform_name: transform_name.to_string(),
                usage_count: 0,
                success_rate: 100.0,
                average_processing_time: 0.0,
                error_count: 0,
                token_throughput: 0.0,
                popular_input_lengths: Vec::new(),
            });
            
            transform_metric.usage_count += 1;
            
            if success {
                transform_metric.average_processing_time = 
                    (transform_metric.average_processing_time * (transform_metric.usage_count - 1) as f64 + processing_time) 
                    / transform_metric.usage_count as f64;
                transform_metric.token_throughput = token_count as f64 / (processing_time / 1000.0); // tokens per second
            } else {
                transform_metric.error_count += 1;
            }
            
            let successful_operations = transform_metric.usage_count - transform_metric.error_count;
            transform_metric.success_rate = (successful_operations as f64 / transform_metric.usage_count as f64) * 100.0;
            
            // Track popular input lengths
            transform_metric.popular_input_lengths.push(token_count);
            if transform_metric.popular_input_lengths.len() > 100 {
                transform_metric.popular_input_lengths.drain(0..1);
            }
        }
    }
    
    pub fn update_system_metrics(&self) {
        if let Ok(mut metrics) = self.system_metrics.lock() {
            // Simulate system metrics collection
            // In a real implementation, you'd use system APIs
            metrics.cpu_usage = self.get_cpu_usage();
            metrics.memory_usage = self.get_memory_usage();
            metrics.uptime_seconds = self.start_time.elapsed().as_secs();
            
            // Record system metrics as events
            self.record_event("cpu_usage".to_string(), metrics.cpu_usage, HashMap::new());
            self.record_event("memory_usage".to_string(), metrics.memory_usage, HashMap::new());
        }
    }
    
    fn get_cpu_usage(&self) -> f64 {
        // Simplified CPU usage simulation
        // In reality, you'd use system APIs like sysinfo crate
        rand::random::<f64>() * 100.0
    }
    
    fn get_memory_usage(&self) -> f64 {
        // Simplified memory usage simulation
        rand::random::<f64>() * 100.0
    }
    
    pub fn get_dashboard_data(&self) -> DashboardData {
        let system_metrics = self.system_metrics.lock().unwrap().clone();
        let transform_metrics: Vec<TransformMetrics> = self.transform_metrics.lock().unwrap().values().cloned().collect();
        let recent_events: Vec<MetricEvent> = self.events.lock().unwrap().iter().rev().take(50).cloned().collect();
        let alerts: Vec<Alert> = self.alerts.lock().unwrap().iter().filter(|a| !a.resolved).cloned().collect();
        
        // Generate time series data
        let mut time_series_data = HashMap::new();
        for event in self.events.lock().unwrap().iter() {
            let series = time_series_data.entry(event.event_type.clone()).or_insert(Vec::new());
            series.push((event.timestamp, event.value));
        }
        
        DashboardData {
            system_metrics,
            transform_metrics,
            recent_events,
            time_series_data,
            alerts,
        }
    }
    
    pub async fn start_background_collection(&self) {
        let collector = Arc::new(self.clone());
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                collector.update_system_metrics();
            }
        });
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
    fn clone(&self) -> Self {
        MetricsCollector {
            events: Arc::clone(&self.events),
            metrics_summaries: Arc::clone(&self.metrics_summaries),
            system_metrics: Arc::clone(&self.system_metrics),
            transform_metrics: Arc::clone(&self.transform_metrics),
            alerts: Arc::clone(&self.alerts),
            start_time: self.start_time,
            alert_thresholds: self.alert_thresholds.clone(),
        }
    }
}

// Web dashboard server
pub struct DashboardServer {
    metrics_collector: Arc<MetricsCollector>,
    port: u16,
}

impl DashboardServer {
    pub fn new(metrics_collector: Arc<MetricsCollector>, port: u16) -> Self {
        DashboardServer {
            metrics_collector,
            port,
        }
    }
    
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let metrics_collector = Arc::clone(&self.metrics_collector);
        
        // API Routes
        let api_metrics = warp::path("api")
            .and(warp::path("metrics"))
            .and(warp::path::end())
            .and(with_metrics(Arc::clone(&metrics_collector)))
            .and_then(get_metrics);
        
        let api_dashboard = warp::path("api")
            .and(warp::path("dashboard"))
            .and(warp::path::end())
            .and(with_metrics(Arc::clone(&metrics_collector)))
            .and_then(get_dashboard_data);
        
        let api_alerts = warp::path("api")
            .and(warp::path("alerts"))
            .and(warp::path::end())
            .and(with_metrics(Arc::clone(&metrics_collector)))
            .and_then(get_alerts);
        
        let api_export = warp::path("api")
            .and(warp::path("export"))
            .and(warp::path::end())
            .and(warp::query::<ExportQuery>())
            .and(with_metrics(Arc::clone(&metrics_collector)))
            .and_then(export_metrics);
        
        // Static files
        let static_files = warp::path("static")
            .and(warp::fs::dir("./dashboard/static"));
        
        let dashboard_html = warp::path::end()
            .and(warp::fs::file("./dashboard/index.html"));
        
        // WebSocket for real-time updates
        let websocket = warp::path("ws")
            .and(warp::ws())
            .and(with_metrics(Arc::clone(&metrics_collector)))
            .map(|ws: warp::ws::Ws, metrics: Arc<MetricsCollector>| {
                ws.on_upgrade(move |socket| handle_websocket(socket, metrics))
            });
        
        let routes = api_metrics
            .or(api_dashboard)
            .or(api_alerts)
            .or(api_export)
            .or(websocket)
            .or(static_files)
            .or(dashboard_html)
            .with(warp::cors().allow_any_origin());
        
        println!("🚀 Dashboard server starting on http://localhost:{}", self.port);
        warp::serve(routes).run(([127, 0, 0, 1], self.port)).await;
        
        Ok(())
    }
}

fn with_metrics(metrics: Arc<MetricsCollector>) -> impl Filter<Extract = (Arc<MetricsCollector>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || Arc::clone(&metrics))
}

async fn get_metrics(metrics: Arc<MetricsCollector>) -> Result<impl warp::Reply, warp::Rejection> {
    let summaries = metrics.metrics_summaries.lock().unwrap().clone();
    Ok(warp::reply::json(&summaries))
}

async fn get_dashboard_data(metrics: Arc<MetricsCollector>) -> Result<impl warp::Reply, warp::Rejection> {
    let dashboard_data = metrics.get_dashboard_data();
    Ok(warp::reply::json(&dashboard_data))
}

async fn get_alerts(metrics: Arc<MetricsCollector>) -> Result<impl warp::Reply, warp::Rejection> {
    let alerts = metrics.alerts.lock().unwrap().clone();
    Ok(warp::reply::json(&alerts))
}

#[derive(Debug, Deserialize)]
struct ExportQuery {
    format: Option<String>,
    start_time: Option<u64>,
    end_time: Option<u64>,
}

async fn export_metrics(query: ExportQuery, metrics: Arc<MetricsCollector>) -> Result<impl warp::Reply, warp::Rejection> {
    let format = query.format.unwrap_or_else(|| "json".to_string());
    let events = metrics.events.lock().unwrap();
    
    // Filter by time range if provided
    let filtered_events: Vec<&MetricEvent> = events.iter()
        .filter(|event| {
            if let Some(start) = query.start_time {
                if event.timestamp < start {
                    return false;
                }
            }
            if let Some(end) = query.end_time {
                if event.timestamp > end {
                    return false;
                }
            }
            true
        })
        .collect();
    
    match format.as_str() {
        "csv" => {
            let mut csv = String::from("timestamp,event_type,value,tags\n");
            for event in filtered_events {
                let tags_str = serde_json::to_string(&event.tags).unwrap_or_default();
                csv.push_str(&format!("{},{},{},{}\n", 
                    event.timestamp, event.event_type, event.value, tags_str));
            }
            Ok(warp::reply::with_header(csv, "content-type", "text/csv"))
        }
        _ => {
            let json = serde_json::to_string_pretty(&filtered_events).unwrap();
            Ok(warp::reply::with_header(json, "content-type", "application/json"))
        }
    }
}

async fn handle_websocket(ws: warp::ws::WebSocket, metrics: Arc<MetricsCollector>) {
    use futures_util::{SinkExt, StreamExt};
    use tokio::time::{interval, Duration};
    
    let (mut ws_tx, mut ws_rx) = ws.split();
    
    // Send periodic updates
    let metrics_clone = Arc::clone(&metrics);
    let update_task = tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(5));
        
        loop {
            interval.tick().await;
            let dashboard_data = metrics_clone.get_dashboard_data();
            
            if let Ok(message) = serde_json::to_string(&dashboard_data) {
                if ws_tx.send(warp::ws::Message::text(message)).await.is_err() {
                    break;
                }
            }
        }
    });
    
    // Handle incoming messages
    while let Some(result) = ws_rx.next().await {
        match result {
            Ok(msg) => {
                if msg.is_close() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    
    update_task.abort();
}

// Dashboard HTML generator
pub fn generate_dashboard_html() -> String {
    r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Every Other Token - Metrics Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #1a1a1a;
            color: #ffffff;
            line-height: 1.6;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #444;
        }
        .metric-title {
            font-size: 1.1em;
            color: #64b5f6;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #4caf50;
        }
        .metric-unit {
            font-size: 0.8em;
            color: #888;
        }
        .chart-container {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .alerts {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
        }
        .alert {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert.warning { border-color: #ff9800; background: rgba(255, 152, 0, 0.1); }
        .alert.error { border-color: #f44336; background: rgba(244, 67, 54, 0.1); }
        .alert.critical { border-color: #d32f2f; background: rgba(211, 47, 47, 0.2); }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background: #4caf50; }
        .status-warning { background: #ff9800; }
        .status-error { background: #f44336; }
        .transform-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .transform-card {
            background: #333;
            padding: 15px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔮 Every Other Token - Metrics Dashboard</h1>
            <p>Real-time monitoring and analytics</p>
            <div id="connection-status">
                <span class="status-indicator status-online"></span>
                Connected
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">CPU Usage</div>
                <div class="metric-value" id="cpu-usage">0<span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Memory Usage</div>
                <div class="metric-value" id="memory-usage">0<span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Total Requests</div>
                <div class="metric-value" id="total-requests">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Error Rate</div>
                <div class="metric-value" id="error-rate">0<span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Avg Response Time</div>
                <div class="metric-value" id="response-time">0<span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Uptime</div>
                <div class="metric-value" id="uptime">0<span class="metric-unit">s</span></div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>System Performance</h3>
            <canvas id="performance-chart"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Transform Usage</h3>
            <canvas id="transform-chart"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Transform Performance</h3>
            <div class="transform-list" id="transform-list"></div>
        </div>
        
        <div class="alerts">
            <h3>Active Alerts</h3>
            <div id="alerts-container">
                <p>No active alerts</p>
            </div>
        </div>
    </div>
    
    <script>
        class Dashboard {
            constructor() {
                this.ws = null;
                this.charts = {};
                this.initWebSocket();
                this.initCharts();
            }
            
            initWebSocket() {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                this.ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.updateConnectionStatus('online');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.updateDashboard(data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.updateConnectionStatus('error');
                    setTimeout(() => this.initWebSocket(), 5000);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus('error');
                };
            }
            
            updateConnectionStatus(status) {
                const indicator = document.querySelector('.status-indicator');
                const statusText = document.querySelector('#connection-status');
                
                indicator.className = `status-indicator status-${status}`;
                statusText.innerHTML = `<span class="status-indicator status-${status}"></span>${status === 'online' ? 'Connected' : 'Disconnected'}`;
            }
            
            initCharts() {
                const performanceCtx = document.getElementById('performance-chart').getContext('2d');
                this.charts.performance = new Chart(performanceCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            {
                                label: 'CPU Usage (%)',
                                data: [],
                                borderColor: '#ff6384',
                                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                tension: 0.4
                            },
                            {
                                label: 'Memory Usage (%)',
                                data: [],
                                borderColor: '#36a2eb',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                tension: 0.4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { labels: { color: '#ffffff' } } },
                        scales: {
                            x: { ticks: { color: '#ffffff' }, grid: { color: '#444' } },
                            y: { ticks: { color: '#ffffff' }, grid: { color: '#444' } }
                        }
                    }
                });
                
                const transformCtx = document.getElementById('transform-chart').getContext('2d');
                this.charts.transform = new Chart(transformCtx, {
                    type: 'doughnut',
                    data: {
                        labels: [],
                        datasets: [{
                            data: [],
                            backgroundColor: [
                                '#ff6384', '#36a2eb', '#ffcd56', '#4bc0c0',
                                '#9966ff', '#ff9f40', '#ff6384', '#c9cbcf'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { labels: { color: '#ffffff' } } }
                    }
                });
            }
            
            updateDashboard(data) {
                // Update system metrics
                document.getElementById('cpu-usage').innerHTML = `${data.system_metrics.cpu_usage.toFixed(1)}<span class="metric-unit">%</span>`;
                document.getElementById('memory-usage').innerHTML = `${data.system_metrics.memory_usage.toFixed(1)}<span class="metric-unit">%</span>`;
                document.getElementById('total-requests').textContent = data.system_metrics.total_requests;
                document.getElementById('error-rate').innerHTML = `${data.system_metrics.error_rate.toFixed(1)}<span class="metric-unit">%</span>`;
                document.getElementById('response-time').innerHTML = `${data.system_metrics.average_response_time.toFixed(0)}<span class="metric-unit">ms</span>`;
                document.getElementById('uptime').innerHTML = `${data.system_metrics.uptime_seconds}<span class="metric-unit">s</span>`;
                
                // Update performance chart
                this.updatePerformanceChart(data);
                
                // Update transform usage chart
                this.updateTransformChart(data.transform_metrics);
                
                // Update transform performance list
                this.updateTransformList(data.transform_metrics);
                
                // Update alerts
                this.updateAlerts(data.alerts);
            }
            
            updatePerformanceChart(data) {
                const chart = this.charts.performance;
                const now = new Date().toLocaleTimeString();
                
                chart.data.labels.push(now);
                chart.data.datasets[0].data.push(data.system_metrics.cpu_usage);
                chart.data.datasets[1].data.push(data.system_metrics.memory_usage);
                
                // Keep only last 20 data points
                if (chart.data.labels.length > 20) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                    chart.data.datasets[1].data.shift();
                }
                
                chart.update('none');
            }
            
            updateTransformChart(transformMetrics) {
                const chart = this.charts.transform;
                chart.data.labels = transformMetrics.map(t => t.transform_name);
                chart.data.datasets[0].data = transformMetrics.map(t => t.usage_count);
                chart.update();
            }
            
            updateTransformList(transformMetrics) {
                const container = document.getElementById('transform-list');
                container.innerHTML = transformMetrics.map(transform => `
                    <div class="transform-card">
                        <h4>${transform.transform_name}</h4>
                        <p>Usage: ${transform.usage_count}</p>
                        <p>Success Rate: ${transform.success_rate.toFixed(1)}%</p>
                        <p>Avg Time: ${transform.average_processing_time.toFixed(1)}ms</p>
                        <p>Throughput: ${transform.token_throughput.toFixed(1)} tokens/s</p>
                    </div>
                `).join('');
            }
            
            updateAlerts(alerts) {
                const container = document.getElementById('alerts-container');
                
                if (alerts.length === 0) {
                    container.innerHTML = '<p>No active alerts</p>';
                    return;
                }
                
                container.innerHTML = alerts.map(alert => `
                    <div class="alert ${alert.level.toLowerCase()}">
                        <strong>${alert.level.toUpperCase()}</strong>: ${alert.message}
                        <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                    </div>
                `).join('');
            }
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new Dashboard();
        });
    </script>
</body>
</html>
    "#.to_string()
}

// CLI for starting the dashboard
pub async fn run_dashboard_cli() -> Result<(), Box<dyn std::error::Error>> {
    let matches = clap::Command::new("dashboard")
        .version("1.0.0")
        .about("Start the metrics dashboard for Every Other Token")
        .arg(clap::Arg::new("port")
            .short('p')
            .long("port")
            .value_name("PORT")
            .default_value("3030")
            .help("Port to run the dashboard on"))
        .get_matches();
    
    let port: u16 = matches.get_one::<String>("port").unwrap().parse()?;
    
    // Create metrics collector and start background collection
    let metrics_collector = Arc::new(MetricsCollector::new());
    metrics_collector.start_background_collection().await;
    
    // Generate dashboard HTML file
    std::fs::create_dir_all("./dashboard")?;
    std::fs::write("./dashboard/index.html", generate_dashboard_html())?;
    
    // Start dashboard server
    let dashboard_server = DashboardServer::new(metrics_collector, port);
    dashboard_server.start().await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        
        // Record some test events
        let mut tags = HashMap::new();
        tags.insert("test".to_string(), "value".to_string());
        
        collector.record_event("test_metric".to_string(), 42.0, tags);
        collector.record_transform_usage("reverse", 150.0, true, 10);
        
        // Get dashboard data
        let dashboard_data = collector.get_dashboard_data();
        
        assert_eq!(dashboard_data.recent_events.len(), 1);
        assert_eq!(dashboard_data.transform_metrics.len(), 1);
        assert_eq!(dashboard_data.transform_metrics[0].transform_name, "reverse");
    }
    
    #[test]
    fn test_alert_generation() {
        let collector = MetricsCollector::new();
        
        // Record high CPU usage (should trigger alert)
        collector.record_event("cpu_usage".to_string(), 90.0, HashMap::new());
        
        let alerts = collector.alerts.lock().unwrap();
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].metric_name, "cpu_usage");
    }
}
