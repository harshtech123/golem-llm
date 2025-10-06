use golem_vector::config::{get_max_retries_config, get_timeout_config};
use golem_vector::golem::vector::types::VectorError;
use log::trace;
use reqwest::{Client, Method, RequestBuilder, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;

/// PostgreSQL Vector (pgvector) HTTP API client
#[derive(Clone)]
pub struct PgVectorClient {
    base_url: String,
    api_key: Option<String>,
    client: Client,
}

impl PgVectorClient {
    pub fn new(
        base_url: String,
        api_key: Option<String>,
    ) -> Self {
        let timeout = get_timeout_config();
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()
            .unwrap();

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            client,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn build_request(&self, method: Method, endpoint: &str) -> RequestBuilder {
        let url = format!("{}/{}", self.base_url, endpoint.trim_start_matches('/'));
        let mut builder = self.client.request(method, &url);
        
        if let Some(ref api_key) = self.api_key {
            builder = builder.header("Authorization", format!("Bearer {}", api_key));
        }
        
        builder.header("Content-Type", "application/json")
    }

    fn parse_response<T: for<'de> Deserialize<'de>>(&self, response: Response) -> Result<T, VectorError> {
        let status = response.status();
        
        if !status.is_success() {
            let error_text = response.text().map_err(|e| {
                VectorError::ProviderError(format!("Failed to read error response: {}", e))
            })?;
            
            return Err(VectorError::ProviderError(format!(
                "HTTP {} error: {}", status, error_text
            )));
        }

        let response_text = response.text().map_err(|e| {
            VectorError::ProviderError(format!("Failed to read response: {}", e))
        })?;

        trace!("Response: {}", response_text);

        serde_json::from_str(&response_text).map_err(|e| {
            VectorError::ProviderError(format!("Failed to parse JSON response: {}", e))
        })
    }

    fn calculate_backoff_delay(attempt: u32, is_rate_limited: bool) -> Duration {
        let base_delay = if is_rate_limited { 1000 } else { 100 };
        let delay_ms = base_delay * (2_u64.pow(attempt.min(8)));
        Duration::from_millis(delay_ms)
    }

    fn execute_with_retry<F, T>(&self, operation: F) -> Result<T, VectorError>
    where
        F: Fn() -> Result<T, VectorError>,
    {
        let max_retries = get_max_retries_config();
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match operation() {
                Ok(result) => {
                    trace!("HTTP operation succeeded on attempt {}", attempt + 1);
                    return Ok(result);
                }
                Err(e) => {
                    trace!("HTTP operation failed on attempt {}: {}", attempt + 1, e);
                    last_error = Some(e);
                    
                    if attempt < max_retries {
                        let is_rate_limited = matches!(&last_error, 
                            Some(VectorError::ProviderError(msg)) if msg.contains("429")
                        );
                        std::thread::sleep(Self::calculate_backoff_delay(attempt, is_rate_limited));
                        continue;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| VectorError::ProviderError("Max retries exceeded".to_string())))
    }

    pub fn enable_extension(&self) -> Result<(), VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::POST, "/extensions/vector")
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            let _: Value = self.parse_response(response)?;
            Ok(())
        })
    }

    pub fn create_table(&self, request: &CreateTableRequest) -> Result<CreateTableResponse, VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::POST, "/tables")
                .json(request)
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            self.parse_response(response)
        })
    }

    pub fn drop_table(&self, table_name: &str) -> Result<DropTableResponse, VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::DELETE, &format!("/tables/{}", table_name))
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            self.parse_response(response)
        })
    }

    pub fn table_exists(&self, table_name: &str) -> Result<TableExistsResponse, VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::GET, &format!("/tables/{}/exists", table_name))
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            self.parse_response(response)
        })
    }

    pub fn upsert_vectors(&self, request: &UpsertVectorsRequest) -> Result<UpsertVectorsResponse, VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::PUT, "/vectors")
                .json(request)
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            self.parse_response(response)
        })
    }

    pub fn search_vectors(&self, request: &SearchRequest) -> Result<SearchResponse, VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::POST, "/search")
                .json(request)
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            self.parse_response(response)
        })
    }

    pub fn get_vectors(&self, request: &GetVectorsRequest) -> Result<GetVectorsResponse, VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::POST, "/vectors/get")
                .json(request)
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            self.parse_response(response)
        })
    }

    pub fn delete_vectors(&self, request: &DeleteVectorsRequest) -> Result<DeleteVectorsResponse, VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::DELETE, "/vectors")
                .json(request)
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            self.parse_response(response)
        })
    }

    pub fn count_vectors(&self, table_name: &str) -> Result<CountVectorsResponse, VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::GET, &format!("/tables/{}/count", table_name))
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            self.parse_response(response)
        })
    }

    pub fn list_tables(&self) -> Result<ListTablesResponse, VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::GET, "/tables")
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            self.parse_response(response)
        })
    }

    pub fn describe_table(&self, table_name: &str) -> Result<DescribeTableResponse, VectorError> {
        self.execute_with_retry(|| {
            let response = self.build_request(Method::GET, &format!("/tables/{}", table_name))
                .send()
                .map_err(|e| VectorError::ProviderError(format!("Failed to send request: {}", e)))?;
            
            self.parse_response(response)
        })
    }
}

//req/res structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTableRequest {
    pub table_name: String,
    pub dimension: Option<u32>,
    pub metadata_columns: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTableResponse {
    pub table_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropTableResponse {
    pub table_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableExistsResponse {
    pub exists: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateIndexRequest {
    pub index_name: String,
    pub table_name: String,
    pub column_name: String,
    pub index_type: String, 
    pub distance_metric: String, 
    pub index_options: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateIndexResponse {
    pub index_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropIndexResponse {
    pub index_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorData {
    pub id: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertVectorsRequest {
    pub table_name: String,
    pub vectors: Vec<VectorData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertVectorsResponse {
    pub inserted_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertVectorsRequest {
    pub table_name: String,
    pub vectors: Vec<VectorData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertVectorsResponse {
    pub upserted_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub table_name: String,
    pub query_vector: Vec<f32>,
    pub distance_metric: String,
    pub limit: i32,
    pub filters: HashMap<String, String>,
    pub select_columns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub embedding: Vec<f32>,
    pub distance: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetVectorsRequest {
    pub table_name: String,
    pub ids: Vec<String>,
    pub select_columns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorResult {
    pub id: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetVectorsResponse {
    pub results: Vec<VectorResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteVectorsRequest {
    pub table_name: String,
    pub ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteVectorsResponse {
    pub deleted_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountVectorsResponse {
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTablesResponse {
    pub tables: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableColumn {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeTableResponse {
    pub table_name: String,
    pub columns: Vec<TableColumn>,
}

// helper functions

impl Default for PgVectorClient {
    fn default() -> Self {
        Self::new(
            "http://localhost:3000".to_string(),
            None,
        )
    }
}
