use golem_vector::config::{get_max_retries_config, get_timeout_config};
use golem_vector::golem::vector::types::VectorError;
use log::trace;
use reqwest::{Client, Method, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::collections::HashMap;
use std::time::Duration;

/// Milvus Vector API client
/// based on https://milvus.io/docs
#[derive(Clone)]
pub struct MilvusClient {
    client: Client,
    base_url: String,
    token: Option<String>,
    database: String,
}

impl MilvusClient {
    pub fn new(uri: String, token: Option<String>, database: Option<String>) -> Self {
        let timeout_secs = get_timeout_config();
        
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .expect("Failed to initialize HTTP client");

        let base_url = if uri.ends_with('/') {
            uri.trim_end_matches('/').to_string()
        } else {
            uri
        };

        Self {
            client,
            base_url,
            token,
            database: database.unwrap_or_else(|| "_default".to_string()),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn database(&self) -> &str {
        &self.database
    }

    fn create_request(&self, method: Method, endpoint: &str) -> RequestBuilder {
        let url = format!("{}{}", self.base_url, endpoint);
        let mut request = self.client.request(method, &url)
            .header("accept", "application/json")
            .header("content-type", "application/json");

        if let Some(ref token) = self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        request
    }

    fn should_retry_error(&self, error: &reqwest::Error) -> bool {
        error.is_timeout() || error.is_request()
    }

    fn calculate_backoff_delay(attempt: u32, is_rate_limited: bool) -> Duration {
        let base_delay_ms = if is_rate_limited { 1000 } else { 200 };
        let max_delay_ms = 30000;

        let delay_ms = std::cmp::min(max_delay_ms, base_delay_ms * (2_u64.pow(attempt)));
        Duration::from_millis(delay_ms)
    }

    fn execute_with_retry_sync<F>(&self, operation: F) -> Result<Response, VectorError>
    where
        F: Fn() -> Result<Response, reqwest::Error> + Send + Sync,
    {
        let max_retries = get_max_retries_config();
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match operation() {
                Ok(response) => return Ok(response),
                Err(error) => {
                    trace!("Request attempt {} failed: {}", attempt, error);
                    let _error_string = error.to_string();
                    last_error = Some(error);

                    if attempt < max_retries && self.should_retry_error(last_error.as_ref().unwrap()) {
                        let is_rate_limited = last_error.as_ref()
                            .and_then(|e| e.status())
                            .map(|s| s.as_u16() == 429)
                            .unwrap_or(false);

                        let delay = Self::calculate_backoff_delay(attempt, is_rate_limited);
                        trace!("Retrying in {:?}", delay);
                        std::thread::sleep(delay);
                    } else {
                        break;
                    }
                }
            }
        }

        let error = last_error.unwrap();
        Err(VectorError::ConnectionError(format!(
            "Request failed after {} attempts: {}",
            max_retries + 1,
            error
        )))
    }

    pub fn list_collections(&self) -> Result<ListCollectionsResponse, VectorError> {
        trace!("Listing collections");

        let request = ListCollectionsRequest {
            db_name: self.database.clone(),
        };

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/collections/list")
                .json(&request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn create_collection(&self, request: &CreateCollectionRequest) -> Result<CreateCollectionResponse, VectorError> {
        trace!("Creating collection: {}", request.collection_name);

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/collections/create")
                .json(request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn describe_collection(&self, collection_name: &str) -> Result<DescribeCollectionResponse, VectorError> {
        trace!("Describing collection: {collection_name}");

        let request = DescribeCollectionRequest {
            db_name: self.database.clone(),
            collection_name: collection_name.to_string(),
        };

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/collections/describe")
                .json(&request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn drop_collection(&self, collection_name: &str) -> Result<DropCollectionResponse, VectorError> {
        trace!("Dropping collection: {collection_name}");

        let request = DropCollectionRequest {
            db_name: self.database.clone(),
            collection_name: collection_name.to_string(),
        };

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/collections/drop")
                .json(&request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn has_collection(&self, collection_name: &str) -> Result<HasCollectionResponse, VectorError> {
        trace!("Checking if collection exists: {collection_name}");

        let request = HasCollectionRequest {
            db_name: self.database.clone(),
            collection_name: collection_name.to_string(),
        };

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/collections/has")
                .json(&request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn load_collection(&self, collection_name: &str) -> Result<LoadCollectionResponse, VectorError> {
        trace!("Loading collection: {collection_name}");

        let request = LoadCollectionRequest {
            db_name: self.database.clone(),
            collection_name: collection_name.to_string(),
        };

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/collections/load")
                .json(&request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn release_collection(&self, collection_name: &str) -> Result<ReleaseCollectionResponse, VectorError> {
        trace!("Releasing collection: {collection_name}");

        let request = ReleaseCollectionRequest {
            db_name: self.database.clone(),
            collection_name: collection_name.to_string(),
        };

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/collections/release")
                .json(&request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn upsert(&self, request: &UpsertRequest) -> Result<UpsertResponse, VectorError> {
        trace!("Upserting {} vectors to collection: {}", request.data.len(), request.collection_name);

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/entities/upsert")
                .json(request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn search(&self, request: &SearchRequest) -> Result<SearchResponse, VectorError> {
        trace!("Searching vectors in collection: {}", request.collection_name);

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/entities/search")
                .json(request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn query(&self, request: &QueryRequest) -> Result<QueryResponse, VectorError> {
        trace!("Querying vectors in collection: {}", request.collection_name);

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/entities/query")
                .json(request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn get(&self, request: &GetRequest) -> Result<GetResponse, VectorError> {
        trace!("Getting vectors by IDs from collection: {}", request.collection_name);

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/entities/get")
                .json(request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn delete(&self, request: &DeleteRequest) -> Result<DeleteResponse, VectorError> {
        trace!("Deleting vectors from collection: {}", request.collection_name);

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/entities/delete")
                .json(request)
                .send()
        })?;

        parse_response(response)
    }

    pub fn get_collection_stats(&self, collection_name: &str) -> Result<GetCollectionStatsResponse, VectorError> {
        trace!("Getting stats for collection: {collection_name}");

        let request = GetCollectionStatsRequest {
            db_name: self.database.clone(),
            collection_name: collection_name.to_string(),
        };

        let response = self.execute_with_retry_sync(|| {
            self.create_request(Method::POST, "/v2/vectordb/collections/get_stats")
                .json(&request)
                .send()
        })?;

        parse_response(response)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListCollectionsRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListCollectionsResponse {
    pub code: i32,
    pub data: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    pub dimension: u32,
    #[serde(rename = "metricType", skip_serializing_if = "Option::is_none")]
    pub metric_type: Option<String>,
    #[serde(rename = "primaryField", skip_serializing_if = "Option::is_none")]
    pub primary_field: Option<String>,
    #[serde(rename = "vectorField", skip_serializing_if = "Option::is_none")]
    pub vector_field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "enableDynamicField", skip_serializing_if = "Option::is_none")]
    pub enable_dynamic_field: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<CollectionSchema>,
    #[serde(rename = "indexParams", skip_serializing_if = "Option::is_none")]
    pub index_params: Option<Vec<IndexParam>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionResponse {
    pub code: i32,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeCollectionRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeCollectionResponse {
    pub code: i32,
    pub data: CollectionInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    pub description: Option<String>,
    pub fields: Vec<FieldSchema>,
    pub indexes: Vec<IndexInfo>,
    pub load: String,
    #[serde(rename = "shardsNum")]
    pub shards_num: i32,
    #[serde(rename = "enableDynamicField")]
    pub enable_dynamic_field: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropCollectionRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropCollectionResponse {
    pub code: i32,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasCollectionRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasCollectionResponse {
    pub code: i32,
    pub data: HasCollectionData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasCollectionData {
    pub has: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadCollectionRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadCollectionResponse {
    pub code: i32,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseCollectionRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseCollectionResponse {
    pub code: i32,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetLoadStateRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetLoadStateResponse {
    pub code: i32,
    pub data: LoadStateData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadStateData {
    pub state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSchema {
    #[serde(rename = "autoId", skip_serializing_if = "Option::is_none")]
    pub auto_id: Option<bool>,
    #[serde(rename = "enableDynamicField", skip_serializing_if = "Option::is_none")]
    pub enable_dynamic_field: Option<bool>,
    pub fields: Vec<FieldSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    #[serde(rename = "fieldName")]
    pub field_name: String,
    #[serde(rename = "dataType")]
    pub data_type: String,
    #[serde(rename = "isPrimary", skip_serializing_if = "Option::is_none")]
    pub is_primary: Option<bool>,
    #[serde(rename = "elementDataType", skip_serializing_if = "Option::is_none")]
    pub element_data_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "elementTypeParams", skip_serializing_if = "Option::is_none")]
    pub element_type_params: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    pub data: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertResponse {
    pub code: i32,
    pub data: InsertData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertData {
    #[serde(rename = "insertCount")]
    pub insert_count: u32,
    #[serde(rename = "insertIds")]
    pub insert_ids: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    pub data: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResponse {
    pub code: i32,
    pub data: UpsertData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertData {
    #[serde(rename = "upsertCount")]
    pub upsert_count: u32,
    #[serde(rename = "upsertIds")]
    pub upsert_ids: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    pub data: Vec<Vec<f32>>,
    #[serde(rename = "annsField")]
    pub anns_field: String,
    #[serde(rename = "metricType")]
    pub metric_type: String,
    pub limit: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    #[serde(rename = "outputFields", skip_serializing_if = "Option::is_none")]
    pub output_fields: Option<Vec<String>>,
    #[serde(rename = "searchParams", skip_serializing_if = "Option::is_none")]
    pub search_params: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub code: i32,
    pub data: Vec<Vec<SearchResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: serde_json::Value,
    pub distance: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ids: Option<Vec<serde_json::Value>>,
    #[serde(rename = "outputFields", skip_serializing_if = "Option::is_none")]
    pub output_fields: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub code: i32,
    pub data: Vec<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    pub ids: Vec<serde_json::Value>,
    #[serde(rename = "outputFields", skip_serializing_if = "Option::is_none")]
    pub output_fields: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetResponse {
    pub code: i32,
    pub data: Vec<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ids: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    pub code: i32,
    pub data: DeleteData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteData {
    #[serde(rename = "deleteCount")]
    pub delete_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateIndexRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    #[serde(rename = "fieldName")]
    pub field_name: String,
    #[serde(rename = "metricType", skip_serializing_if = "Option::is_none")]
    pub metric_type: Option<String>,
    #[serde(rename = "indexParams", skip_serializing_if = "Option::is_none")]
    pub index_params: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateIndexResponse {
    pub code: i32,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeIndexRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    #[serde(rename = "fieldName")]
    pub field_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeIndexResponse {
    pub code: i32,
    pub data: Vec<IndexInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    #[serde(rename = "fieldName")]
    pub field_name: String,
    #[serde(rename = "indexName")]
    pub index_name: String,
    #[serde(rename = "metricType")]
    pub metric_type: String,
    #[serde(rename = "indexType")]
    pub index_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropIndexRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
    #[serde(rename = "fieldName")]
    pub field_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropIndexResponse {
    pub code: i32,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexParam {
    #[serde(rename = "fieldName")]
    pub field_name: String,
    #[serde(rename = "indexName")]
    pub index_name: String,
    #[serde(rename = "metricType", skip_serializing_if = "Option::is_none")]
    pub metric_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetCollectionStatsRequest {
    #[serde(rename = "dbName")]
    pub db_name: String,
    #[serde(rename = "collectionName")]
    pub collection_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetCollectionStatsResponse {
    pub code: i32,
    pub data: CollectionStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    #[serde(rename = "rowCount")]
    pub row_count: u64,
}

//parsing function

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, VectorError> {
    let status = response.status();

    trace!("Received response from Milvus API: {response:?}");

    if status.is_success() {
        match response.text() {
            Ok(body) => {
                trace!("Received response body from Milvus API: {body:?}");
                
                match serde_json::from_str::<T>(&body) {
                    Ok(parsed) => {
                        trace!("Successfully parsed response: {parsed:?}");
                        Ok(parsed)
                    }
                    Err(parse_error) => {
                        trace!("Failed to parse response: {parse_error}");
                        Err(VectorError::ProviderError(format!(
                            "Failed to parse Milvus response: {}",
                            parse_error
                        )))
                    }
                }
            }
            Err(body_error) => {
                trace!("Failed to read response body: {body_error}");
                Err(VectorError::ProviderError(format!(
                    "Failed to read Milvus response body: {}",
                    body_error
                )))
            }
        }
    } else {
        let error_body = response.text().unwrap_or_else(|_| "Unknown error".to_string());

        trace!("Received {status} response from Milvus API: {error_body:?}");

        let error_message = match status.as_u16() {
            400 => VectorError::InvalidParams(format!("Bad request: {}", error_body)),
            401 => VectorError::Unauthorized("Authentication failed".to_string()),
            404 => VectorError::NotFound(format!("Resource not found: {}", error_body)),
            409 => VectorError::AlreadyExists(format!("Resource already exists: {}", error_body)),
            429 => VectorError::RateLimited("Rate limit exceeded".to_string()),
            500..=599 => VectorError::ProviderError(format!("Server error: {}", error_body)),
            _ => VectorError::ProviderError(format!("HTTP {}: {}", status.as_u16(), error_body)),
        };

        Err(error_message)
    }
}
