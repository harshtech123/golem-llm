use golem_vector::config::{get_max_retries_config};
use golem_vector::golem::vector::types::VectorError;
use log::trace;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use golem_rust::bindings::golem::rdbms::postgres::{DbConnection, DbValue, DbResult};
use crate::conversions::{distance_metric_to_pgvector_operator, string_to_distance_metric, string_value_to_db_value};


/// PostgreSQL Vector (pgvector) client using Golem RDBMS interface
#[derive(Clone)]
pub struct PgVectorClient {
    connection_string: String,
}

impl PgVectorClient {
    pub fn new(connection_string: String) -> Self {
        Self { connection_string }
    }

    pub fn connection_string(&self) -> &str {
        &self.connection_string
    }

    fn get_connection(&self) -> Result<DbConnection, VectorError> {
        match DbConnection::open(&self.connection_string) {
            Ok(conn) => Ok(conn),
            Err(e) => Err(VectorError::ConnectionError(format!("Failed to connect to PostgreSQL: {:?}", e)))
        }
    }

    fn execute_sql(&self, sql: &str, params: Vec<DbValue>) -> Result<u64, VectorError> {
        let conn = self.get_connection()?;
        trace!("Executing SQL: {} with {} params", sql, params.len());
        
        match conn.execute(sql, params) {
            Ok(rows_affected) => Ok(rows_affected as u64),
            Err(e) => Err(VectorError::ProviderError(format!("SQL execution failed: {:?}", e)))
        }
    }

    fn query_sql(&self, sql: &str, params: Vec<DbValue>) -> Result<DbResult, VectorError> {
        let conn = self.get_connection()?;
        trace!("Querying SQL: {} with {} params", sql, params.len());
        
        match conn.query(sql, params) {
            Ok(result) => Ok(result),
            Err(e) => Err(VectorError::ProviderError(format!("SQL query failed: {:?}", e)))
        }
    }

    fn db_value_to_string(&self, value: &DbValue) -> Option<String> {
        match value {
            DbValue::Text(val) => Some(val.clone()),
            DbValue::Varchar(val) => Some(val.clone()),
            DbValue::Bpchar(val) => Some(val.clone()),
            DbValue::Int4(val) => Some(val.to_string()),
            DbValue::Int8(val) => Some(val.to_string()),
            DbValue::Float4(val) => Some(val.to_string()),
            DbValue::Float8(val) => Some(val.to_string()),
            DbValue::Boolean(val) => Some(val.to_string()),
            DbValue::Date(val) => Some(format!("{:?}", val)),
            DbValue::Time(val) => Some(format!("{:?}", val)),
            DbValue::Timestamp(val) => Some(format!("{:?}", val)),
            DbValue::Timestamptz(val) => Some(format!("{:?}", val)),
            DbValue::Uuid(val) => Some(format!("{:?}", val)),
            DbValue::Json(val) => Some(val.clone()),
            DbValue::Jsonb(val) => Some(val.clone()),
            DbValue::Null => None,
            _ => Some(format!("{:?}", value)),
        }
    }

    fn db_value_to_f32(&self, value: &DbValue) -> Option<f32> {
        match value {
            DbValue::Float4(val) => Some(*val),
            DbValue::Float8(val) => Some(*val as f32),
            DbValue::Int4(val) => Some(*val as f32),
            DbValue::Text(val) | DbValue::Varchar(val) => val.parse().ok(),
            _ => None,
        }
    }

    fn parse_pgvector_string(&self, s: &str) -> Result<Vec<f32>, VectorError> {
        let trimmed = s.trim_start_matches('[').trim_end_matches(']');
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }
        
        trimmed.split(',')
            .map(|part| part.trim().parse::<f32>()
                .map_err(|e| VectorError::ProviderError(format!("Failed to parse vector component '{}': {}", part, e))))
            .collect()
    }

    fn get_table_column_types(&self, table_name: &str) -> Result<HashMap<String, String>, VectorError> {
        let sql = r#"
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = $1 AND table_schema = 'public'
        "#;
        
        let params = vec![DbValue::Text(table_name.to_string())];
        let result = self.query_sql(sql, params)?;
        
        let mut column_types = HashMap::new();
        for row in &result.rows {
            if row.values.len() >= 2 {
                let column_name = match &row.values[0] {
                    DbValue::Text(name) => name.clone(),
                    _ => continue,
                };
                let data_type = match &row.values[1] {
                    DbValue::Text(dtype) => dtype.clone(),
                    _ => continue,
                };
                column_types.insert(column_name, data_type);
            }
        }
        
        Ok(column_types)
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
                    trace!("SQL operation succeeded on attempt {}", attempt + 1);
                    return Ok(result);
                }
                Err(e) => {
                    trace!("SQL operation failed on attempt {}: {}", attempt + 1, e);
                    last_error = Some(e);
                    
                    if attempt < max_retries {
                        std::thread::sleep(std::time::Duration::from_millis(100 * (2_u64.pow(attempt))));
                        continue;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| VectorError::ProviderError("Max retries exceeded".to_string())))
    }

    pub fn enable_extension(&self) -> Result<(), VectorError> {
        self.execute_with_retry(|| {
            self.execute_sql("CREATE EXTENSION IF NOT EXISTS vector", vec![])?;
            Ok(())
        })
    }

    pub fn create_table(&self, request: &CreateTableRequest) -> Result<CreateTableResponse, VectorError> {
        self.execute_with_retry(|| {
            let mut sql = format!("CREATE TABLE IF NOT EXISTS \"{}\" (", request.table_name);
            sql.push_str("id VARCHAR PRIMARY KEY, ");
            
            if let Some(dimension) = request.dimension {
                sql.push_str(&format!("embedding vector({}), ", dimension));
            } else {
                sql.push_str("embedding vector, ");
            }
            
            for (column_name, column_type) in &request.metadata_columns {
                sql.push_str(&format!("\"{}\" {}, ", column_name, column_type));
            }
            
            sql = sql.trim_end_matches(", ").to_string();
            sql.push_str(")");
            
            self.execute_sql(&sql, vec![])?;
            
            Ok(CreateTableResponse {
                table_name: request.table_name.clone(),
            })
        })
    }

    pub fn drop_table(&self, table_name: &str) -> Result<DropTableResponse, VectorError> {
        self.execute_with_retry(|| {
            let sql = format!("DROP TABLE IF EXISTS \"{}\"", table_name);
            self.execute_sql(&sql, vec![])?;
            
            Ok(DropTableResponse {
                table_name: table_name.to_string(),
            })
        })
    }

    pub fn table_exists(&self, table_name: &str) -> Result<TableExistsResponse, VectorError> {
        self.execute_with_retry(|| {
            let sql = "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)";
            let params = vec![DbValue::Text(table_name.to_string())];
            let result = self.query_sql(sql, params)?;
            
            let exists = if let Some(row) = result.rows.first() {
                if let Some(value) = row.values.first() {
                    match value {
                        DbValue::Boolean(val) => *val,
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            };
            
            Ok(TableExistsResponse { exists })
        })
    }

    pub fn upsert_vectors(&self, request: &UpsertVectorsRequest) -> Result<UpsertVectorsResponse, VectorError> {
        self.execute_with_retry(|| {
            let column_types = self.get_table_column_types(&request.table_name)?;
            let mut upserted_count = 0;
            
            for vector in &request.vectors {
                let embedding_str = format!("[{}]", vector.embedding.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(","));
                
                let mut columns = vec!["id".to_string(), "embedding".to_string()];
                let mut placeholders = vec!["$1".to_string(), "$2".to_string()];
                let mut params = vec![
                    DbValue::Text(vector.id.clone()),
                    DbValue::Text(embedding_str),
                ];
                let mut update_clauses = vec!["embedding = EXCLUDED.embedding".to_string()];
                
                let mut param_index = 3;
                for (key, value) in &vector.metadata {
                    columns.push(format!("\"{}\"", key));
                    placeholders.push(format!("${}", param_index));
                    
                    let db_value = if let Some(column_type) = column_types.get(key) {
                        string_value_to_db_value(value, column_type)?
                    } else {
                        DbValue::Text(value.clone())
                    };
                    
                    params.push(db_value);
                    update_clauses.push(format!("\"{}\" = EXCLUDED.\"{}\"", key, key));
                    param_index += 1;
                }
                
                let sql = format!(
                    "INSERT INTO \"{}\" ({}) VALUES ({}) ON CONFLICT (id) DO UPDATE SET {}",
                    request.table_name,
                    columns.join(", "),
                    placeholders.join(", "),
                    update_clauses.join(", ")
                );
                
                self.execute_sql(&sql, params)?;
                upserted_count += 1;
            }
            
            Ok(UpsertVectorsResponse { upserted_count: upserted_count as u32 })
        })
    }

    pub fn search_vectors(&self, request: &SearchRequest) -> Result<SearchResponse, VectorError> {
        self.execute_with_retry(|| {
            let operator = distance_metric_to_pgvector_operator(&string_to_distance_metric(&request.distance_metric));
            
            let mut sql = format!(
                "SELECT {}, embedding {} $1 as distance FROM \"{}\"",
                request.select_columns.join(", "),
                operator,
                request.table_name
            );
            
            let query_vector_str = format!("[{}]", request.query_vector.iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(","));
            
            let mut params = vec![DbValue::Text(query_vector_str)];
            
            if !request.filters.is_empty() {
                if let Some(where_clause) = request.filters.get("where_clause") {
                    sql.push_str(&format!(" WHERE {}", where_clause));
                } else {
                    let mut filter_conditions = Vec::new();
                    for (key, value) in &request.filters {
                        params.push(DbValue::Text(value.clone()));
                        filter_conditions.push(format!("\"{}\" = ${}", key, params.len()));
                    }
                    
                    if !filter_conditions.is_empty() {
                        sql.push_str(&format!(" WHERE {}", filter_conditions.join(" AND ")));
                    }
                }
            }
            
            sql.push_str(&format!(" ORDER BY embedding {} $1 LIMIT {}", operator, request.limit));
            
            let result = self.query_sql(&sql, params)?;
            
            let mut results = Vec::new();
            for row in &result.rows {
                let mut id = String::new();
                let mut distance = 0.0f32;
                let mut embedding = Vec::new();
                let mut metadata = HashMap::new();
                
                for (i, column) in result.columns.iter().enumerate() {
                    if let Some(value) = row.values.get(i) {
                        match column.name.as_str() {
                            "id" => {
                                if let Some(s) = self.db_value_to_string(value) {
                                    id = s;
                                }
                            },
                            "distance" => {
                                if let Some(f) = self.db_value_to_f32(value) {
                                    distance = f;
                                }
                            },
                            "embedding" => {
                                if let Some(s) = self.db_value_to_string(value) {
                                    if let Ok(parsed) = self.parse_pgvector_string(&s) {
                                        embedding = parsed;
                                    }
                                }
                            },
                            _ => {
                                if let Some(s) = self.db_value_to_string(value) {
                                    metadata.insert(column.name.clone(), s);
                                }
                            }
                        }
                    }
                }
                
                results.push(SearchResult {
                    id,
                    embedding,
                    distance,
                    metadata,
                });
            }
            
            Ok(SearchResponse { results })
        })
    }

    pub fn get_vectors(&self, request: &GetVectorsRequest) -> Result<GetVectorsResponse, VectorError> {
        self.execute_with_retry(|| {
            let columns = if request.select_columns.is_empty() {
                "id, embedding".to_string()
            } else {
                request.select_columns.join(", ")
            };
            
            let placeholders: Vec<String> = (1..=request.ids.len())
                .map(|i| format!("${}", i))
                .collect();
            
            let sql = format!(
                "SELECT {} FROM \"{}\" WHERE id IN ({})",
                columns,
                request.table_name,
                placeholders.join(", ")
            );
            
            let params: Vec<DbValue> = request.ids.iter().map(|id| DbValue::Text(id.clone())).collect();
            let result = self.query_sql(&sql, params)?;
            
            let mut results = Vec::new();
            for row in &result.rows {
                let mut id = String::new();
                let mut embedding = Vec::new();
                let mut metadata = HashMap::new();
                
                for (i, column) in result.columns.iter().enumerate() {
                    if let Some(value) = row.values.get(i) {
                        match column.name.as_str() {
                            "id" => {
                                if let Some(s) = self.db_value_to_string(value) {
                                    id = s;
                                }
                            },
                            "embedding" => {
                                if let Some(s) = self.db_value_to_string(value) {
                                    if let Ok(parsed) = self.parse_pgvector_string(&s) {
                                        embedding = parsed;
                                    }
                                }
                            },
                            _ => {
                                if let Some(s) = self.db_value_to_string(value) {
                                    metadata.insert(column.name.clone(), s);
                                }
                            }
                        }
                    }
                }
                
                results.push(VectorResult {
                    id,
                    embedding,
                    metadata,
                });
            }
            
            Ok(GetVectorsResponse { results })
        })
    }

    pub fn delete_vectors(&self, request: &DeleteVectorsRequest) -> Result<DeleteVectorsResponse, VectorError> {
        self.execute_with_retry(|| {
            let placeholders: Vec<String> = (1..=request.ids.len())
                .map(|i| format!("${}", i))
                .collect();
                
            let sql = format!(
                "DELETE FROM \"{}\" WHERE id IN ({})",
                request.table_name,
                placeholders.join(", ")
            );
            
            let params: Vec<DbValue> = request.ids.iter().map(|id| DbValue::Text(id.clone())).collect();
            let rows_affected = self.execute_sql(&sql, params)?;
            
            Ok(DeleteVectorsResponse {
                deleted_count: rows_affected as u32,
            })
        })
    }

    pub fn count_vectors(&self, table_name: &str) -> Result<CountVectorsResponse, VectorError> {
        self.execute_with_retry(|| {
            let sql = format!("SELECT COUNT(*) as count FROM \"{}\"", table_name);
            let result = self.query_sql(&sql, vec![])?;
            
            let count = if let Some(row) = result.rows.first() {
                if let Some(value) = row.values.first() {
                    match value {
                        DbValue::Int8(val) => *val as u64,
                        DbValue::Int4(val) => *val as u64,
                        _ => 0,
                    }
                } else {
                    0
                }
            } else {
                0
            };
            
            Ok(CountVectorsResponse { count })
        })
    }

    pub fn list_tables(&self) -> Result<ListTablesResponse, VectorError> {
        self.execute_with_retry(|| {
            let sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'";
            let result = self.query_sql(sql, vec![])?;
            
            let tables: Vec<String> = result.rows.iter()
                .filter_map(|row| {
                    if let Some(value) = row.values.first() {
                        match value {
                            DbValue::Text(table_name) => Some(table_name.clone()),
                            _ => None,
                        }
                    } else {
                        None
                    }
                })
                .collect();
            
            Ok(ListTablesResponse { tables })
        })
    }

    pub fn describe_table(&self, table_name: &str) -> Result<DescribeTableResponse, VectorError> {
        self.execute_with_retry(|| {
            let sql = r#"
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = $1 AND table_schema = 'public'
                ORDER BY ordinal_position
            "#;
            
            let params = vec![DbValue::Text(table_name.to_string())];
            let result = self.query_sql(sql, params)?;
            
            let columns: Vec<TableColumn> = result.rows.iter()
                .map(|row| {
                    let name = if row.values.len() > 0 {
                        match &row.values[0] {
                            DbValue::Text(name) => name.clone(),
                            _ => String::new(),
                        }
                    } else {
                        String::new()
                    };
                    
                    let data_type = if row.values.len() > 1 {
                        match &row.values[1] {
                            DbValue::Text(dtype) => dtype.clone(),
                            _ => String::new(),
                        }
                    } else {
                        String::new()
                    };
                    
                    let nullable = if row.values.len() > 2 {
                        match &row.values[2] {
                            DbValue::Text(is_null) => is_null == "YES",
                            _ => false,
                        }
                    } else {
                        false
                    };
                    
                    TableColumn {
                        name,
                        data_type,
                        nullable,
                    }
                })
                .collect();
            
            Ok(DescribeTableResponse {
                table_name: table_name.to_string(),
                columns,
            })
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
pub struct VectorData {
    pub id: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
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
        Self::new("postgres://postgres@localhost:5432/postgres".to_string())
    }
}
