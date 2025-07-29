use golem_graph::golem::graph::errors::GraphError;
use golem_graph::golem::graph::types::{Edge, Vertex};
use log::trace;
use reqwest::{Client, Response};
use serde_json::{json, Value};
use uuid::Uuid;

#[derive(Clone)]
pub struct JanusGraphApi {
    endpoint: String,
    client: Client,
    session_id: String,
}

// Response type definitions for type-safe parsing
pub struct VertexDocument {
    pub id: Value,
    pub label: String,
    pub properties: std::collections::HashMap<String, Value>,
}

pub struct EdgeDocument {
    pub id: Value,
    pub label: String,
    pub in_v: Value,
    pub out_v: Value,
    pub properties: std::collections::HashMap<String, Value>,
}

pub struct GremlinOperationResponse {
    pub data: Vec<Value>,
    pub meta: Option<GremlinOperationExtra>,
}

pub struct GremlinOperationExtra {
    pub request_id: Option<String>,
    pub status: Option<GremlinOperationStatus>,
}

pub struct GremlinOperationStatus {
    pub message: String,
    pub code: u16,
}

impl JanusGraphApi {
    pub fn new(
        host: &str,
        port: u16,
        _username: Option<&str>,
        _password: Option<&str>,
    ) -> Result<Self, GraphError> {
        trace!("Initializing JanusGraphApi for host: {host}, port: {port}");
        let endpoint = format!("http://{host}:{port}/gremlin");
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        let session_id = Uuid::new_v4().to_string();
        Ok(JanusGraphApi {
            endpoint,
            client,
            session_id,
        })
    }

    pub fn new_with_session(
        host: &str,
        port: u16,
        _username: Option<&str>,
        _password: Option<&str>,
        session_id: String,
    ) -> Result<Self, GraphError> {
        trace!(
            "Initializing JanusGraphApi with session for host: {host}, port: {port}, session_id: {session_id}"
        );
        let endpoint = format!("http://{host}:{port}/gremlin");
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Ok(JanusGraphApi {
            endpoint,
            client,
            session_id,
        })
    }

    pub fn commit(&self) -> Result<(), GraphError> {
        trace!("Commit transaction");
        self.execute("g.tx().commit()", None)?;
        self.execute("g.tx().open()", None)?;
        Ok(())
    }

    pub fn execute(&self, gremlin: &str, bindings: Option<Value>) -> Result<Value, GraphError> {
        trace!("Execute Gremlin query: {gremlin}");
        let bindings = bindings.unwrap_or_else(|| json!({}));
        let request_body = json!({
            "gremlin": gremlin,
            "bindings": bindings,
            "session": self.session_id,
            "processor": "session",
            "op": "eval",

        });

        trace!("[JanusGraphApi] DEBUG - Full request details:");
        trace!("[JanusGraphApi] Endpoint: {}", self.endpoint);
        trace!("[JanusGraphApi] Session ID: {}", self.session_id);
        trace!("[JanusGraphApi] Gremlin Query: {gremlin}");
        trace!(
            "[JanusGraphApi] Request Body: {}",
            serde_json::to_string_pretty(&request_body)
                .unwrap_or_else(|_| "Failed to serialize".to_string())
        );

        let body_string = serde_json::to_string(&request_body).map_err(|e| {
            GraphError::InternalError(format!("Failed to serialize request body: {e}"))
        })?;

        trace!(
            "[JanusGraphApi] Sending POST request to: {} with body length: {}",
            self.endpoint,
            body_string.len()
        );
        let response = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Content-Length", body_string.len().to_string())
            .body(body_string)
            .send()
            .map_err(|e| {
                log::error!("[JanusGraphApi] ERROR - Request failed: {e}");
                self.handle_janusgraph_reqwest_error("JanusGraph request failed", e)
            })?;

        log::info!(
            "[JanusGraphApi] Got response with status: {}",
            response.status()
        );
        Self::handle_response(response)
    }

    fn _read(&self, gremlin: &str, bindings: Option<Value>) -> Result<Value, GraphError> {
        trace!("Read Gremlin query: {gremlin}");
        let bindings = bindings.unwrap_or_else(|| json!({}));
        let request_body = json!({
            "gremlin": gremlin,
            "bindings": bindings,
        });

        let body_string = serde_json::to_string(&request_body).map_err(|e| {
            GraphError::InternalError(format!("Failed to serialize request body: {e}"))
        })?;

        let response = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Content-Length", body_string.len().to_string())
            .body(body_string)
            .send()
            .map_err(|e| {
                self.handle_janusgraph_reqwest_error("JanusGraph read request failed", e)
            })?;
        Self::handle_response(response)
    }

    pub fn close_session(&self) -> Result<(), GraphError> {
        trace!("Close session: {}", self.session_id);
        let request_body = json!({
            "session": self.session_id,
            "op": "close",
            "processor": "session"
        });

        let body_string = serde_json::to_string(&request_body).map_err(|e| {
            GraphError::InternalError(format!("Failed to serialize request body: {e}"))
        })?;

        let response = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Content-Length", body_string.len().to_string())
            .body(body_string)
            .send()
            .map_err(|e| {
                self.handle_janusgraph_reqwest_error("JanusGraph close session failed", e)
            })?;
        Self::handle_response(response).map(|_| ())
    }

    pub fn session_id(&self) -> &str {
        trace!("Get session ID: {}", self.session_id);
        &self.session_id
    }

    fn handle_janusgraph_reqwest_error(&self, details: &str, err: reqwest::Error) -> GraphError {
        if err.is_timeout() {
            return GraphError::Timeout;
        }

        if err.is_request() {
            return GraphError::ConnectionFailed(format!(
                "JanusGraph request failed ({details}): {err}"
            ));
        }

        if err.is_decode() {
            return GraphError::InternalError(format!(
                "JanusGraph response decode failed ({details}): {err}"
            ));
        }

        if err.is_status() {
            if let Some(status) = err.status() {
                let error_msg = format!(
                    "JanusGraph HTTP error {} ({}): {}",
                    status.as_u16(),
                    details,
                    err
                );
                return Self::map_janusgraph_http_status(
                    status.as_u16(),
                    &error_msg,
                    &serde_json::Value::Null,
                );
            }
        }

        GraphError::InternalError(format!("JanusGraph request error ({details}): {err}"))
    }

    fn handle_response(response: Response) -> Result<Value, GraphError> {
        let status = response.status();
        let status_code = status.as_u16();

        if status.is_success() {
            let response_body: Value = response.json().map_err(|e| {
                GraphError::InternalError(format!("Failed to parse response body: {e}"))
            })?;
            Ok(response_body)
        } else {
            let error_body: Value = response.json().map_err(|e| {
                GraphError::InternalError(format!("Failed to read error response: {e}"))
            })?;

            let error_msg = error_body
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");

            Err(Self::map_janusgraph_error(
                status_code,
                error_msg,
                &error_body,
            ))
        }
    }

    fn map_janusgraph_error(
        status_code: u16,
        message: &str,
        error_body: &serde_json::Value,
    ) -> GraphError {
        if let Some(status) = error_body.get("status") {
            if let Some(status_obj) = status.as_object() {
                if let Some(code) = status_obj.get("code").and_then(|c| c.as_u64()) {
                    return Self::from_gremlin_status_code(code as u16, message, error_body);
                }
            }
        }

        if let Some(result) = error_body.get("result") {
            if let Some(result_obj) = result.as_object() {
                if let Some(data) = result_obj.get("data") {
                    if let Some(detailed_message_val) = data.get("detailedMessage") {
                        if let Some(detailed_msg) = detailed_message_val.as_str() {
                            return Self::from_janusgraph_detailed_error(detailed_msg, error_body);
                        }
                    }
                }
            }
        }

        if let Some(exceptions) = error_body.get("exceptions") {
            if let Some(exceptions_array) = exceptions.as_array() {
                if let Some(first_exception) = exceptions_array.first() {
                    if let Some(exception_msg) = first_exception.as_str() {
                        return Self::from_janusgraph_exception(exception_msg, error_body);
                    }
                }
            }
        }

        Self::map_janusgraph_http_status(status_code, message, error_body)
    }

    fn from_gremlin_status_code(
        code: u16,
        message: &str,
        error_body: &serde_json::Value,
    ) -> GraphError {
        let detailed_error = Self::extract_gremlin_exception_info(error_body);

        match code {
            // Authentication and Authorization
            401 => GraphError::AuthenticationFailed(format!(
                "Gremlin authentication failed: {message}"
            )),
            403 => {
                GraphError::AuthorizationFailed(format!("Gremlin authorization failed: {message}"))
            }
            407 => GraphError::AuthenticationFailed(format!(
                "Gremlin authentication challenge: {message}"
            )),

            // Client Request Errors
            498 => {
                if let Some(detailed) = detailed_error {
                    return detailed;
                }
                GraphError::InvalidQuery(format!("Gremlin malformed request: {message}"))
            }
            499 => {
                GraphError::InvalidQuery(format!("Gremlin invalid request arguments: {message}"))
            }

            // Server Errors
            500 => {
                if let Some(detailed) = detailed_error {
                    return detailed;
                }
                GraphError::InternalError(format!("Gremlin server error: {message}"))
            }
            597 => {
                if let Some(detailed) = detailed_error {
                    return detailed;
                }
                GraphError::InvalidQuery(format!("Gremlin script evaluation error: {message}"))
            }
            598 => GraphError::Timeout,
            599 => GraphError::InternalError(format!("Gremlin serialization error: {message}")),

            // Default fallback
            _ => {
                if let Some(detailed) = detailed_error {
                    return detailed;
                }
                let debug_info = format!(
                    "Gremlin status code [{code}]: {message} | Error body sample: {}",
                    error_body.to_string().chars().take(200).collect::<String>()
                );
                GraphError::InternalError(debug_info)
            }
        }
    }

    fn extract_gremlin_exception_info(error_body: &serde_json::Value) -> Option<GraphError> {
        if let Some(result) = error_body.get("result") {
            if let Some(data) = result.get("data") {
                if let Some(at_type) = data.get("@type") {
                    if at_type.as_str() == Some("g:Map") {
                        if let Some(exception_class) = data.get("java.lang.Class") {
                            if let Some(class_name) = exception_class.as_str() {
                                return Self::map_java_exception_class(class_name, data);
                            }
                        }

                        if let Some(stack_trace) = data.get("stackTrace") {
                            if let Some(stack_str) = stack_trace.as_str() {
                                return Self::extract_from_stack_trace(stack_str);
                            }
                        }
                    }
                }

                if let Some(exception_type) = data.get("exceptionType") {
                    if let Some(ex_type) = exception_type.as_str() {
                        return Self::map_java_exception_class(ex_type, data);
                    }
                }
            }
        }

        if let Some(exceptions) = error_body.get("exceptions") {
            if let Some(exceptions_array) = exceptions.as_array() {
                if let Some(first_exception) = exceptions_array.first() {
                    if let Some(exception_obj) = first_exception.as_object() {
                        if let Some(exception_type) = exception_obj.get("@type") {
                            if let Some(ex_type) = exception_type.as_str() {
                                return Self::map_java_exception_class(ex_type, first_exception);
                            }
                        }
                    }
                }
            }
        }

        None
    }

    fn map_java_exception_class(
        class_name: &str,
        exception_data: &serde_json::Value,
    ) -> Option<GraphError> {
        let message = exception_data
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or(class_name);

        match class_name {
            // JanusGraph specific exceptions
            "org.janusgraph.core.SchemaViolationException" => {
                Some(GraphError::SchemaViolation(format!("JanusGraph schema violation: {message}")))
            }
            "org.janusgraph.core.JanusGraphException" => {
                Some(GraphError::InternalError(format!("JanusGraph error: {message}")))
            }
            "org.janusgraph.diskstorage.TemporaryBackendException" => {
                Some(GraphError::ServiceUnavailable(format!("JanusGraph backend temporarily unavailable: {message}")))
            }
            "org.janusgraph.diskstorage.PermanentBackendException" => {
                Some(GraphError::InternalError(format!("JanusGraph backend permanent error: {message}")))
            }
            // Gremlin/TinkerPop exceptions
            "org.apache.tinkerpop.gremlin.process.traversal.strategy.verification.VerificationException" => {
                Some(GraphError::InvalidQuery(format!("Gremlin verification error: {message}")))
            }
            "org.apache.tinkerpop.gremlin.groovy.CompilationFailedException" => {
                Some(GraphError::InvalidQuery(format!("Gremlin compilation error: {message}")))
            }
            "org.apache.tinkerpop.gremlin.driver.exception.ResponseException" => {
                if let Some(response_code) = exception_data.get("responseStatusCode") {
                    if let Some(code) = response_code.as_u64() {
                        return Some(Self::from_gremlin_status_code(code as u16, message, exception_data));
                    }
                }
                Some(GraphError::InternalError(format!("Gremlin response error: {message}")))
            }
            // Standard Java exceptions with graph context
            "java.util.concurrent.TimeoutException" => {
                Some(GraphError::Timeout)
            }
            "java.lang.IllegalArgumentException" => {
                Some(GraphError::InvalidQuery(format!("Invalid argument: {message}")))
            }
            "java.lang.UnsupportedOperationException" => {
                Some(GraphError::UnsupportedOperation(format!("Unsupported operation: {message}")))
            }
            "java.lang.IllegalStateException" => {
                Some(GraphError::TransactionFailed(format!("Illegal state: {message}")))
            }
            "java.util.NoSuchElementException" => {
                if let Some(element_id) = golem_graph::error::mapping::extract_element_id_from_message(message) {
                    Some(GraphError::ElementNotFound(element_id))
                } else {
                    Some(GraphError::InternalError(format!("Element not found: {message}")))
                }
            }
            // Transaction related exceptions
            "org.apache.tinkerpop.gremlin.structure.util.TransactionException" => {
                Some(GraphError::TransactionFailed(format!("Transaction error: {message}")))
            }
            _ => None

        }
    }

    fn extract_from_stack_trace(stack_trace: &str) -> Option<GraphError> {
        let first_line = stack_trace.lines().next()?;

        if let Some(colon_pos) = first_line.find(':') {
            let exception_part = &first_line[..colon_pos];
            let message_part = &first_line[colon_pos + 1..].trim();

            if let Some(last_dot) = exception_part.rfind('.') {
                let _class_name = &exception_part[last_dot + 1..];
                let full_class_name = exception_part;

                let exception_data = serde_json::json!({
                    "message": message_part
                });

                return Self::map_java_exception_class(full_class_name, &exception_data);
            }
        }

        None
    }

    fn from_janusgraph_detailed_error(
        detailed_message: &str,
        error_body: &serde_json::Value,
    ) -> GraphError {
        if let Some(detailed_error) = Self::extract_gremlin_exception_info(error_body) {
            return detailed_error;
        }

        if let Some(exception_error) = Self::extract_from_stack_trace(detailed_message) {
            return exception_error;
        }

        GraphError::InternalError(format!("JanusGraph detailed error: {detailed_message}"))
    }

    fn from_janusgraph_exception(
        exception_message: &str,
        error_body: &serde_json::Value,
    ) -> GraphError {
        if let Some(detailed_error) = Self::extract_gremlin_exception_info(error_body) {
            return detailed_error;
        }

        if let Some(exception_error) = Self::extract_from_stack_trace(exception_message) {
            return exception_error;
        }

        GraphError::InternalError(format!("JanusGraph exception: {exception_message}"))
    }

    fn map_janusgraph_http_status(
        status: u16,
        message: &str,
        error_body: &serde_json::Value,
    ) -> GraphError {
        match status {
            // Authentication and Authorization
            401 => GraphError::AuthenticationFailed(format!(
                "JanusGraph authentication failed: {message}"
            )),
            403 => GraphError::AuthorizationFailed(format!(
                "JanusGraph authorization failed: {message}"
            )),

            // Client errors
            400 => {
                if let Some(detailed_error) = Self::extract_gremlin_exception_info(error_body) {
                    return detailed_error;
                }
                GraphError::InvalidQuery(format!("JanusGraph bad request: {message}"))
            }
            404 => {
                GraphError::ServiceUnavailable(format!("JanusGraph resource not found: {message}"))
            }
            409 => GraphError::TransactionConflict,
            413 => {
                GraphError::ResourceExhausted(format!("JanusGraph request too large: {message}"))
            }
            429 => {
                GraphError::ResourceExhausted(format!("JanusGraph rate limit exceeded: {message}"))
            }

            // Server errors
            500 => {
                if let Some(detailed_error) = Self::extract_gremlin_exception_info(error_body) {
                    return detailed_error;
                }
                GraphError::InternalError(format!("JanusGraph internal server error: {message}"))
            }
            502 => GraphError::ServiceUnavailable(format!("JanusGraph bad gateway: {message}")),
            503 => {
                GraphError::ServiceUnavailable(format!("JanusGraph service unavailable: {message}"))
            }
            504 => GraphError::Timeout,
            507 => {
                GraphError::ResourceExhausted(format!("JanusGraph insufficient storage: {message}"))
            }

            _ => {
                if let Some(detailed_error) = Self::extract_gremlin_exception_info(error_body) {
                    return detailed_error;
                }

                let debug_info = format!(
                    "JanusGraph HTTP error [{}]: {} | Error body sample: {}",
                    status,
                    message,
                    error_body.to_string().chars().take(200).collect::<String>()
                );
                GraphError::InternalError(debug_info)
            }
        }
    }

    // Centralized response parsing functions
    pub fn parse_gremlin_response(&self, response: Value, operation: &str) -> Result<GremlinOperationResponse, GraphError> {
        let result = response
            .get("result")
            .ok_or_else(|| GraphError::InternalError(
                format!("Missing 'result' in {} response", operation)
            ))?;

        let data = result
            .get("data")
            .ok_or_else(|| GraphError::InternalError(
                format!("Missing 'data' in {} response", operation)
            ))?;

        let data_vec = if let Some(graphson_obj) = data.as_object() {
            if data.get("@type") == Some(&json!("g:List")) {
                data.get("@value")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.clone())
                    .unwrap_or_default()
            } else {
                vec![data.clone()]
            }
        } else if let Some(arr) = data.as_array() {
            arr.clone()
        } else {
            vec![data.clone()]
        };

        let meta = result.get("meta").map(|_| GremlinOperationExtra {
            request_id: result.get("requestId").and_then(|v| v.as_str()).map(String::from),
            status: result.get("status").and_then(|s| {
                s.as_object().map(|obj| GremlinOperationStatus {
                    message: obj.get("message").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    code: obj.get("code").and_then(|v| v.as_u64()).unwrap_or(0) as u16,
                })
            }),
        });

        Ok(GremlinOperationResponse {
            data: data_vec,
            meta,
        })
    }

    pub fn parse_gremlin_vertex_response(&self, response: Value, operation: &str) -> Result<Vec<Vertex>, GraphError> {
        let gremlin_response = self.parse_gremlin_response(response, operation)?;
        
        gremlin_response.data
            .iter()
            .map(|item| crate::helpers::parse_vertex_from_gremlin(item)
                .map_err(|e| GraphError::InternalError(
                    format!("Failed to parse vertex in {} operation: {}", operation, e)
                )))
            .collect()
    }

    pub fn parse_gremlin_edge_response(&self, response: Value, operation: &str) -> Result<Vec<Edge>, GraphError> {
        let gremlin_response = self.parse_gremlin_response(response, operation)?;
        
        gremlin_response.data
            .iter()
            .map(|item| crate::helpers::parse_edge_from_gremlin(item)
                .map_err(|e| GraphError::InternalError(
                    format!("Failed to parse edge in {} operation: {}", operation, e)
                )))
            .collect()
    }

    pub fn parse_single_vertex_response(&self, response: Value, operation: &str) -> Result<Option<Vertex>, GraphError> {
        let vertices = self.parse_gremlin_vertex_response(response, operation)?;
        Ok(vertices.into_iter().next())
    }

    pub fn parse_single_edge_response(&self, response: Value, operation: &str) -> Result<Option<Edge>, GraphError> {
        let edges = self.parse_gremlin_edge_response(response, operation)?;
        Ok(edges.into_iter().next())
    }

    // Static helper functions for centralized parsing
    pub fn parse_gremlin_vertex_list(data: &Value, operation: &str) -> Result<Vec<Vertex>, GraphError> {
        let data_vec = if let Some(graphson_obj) = data.as_object() {
            if data.get("@type") == Some(&json!("g:List")) {
                data.get("@value")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.clone())
                    .unwrap_or_default()
            } else {
                vec![data.clone()]
            }
        } else if let Some(arr) = data.as_array() {
            arr.clone()
        } else {
            vec![data.clone()]
        };

        let mut vertices = Vec::new();
        for item in data_vec {
            if let Some(vertex_obj) = item.as_object() {
                if vertex_obj.get("@type") == Some(&json!("g:Vertex")) {
                    if let Some(value_obj) = vertex_obj.get("@value").and_then(|v| v.as_object()) {
                        let vertex = JanusGraphApi::parse_vertex_from_graphson(value_obj, operation)?;
                        vertices.push(vertex);
                    }
                }
            }
        }
        Ok(vertices)
    }

    pub fn parse_gremlin_edge_list(data: &Value, operation: &str) -> Result<Vec<Edge>, GraphError> {
        let data_vec = if let Some(graphson_obj) = data.as_object() {
            if data.get("@type") == Some(&json!("g:List")) {
                data.get("@value")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.clone())
                    .unwrap_or_default()
            } else {
                vec![data.clone()]
            }
        } else if let Some(arr) = data.as_array() {
            arr.clone()
        } else {
            vec![data.clone()]
        };

        let mut edges = Vec::new();
        for item in data_vec {
            if let Some(edge_obj) = item.as_object() {
                if edge_obj.get("@type") == Some(&json!("g:Edge")) {
                    if let Some(value_obj) = edge_obj.get("@value").and_then(|v| v.as_object()) {
                        let edge = JanusGraphApi::parse_edge_from_graphson(value_obj, operation)?;
                        edges.push(edge);
                    }
                }
            }
        }
        Ok(edges)
    }

    pub fn parse_gremlin_single_vertex(data: &Value, operation: &str) -> Result<Option<Vertex>, GraphError> {
        let vertices = JanusGraphApi::parse_gremlin_vertex_list(data, operation)?;
        Ok(vertices.into_iter().next())
    }

    fn parse_vertex_from_graphson(value_obj: &serde_json::Map<String, Value>, operation: &str) -> Result<Vertex, GraphError> {
        use crate::conversions::from_gremlin_value;
        use golem_graph::golem::graph::types::{ElementId, PropertyValue};

        let id = value_obj
            .get("id")
            .ok_or_else(|| GraphError::InternalError(format!("Missing vertex id in {}", operation)))?;
        
        let vertex_id = match id {
            Value::String(s) => ElementId::StringValue(s.clone()),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ElementId::Int64(i)
                } else {
                    return Err(GraphError::InternalError(format!("Invalid vertex id type in {}", operation)));
                }
            },
            _ => return Err(GraphError::InternalError(format!("Unsupported vertex id type in {}", operation))),
        };

        let label = value_obj
            .get("label")
            .and_then(|l| l.as_str())
            .unwrap_or("vertex")
            .to_string();

        let mut properties = Vec::new();
        if let Some(props_obj) = value_obj.get("properties").and_then(|p| p.as_object()) {
            for (prop_key, prop_value_array) in props_obj {
                if let Some(prop_array) = prop_value_array.as_array() {
                    if let Some(first_prop) = prop_array.first() {
                        if let Some(prop_obj) = first_prop.as_object() {
                            if let Some(prop_value_obj) = prop_obj.get("@value").and_then(|v| v.as_object()) {
                                if let Some(actual_value) = prop_value_obj.get("value") {
                                    if let Ok(converted_value) = from_gremlin_value(actual_value) {
                                        properties.push((prop_key.clone(), converted_value));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Vertex {
            id: vertex_id,
            vertex_type: label,
            properties,
            additional_labels: Vec::new(),
        })
    }

    fn parse_edge_from_graphson(value_obj: &serde_json::Map<String, Value>, operation: &str) -> Result<Edge, GraphError> {
        use crate::conversions::from_gremlin_value;
        use golem_graph::golem::graph::types::{ElementId, PropertyValue};

        let id = value_obj
            .get("id")
            .ok_or_else(|| GraphError::InternalError(format!("Missing edge id in {}", operation)))?;
        
        let edge_id = match id {
            Value::String(s) => ElementId::StringValue(s.clone()),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ElementId::Int64(i)
                } else {
                    return Err(GraphError::InternalError(format!("Invalid edge id type in {}", operation)));
                }
            },
            _ => return Err(GraphError::InternalError(format!("Unsupported edge id type in {}", operation))),
        };

        let label = value_obj
            .get("label")
            .and_then(|l| l.as_str())
            .unwrap_or("edge")
            .to_string();

        let from_vertex = value_obj
            .get("outV")
            .ok_or_else(|| GraphError::InternalError(format!("Missing outV in edge for {}", operation)))?;
        let from_vertex_id = match from_vertex {
            Value::String(s) => ElementId::StringValue(s.clone()),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ElementId::Int64(i)
                } else {
                    return Err(GraphError::InternalError(format!("Invalid from vertex id type in {}", operation)));
                }
            },
            _ => return Err(GraphError::InternalError(format!("Unsupported from vertex id type in {}", operation))),
        };

        let to_vertex = value_obj
            .get("inV")
            .ok_or_else(|| GraphError::InternalError(format!("Missing inV in edge for {}", operation)))?;
        let to_vertex_id = match to_vertex {
            Value::String(s) => ElementId::StringValue(s.clone()),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ElementId::Int64(i)
                } else {
                    return Err(GraphError::InternalError(format!("Invalid to vertex id type in {}", operation)));
                }
            },
            _ => return Err(GraphError::InternalError(format!("Unsupported to vertex id type in {}", operation))),
        };

        let mut properties = Vec::new();
        if let Some(props_obj) = value_obj.get("properties").and_then(|p| p.as_object()) {
            for (prop_key, prop_value) in props_obj {
                if let Ok(converted_value) = from_gremlin_value(prop_value) {
                    properties.push((prop_key.clone(), converted_value));
                }
            }
        }

        Ok(Edge {
            id: edge_id,
            edge_type: label,
            from_vertex: from_vertex_id,
            to_vertex: to_vertex_id,
            properties,
        })
    }
}
