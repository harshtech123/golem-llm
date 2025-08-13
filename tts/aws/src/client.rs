use chrono::Utc;
use golem_tts::config::{get_max_retries_config, get_timeout_config};
use golem_tts::error::{from_reqwest_error, internal_error, tts_error_from_status};
use golem_tts::golem::tts::types::TtsError;
use log::{error, trace};
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::time::Duration;
use sha2::{Sha256, Digest};
use hmac::{Hmac, Mac};
use hex;

type HmacSha256 = Hmac<Sha256>;

/// Helper function to calculate HMAC-SHA256 (same as STT client)
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// Simple URL encoding for query parameters
fn url_encode(input: &str) -> String {
    input
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect()
}

/// Rate limiting configuration for AWS Polly
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_retries: get_max_retries_config(),
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }
}

/// AWS Polly engine types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum Engine {
    #[serde(rename = "standard")]
    Standard,
    #[serde(rename = "neural")]
    #[default]
    Neural,
    #[serde(rename = "long-form")]
    LongForm,
    #[serde(rename = "generative")]
    Generative,
}

/// AWS Polly output formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum OutputFormat {
    #[serde(rename = "json")]
    Json,
    #[serde(rename = "mp3")]
    #[default]
    Mp3,
    #[serde(rename = "ogg_vorbis")]
    OggVorbis,
    #[serde(rename = "pcm")]
    Pcm,
}

/// AWS Polly text types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum TextType {
    #[serde(rename = "text")]
    #[default]
    Text,
    #[serde(rename = "ssml")]
    Ssml,
}

/// AWS Polly speech mark types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SpeechMarkType {
    #[serde(rename = "sentence")]
    Sentence,
    #[serde(rename = "ssml")]
    Ssml,
    #[serde(rename = "viseme")]
    Viseme,
    #[serde(rename = "word")]
    Word,
}

/// AWS Polly API client for text-to-speech operations
/// Based on the working client_stt.rs pattern
#[derive(Debug)]
pub struct AwsPollyTtsApi {
    access_key_id: String,
    secret_access_key: String,
    region: String,
    client: Client,
    base_url: String,
    timeout: Duration,
    max_retries: u32,
}

impl AwsPollyTtsApi {
    pub fn new(
        access_key_id: String,
        secret_access_key: String,
        region: String,
        session_token: Option<String>,
    ) -> Result<Self, TtsError> {
        let timeout_str = std::env::var("TTS_PROVIDER_TIMEOUT").unwrap_or_else(|_| "30".to_string());
        let timeout = Duration::from_secs(timeout_str.parse().unwrap_or(30));
        
        let max_retries_str = std::env::var("TTS_PROVIDER_MAX_RETRIES").unwrap_or_else(|_| "3".to_string());
        let max_retries = max_retries_str.parse().unwrap_or(3);
        
        let base_url = std::env::var("TTS_PROVIDER_ENDPOINT")
            .unwrap_or_else(|_| format!("https://polly.{}.amazonaws.com", region));
        
        // Initialize logging level if specified
        if let Ok(log_level) = std::env::var("TTS_PROVIDER_LOG_LEVEL") {
            trace!("TTS Provider log level set to: {}", log_level);
        }

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|err| from_reqwest_error("Failed to create HTTP client", err))?;

        // Store session token in access_key_id if provided (following STT pattern)
        let final_access_key = if let Some(token) = session_token {
            format!("{}:{}", access_key_id, token)
        } else {
            access_key_id
        };

        Ok(Self {
            access_key_id: final_access_key,
            secret_access_key,
            region,
            client,
            base_url,
            timeout,
            max_retries,
        })
    }

    /// Validate credentials (following STT pattern)
    fn validate_credentials(&self) -> Result<(), TtsError> {
        if self.access_key_id.is_empty() || self.secret_access_key.is_empty() {
            return Err(TtsError::Unauthorized(
                "AWS credentials not properly configured".to_string()
            ));
        }
        
        trace!("AWS credentials basic validation passed");
        Ok(())
    }

    /// Make authenticated REST request with AWS Signature V4 for Polly
    fn make_rest_request<T: Serialize>(
        &self,
        method: Method,
        path: &str,
        body: Option<&T>,
        query_params: Option<&[(&str, &str)]>,
    ) -> Result<Response, reqwest::Error> {
        let mut url = format!("{}{}", self.base_url, path);
        
        // Add query parameters if provided
        if let Some(params) = query_params {
            if !params.is_empty() {
                url.push('?');
                for (i, (key, value)) in params.iter().enumerate() {
                    if i > 0 {
                        url.push('&');
                    }
                    url.push_str(&format!("{}={}", key, url_encode(value)));
                }
            }
        }
        
        let timestamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        
        let request_body = if let Some(body) = body {
            serde_json::to_string(body).unwrap_or_default()
        } else {
            String::new()
        };
        
        let payload_hash = self.sha256_hex(request_body.as_bytes());
        let authorization = self.create_rest_auth_header(&method, path, query_params, &timestamp, &payload_hash);
        
        trace!("AWS Polly REST API request to: {} {}", method, url);
        
        let mut request_builder = self.client
            .request(method, &url)
            .header("Authorization", authorization)
            .header("X-Amz-Date", timestamp);
        
        if !request_body.is_empty() {
            request_builder = request_builder
                .header("Content-Type", "application/json")
                .body(request_body);
        }
        
        request_builder.send()
    }

    /// Execute request with retry logic (following STT pattern)
    #[allow(dead_code)]
    fn execute_with_retry<F>(&self, operation: F) -> Result<Response, TtsError>
    where
        F: Fn() -> Result<Response, TtsError>,
    {
        let mut delay = Duration::from_millis(1000);

        for attempt in 0..=self.max_retries {
            match operation() {
                Ok(response) => {
                    if response.status().is_success() {
                        return Ok(response);
                    } else if (response.status().as_u16() == 429
                        || response.status().as_u16() >= 500)
                        && attempt < self.max_retries
                    {
                        trace!(
                            "Request failed with status {}, retrying in {:?}",
                            response.status(),
                            delay
                        );
                        std::thread::sleep(delay);
                        delay = std::cmp::min(
                            Duration::from_millis((delay.as_millis() as f64 * 2.0) as u64),
                            Duration::from_secs(30),
                        );
                        continue;
                    }

                    let _status = response.status();
                    return Err(self.handle_error_response(response));
                }
                Err(e) => {
                    if self.should_retry(&e) && attempt < self.max_retries {
                        trace!(
                            "Request failed with error: {:?}, retrying in {:?}",
                            e,
                            delay
                        );
                        std::thread::sleep(delay);
                        delay = std::cmp::min(
                            Duration::from_millis((delay.as_millis() as f64 * 2.0) as u64),
                            Duration::from_secs(30),
                        );
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(TtsError::InternalError("Max retries exceeded".to_string()))
    }

    /// Describe available voices using REST API
    pub fn describe_voices(
        &self,
        params: Option<DescribeVoicesParams>,
    ) -> Result<DescribeVoicesResponse, TtsError> {
        trace!("Describing voices");

        // Validate credentials first
        self.validate_credentials()?;

        // Build query parameters
        let mut query_params = Vec::new();
        if let Some(ref p) = params {
            if let Some(ref engine) = p.engine {
                query_params.push(("Engine", match engine {
                    Engine::Standard => "standard",
                    Engine::Neural => "neural", 
                    Engine::LongForm => "long-form",
                    Engine::Generative => "generative",
                }));
            }
            if let Some(ref lang) = p.language_code {
                query_params.push(("LanguageCode", lang.as_str()));
            }
            if let Some(include_additional) = p.include_additional_language_codes {
                query_params.push(("IncludeAdditionalLanguageCodes", if include_additional { "true" } else { "false" }));
            }
            if let Some(ref token) = p.next_token {
                query_params.push(("NextToken", token.as_str()));
            }
        }

        let query_slice = if query_params.is_empty() { 
            None 
        } else { 
            Some(query_params.as_slice()) 
        };

        let mut attempts = 0;
        loop {
            let result = self.make_rest_request(
                Method::GET,
                "/v1/voices",
                None::<&()>,
                query_slice
            );

            match result {
                Ok(response) => {
                    if response.status().is_success() {
                        return parse_response::<DescribeVoicesResponse>(response);
                    } else {
                        let error = self.handle_error_response(response);
                        if self.should_retry(&error) && attempts < self.max_retries {
                            attempts += 1;
                            let delay = Duration::from_secs(2_u64.pow(attempts.min(5)));
                            trace!("Retrying describe_voices in {:?}", delay);
                            std::thread::sleep(delay);
                            continue;
                        }
                        return Err(error);
                    }
                }
                Err(e) => {
                    let tts_error = from_reqwest_error("Failed to send describe_voices request", e);
                    if self.should_retry(&tts_error) && attempts < self.max_retries {
                        attempts += 1;
                        let delay = Duration::from_secs(2_u64.pow(attempts.min(5)));
                        trace!("Retrying describe_voices request in {:?}", delay);
                        std::thread::sleep(delay);
                        continue;
                    }
                    return Err(tts_error);
                }
            }
        }
    }

    /// Synthesize speech from text using REST API
    pub fn synthesize_speech(&self, params: SynthesizeSpeechParams) -> Result<Vec<u8>, TtsError> {
        trace!("Synthesizing speech");
        
        // Validate credentials first
        self.validate_credentials()?;

        let mut attempts = 0;
        loop {
            let result = self.make_rest_request(
                Method::POST,
                "/v1/speech",
                Some(&params),
                None
            );

            match result {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return response.bytes()
                            .map_err(|e| from_reqwest_error("Failed to read audio data", e))
                            .map(|bytes| bytes.to_vec());
                    } else {
                        let error = self.handle_error_response(response);
                        if self.should_retry(&error) && attempts < self.max_retries {
                            attempts += 1;
                            let delay = Duration::from_secs(2_u64.pow(attempts.min(5)));
                            trace!("Retrying synthesize_speech in {:?}", delay);
                            std::thread::sleep(delay);
                            continue;
                        }
                        return Err(error);
                    }
                }
                Err(e) => {
                    let tts_error = from_reqwest_error("Failed to send synthesize_speech request", e);
                    if self.should_retry(&tts_error) && attempts < self.max_retries {
                        attempts += 1;
                        let delay = Duration::from_secs(2_u64.pow(attempts.min(5)));
                        trace!("Retrying synthesize_speech request in {:?}", delay);
                        std::thread::sleep(delay);
                        continue;
                    }
                    return Err(tts_error);
                }
            }
        }
    }

    /// Start speech synthesis task (for long-form content) using REST API
    pub fn start_speech_synthesis_task(
        &self,
        params: StartSpeechSynthesisTaskParams,
    ) -> Result<SpeechSynthesisTask, TtsError> {
        trace!("Starting speech synthesis task");
        
        // Validate credentials first
        self.validate_credentials()?;

        let mut attempts = 0;
        loop {
            let result = self.make_rest_request(
                Method::POST,
                "/v1/synthesisTasks",
                Some(&params),
                None
            );

            match result {
                Ok(response) => {
                    if response.status().is_success() {
                        return parse_response::<SpeechSynthesisTask>(response);
                    } else {
                        let error = self.handle_error_response(response);
                        if self.should_retry(&error) && attempts < self.max_retries {
                            attempts += 1;
                            let delay = Duration::from_secs(2_u64.pow(attempts.min(5)));
                            trace!("Retrying start_speech_synthesis_task in {:?}", delay);
                            std::thread::sleep(delay);
                            continue;
                        }
                        return Err(error);
                    }
                }
                Err(e) => {
                    let tts_error = from_reqwest_error("Failed to send start_speech_synthesis_task request", e);
                    if self.should_retry(&tts_error) && attempts < self.max_retries {
                        attempts += 1;
                        let delay = Duration::from_secs(2_u64.pow(attempts.min(5)));
                        trace!("Retrying start_speech_synthesis_task request in {:?}", delay);
                        std::thread::sleep(delay);
                        continue;
                    }
                    return Err(tts_error);
                }
            }
        }
    }

    /// Get speech synthesis task status using REST API
    pub fn get_speech_synthesis_task(
        &self,
        task_id: &str,
    ) -> Result<SpeechSynthesisTask, TtsError> {
        trace!("Getting speech synthesis task: {}", task_id);
        
        // Validate credentials first
        self.validate_credentials()?;

        let mut attempts = 0;
        loop {
            let result = self.make_rest_request(
                Method::GET,
                &format!("/v1/synthesisTasks/{}", task_id),
                None::<&()>,
                None
            );

            match result {
                Ok(response) => {
                    if response.status().is_success() {
                        return parse_response::<SpeechSynthesisTask>(response);
                    } else {
                        let error = self.handle_error_response(response);
                        if self.should_retry(&error) && attempts < self.max_retries {
                            attempts += 1;
                            let delay = Duration::from_secs(2_u64.pow(attempts.min(5)));
                            trace!("Retrying get_speech_synthesis_task in {:?}", delay);
                            std::thread::sleep(delay);
                            continue;
                        }
                        return Err(error);
                    }
                }
                Err(e) => {
                    let tts_error = from_reqwest_error("Failed to send get_speech_synthesis_task request", e);
                    if self.should_retry(&tts_error) && attempts < self.max_retries {
                        attempts += 1;
                        let delay = Duration::from_secs(2_u64.pow(attempts.min(5)));
                        trace!("Retrying get_speech_synthesis_task request in {:?}", delay);
                        std::thread::sleep(delay);
                        continue;
                    }
                    return Err(tts_error);
                }
            }
        }
    }

    /// Put lexicon for custom pronunciations using REST API
    pub fn put_lexicon(&self, name: &str, content: &str) -> Result<(), TtsError> {
        trace!("Putting lexicon: {}", name);

        // Validate credentials first
        self.validate_credentials()?;

        let request = PutLexiconRequest {
            name: name.to_string(),
            content: content.to_string(),
        };

        let mut attempts = 0;
        loop {
            let result = self.make_rest_request(
                Method::PUT,
                &format!("/v1/lexicons/{}", name),
                Some(&request),
                None
            );

            match result {
                Ok(response) => {
                    if response.status().is_success() {
                        return Ok(());
                    } else {
                        let error = self.handle_error_response(response);
                        if self.should_retry(&error) && attempts < self.max_retries {
                            attempts += 1;
                            let delay = Duration::from_secs(2_u64.pow(attempts.min(5)));
                            trace!("Retrying put_lexicon in {:?}", delay);
                            std::thread::sleep(delay);
                            continue;
                        }
                        return Err(error);
                    }
                }
                Err(e) => {
                    let tts_error = from_reqwest_error("Failed to send put_lexicon request", e);
                    if self.should_retry(&tts_error) && attempts < self.max_retries {
                        attempts += 1;
                        let delay = Duration::from_secs(2_u64.pow(attempts.min(5)));
                        trace!("Retrying put_lexicon request in {:?}", delay);
                        std::thread::sleep(delay);
                        continue;
                    }
                    return Err(tts_error);
                }
            }
        }
    }

    /// Create AWS Signature V4 authorization header for REST API
    fn create_rest_auth_header(
        &self, 
        method: &Method, 
        path: &str, 
        query_params: Option<&[(&str, &str)]>,
        timestamp: &str, 
        payload_hash: &str
    ) -> String {
        let date = &timestamp[0..8];
        let host = format!("polly.{}.amazonaws.com", self.region);
        
        // Build canonical query string
        let canonical_query_string = if let Some(params) = query_params {
            let mut sorted_params = params.to_vec();
            sorted_params.sort_by(|a, b| a.0.cmp(b.0));
            sorted_params
                .iter()
                .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
                .collect::<Vec<_>>()
                .join("&")
        } else {
            String::new()
        };
        
        // Step 1: Create canonical request
        let canonical_headers = format!("host:{}\nx-amz-date:{}", host, timestamp);
        let signed_headers = "host;x-amz-date";
        
        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n\n{}\n{}",
            method.as_str(),
            path,
            canonical_query_string,
            canonical_headers,
            signed_headers,
            payload_hash
        );
        
        let canonical_request_hash = self.sha256_hex(canonical_request.as_bytes());
        
        // Step 2: Create string to sign
        let credential_scope = format!("{}/{}/polly/aws4_request", date, self.region);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            timestamp, credential_scope, canonical_request_hash
        );
        
        // Step 3: Calculate signature
        let signature = self.calculate_signature(&string_to_sign, date);
        
        // Step 4: Create authorization header
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
            self.access_key_id.split(':').next().unwrap_or(&self.access_key_id), 
            credential_scope, 
            signed_headers,
            signature
        )
    }

    /// Calculate AWS Signature V4 signature (following STT pattern)
    fn calculate_signature(&self, string_to_sign: &str, date: &str) -> String {
        // AWS V4 signature derivation
        let date_key = hmac_sha256(format!("AWS4{}", self.secret_access_key).as_bytes(), date.as_bytes());
        let date_region_key = hmac_sha256(&date_key, self.region.as_bytes());
        let date_region_service_key = hmac_sha256(&date_region_key, b"polly");
        let signing_key = hmac_sha256(&date_region_service_key, b"aws4_request");
        
        let signature = hmac_sha256(&signing_key, string_to_sign.as_bytes());
        hex::encode(signature)
    }

    /// Calculate SHA256 hash (following STT pattern)
    fn sha256_hex(&self, data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    /// Determine if error should be retried (following STT pattern)
    fn should_retry(&self, error: &TtsError) -> bool {
        match error {
            TtsError::NetworkError(_) => true,
            TtsError::RateLimited(_) => true,
            TtsError::InternalError(msg) => {
                msg.contains("timeout") || msg.contains("connection") || msg.contains("network")
            }
            _ => false,
        }
    }

    /// Handle error response (following STT pattern)
    fn handle_error_response(&self, response: Response) -> TtsError {
        let status = response.status();
        let error_text = response.text().unwrap_or_else(|_| "Unknown error".to_string());
        
        error!("AWS Polly error response ({}): {}", status, error_text);
        
        match status.as_u16() {
            400 => TtsError::InvalidText(format!("Bad request: {}", error_text)),
            401 | 403 => TtsError::Unauthorized(format!("Authentication failed: {}", error_text)),
            404 => TtsError::VoiceNotFound(format!("Resource not found: {}", error_text)),
            429 => TtsError::RateLimited(60), // Default 60 seconds wait time
            500..=599 => TtsError::InternalError(format!("Server error: {}", error_text)),
            _ => TtsError::InternalError(format!("HTTP {}: {}", status, error_text)),
        }
    }
}

impl Clone for AwsPollyTtsApi {
    fn clone(&self) -> Self {
        Self {
            access_key_id: self.access_key_id.clone(),
            secret_access_key: self.secret_access_key.clone(),
            region: self.region.clone(),
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            timeout: self.timeout,
            max_retries: self.max_retries,
        }
    }
}

// Request/Response Types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeVoicesParams {
    #[serde(rename = "Engine", skip_serializing_if = "Option::is_none")]
    pub engine: Option<Engine>,
    #[serde(rename = "LanguageCode", skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(rename = "IncludeAdditionalLanguageCodes", skip_serializing_if = "Option::is_none")]
    pub include_additional_language_codes: Option<bool>,
    #[serde(rename = "NextToken", skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeVoicesResponse {
    #[serde(rename = "Voices")]
    pub voices: Vec<Voice>,
    #[serde(rename = "NextToken", skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    #[serde(rename = "Gender")]
    pub gender: String,
    #[serde(rename = "Id")]
    pub id: String,
    #[serde(rename = "LanguageCode")]
    pub language_code: String,
    #[serde(rename = "LanguageName")]
    pub language_name: String,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "AdditionalLanguageCodes", skip_serializing_if = "Option::is_none")]
    pub additional_language_codes: Option<Vec<String>>,
    #[serde(rename = "SupportedEngines")]
    pub supported_engines: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizeSpeechParams {
    #[serde(rename = "Engine", skip_serializing_if = "Option::is_none")]
    pub engine: Option<Engine>,
    #[serde(rename = "LanguageCode", skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(rename = "LexiconNames", skip_serializing_if = "Option::is_none")]
    pub lexicon_names: Option<Vec<String>>,
    #[serde(rename = "OutputFormat")]
    pub output_format: OutputFormat,
    #[serde(rename = "SampleRate", skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<String>,
    #[serde(rename = "SpeechMarkTypes", skip_serializing_if = "Option::is_none")]
    pub speech_mark_types: Option<Vec<SpeechMarkType>>,
    #[serde(rename = "Text")]
    pub text: String,
    #[serde(rename = "TextType", skip_serializing_if = "Option::is_none")]
    pub text_type: Option<TextType>,
    #[serde(rename = "VoiceId")]
    pub voice_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartSpeechSynthesisTaskParams {
    #[serde(rename = "Engine", skip_serializing_if = "Option::is_none")]
    pub engine: Option<Engine>,
    #[serde(rename = "LanguageCode", skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(rename = "LexiconNames", skip_serializing_if = "Option::is_none")]
    pub lexicon_names: Option<Vec<String>>,
    #[serde(rename = "OutputFormat")]
    pub output_format: OutputFormat,
    #[serde(rename = "OutputS3BucketName")]
    pub output_s3_bucket_name: String,
    #[serde(rename = "OutputS3KeyPrefix", skip_serializing_if = "Option::is_none")]
    pub output_s3_key_prefix: Option<String>,
    #[serde(rename = "SampleRate", skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<String>,
    #[serde(rename = "SnsTopicArn", skip_serializing_if = "Option::is_none")]
    pub sns_topic_arn: Option<String>,
    #[serde(rename = "SpeechMarkTypes", skip_serializing_if = "Option::is_none")]
    pub speech_mark_types: Option<Vec<SpeechMarkType>>,
    #[serde(rename = "Text")]
    pub text: String,
    #[serde(rename = "TextType", skip_serializing_if = "Option::is_none")]
    pub text_type: Option<TextType>,
    #[serde(rename = "VoiceId")]
    pub voice_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechSynthesisTask {
    #[serde(rename = "CreationTime", skip_serializing_if = "Option::is_none")]
    pub creation_time: Option<f64>,
    #[serde(rename = "Engine", skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(rename = "LanguageCode", skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(rename = "LexiconNames", skip_serializing_if = "Option::is_none")]
    pub lexicon_names: Option<Vec<String>>,
    #[serde(rename = "OutputFormat", skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    #[serde(rename = "OutputUri", skip_serializing_if = "Option::is_none")]
    pub output_uri: Option<String>,
    #[serde(rename = "RequestCharacters", skip_serializing_if = "Option::is_none")]
    pub request_characters: Option<i32>,
    #[serde(rename = "SampleRate", skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<String>,
    #[serde(rename = "SnsTopicArn", skip_serializing_if = "Option::is_none")]
    pub sns_topic_arn: Option<String>,
    #[serde(rename = "SpeechMarkTypes", skip_serializing_if = "Option::is_none")]
    pub speech_mark_types: Option<Vec<String>>,
    #[serde(rename = "TaskId", skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(rename = "TaskStatus", skip_serializing_if = "Option::is_none")]
    pub task_status: Option<String>,
    #[serde(rename = "TaskStatusReason", skip_serializing_if = "Option::is_none")]
    pub task_status_reason: Option<String>,
    #[serde(rename = "TextType", skip_serializing_if = "Option::is_none")]
    pub text_type: Option<String>,
    #[serde(rename = "VoiceId", skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetSpeechSynthesisTaskRequest {
    #[serde(rename = "TaskId")]
    pub task_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutLexiconRequest {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Content")]
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListSpeechSynthesisTasksParams {
    #[serde(rename = "MaxResults", skip_serializing_if = "Option::is_none")]
    pub max_results: Option<i32>,
    #[serde(rename = "NextToken", skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
    #[serde(rename = "Status", skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListSpeechSynthesisTasksResponse {
    #[serde(rename = "NextToken", skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
    #[serde(rename = "SynthesisTasks", skip_serializing_if = "Option::is_none")]
    pub synthesis_tasks: Option<Vec<SpeechSynthesisTask>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetLexiconResponse {
    #[serde(rename = "Lexicon", skip_serializing_if = "Option::is_none")]
    pub lexicon: Option<Lexicon>,
    #[serde(rename = "LexiconAttributes", skip_serializing_if = "Option::is_none")]
    pub lexicon_attributes: Option<LexiconAttributes>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lexicon {
    #[serde(rename = "Content", skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(rename = "Name", skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconAttributes {
    #[serde(rename = "Alphabet", skip_serializing_if = "Option::is_none")]
    pub alphabet: Option<String>,
    #[serde(rename = "LanguageCode", skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(rename = "LastModified", skip_serializing_if = "Option::is_none")]
    pub last_modified: Option<f64>,
    #[serde(rename = "LexemesCount", skip_serializing_if = "Option::is_none")]
    pub lexemes_count: Option<i32>,
    #[serde(rename = "LexiconArn", skip_serializing_if = "Option::is_none")]
    pub lexicon_arn: Option<String>,
    #[serde(rename = "Size", skip_serializing_if = "Option::is_none")]
    pub size: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListLexiconsParams {
    #[serde(rename = "NextToken", skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListLexiconsResponse {
    #[serde(rename = "Lexicons", skip_serializing_if = "Option::is_none")]
    pub lexicons: Option<Vec<LexiconDescription>>,
    #[serde(rename = "NextToken", skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconDescription {
    #[serde(rename = "Attributes", skip_serializing_if = "Option::is_none")]
    pub attributes: Option<LexiconAttributes>,
    #[serde(rename = "Name", skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, TtsError> {
    let status = response.status();
    if !status.is_success() {
        return Err(tts_error_from_status(status));
    }

    let response_text = response
        .text()
        .map_err(|e| from_reqwest_error("Failed to read response", e))?;

    trace!("AWS Polly API response: {}", response_text);
    
    // Add additional logging to help debug voice parsing
    if response_text.contains("voice") || response_text.contains("Voice") {
        error!("DEBUG - Raw response contains voice data: {}", response_text);
    }

    let parsed_result = serde_json::from_str(&response_text)
        .map_err(|e| {
            error!("Failed to parse AWS Polly response: {}, Raw response: {}", e, response_text);
            internal_error(format!("Failed to parse AWS Polly response: {}", e))
        })?;
        
    trace!("Parsed response: {:?}", parsed_result);
    Ok(parsed_result)
}
