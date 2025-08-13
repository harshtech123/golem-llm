use chrono::{DateTime, Utc};
use golem_tts::config::{get_max_retries_config, get_timeout_config};
use golem_tts::error::{from_reqwest_error, internal_error, tts_error_from_status};
use golem_tts::golem::tts::types::TtsError;
use log::{trace, error};
use reqwest::{Client, Method, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;
use sha2::{Digest, Sha256};
use hmac_sha256::HMAC;

/// Helper function to calculate HMAC-SHA256 using hmac_sha256 crate (like Kling auth)
fn hmac_sha256(key: &[u8], data: &str) -> Vec<u8> {
    HMAC::mac(data.as_bytes(), key).to_vec()
}

/// Simple hex encoding function
fn hex_encode(data: &[u8]) -> String {
    data.iter().map(|byte| format!("{:02x}", byte)).collect()
}


/// Rate limiting configuration for AWS Polly
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
/// Based on https://docs.aws.amazon.com/polly/latest/dg/API_Reference.html
#[derive(Clone)]
pub struct AwsPollyTtsApi {
    client: Client,
    access_key_id: String,
    #[allow(dead_code)]
    secret_access_key: String,
    session_token: Option<String>,
    #[allow(dead_code)]
    region: String,
    base_url: String,
    rate_limit_config: RateLimitConfig,
}

impl AwsPollyTtsApi {
    pub fn new(
        access_key_id: String,
        secret_access_key: String,
        region: String,
        session_token: Option<String>,
    ) -> Result<Self, TtsError> {
        let base_url = format!("https://polly.{}.amazonaws.com", region);

        let client = Client::builder()
            .timeout(Duration::from_secs(get_timeout_config()))
            .build()
            .map_err(|err| from_reqwest_error("Failed to create HTTP client", err))?;

        Ok(Self {
            client,
            access_key_id,
            secret_access_key,
            session_token,
            region: region.clone(),
            base_url,
            rate_limit_config: RateLimitConfig::default(),
        })
    }

    /// Create an authenticated request with AWS Signature Version 4
    fn create_request(&self, method: Method, url: &str, _body: Option<&str>) -> RequestBuilder {
        self.client.request(method, url)
            .header("User-Agent", "golem-aws-polly-client/1.0")
    }

    /// Create AWS Signature Version 4 headers
    fn create_aws_headers(
        &self,
        method: &str,
        url: &str,
        body: Option<&str>,
        datetime: &DateTime<Utc>,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        // Simple URL parsing without url crate
        let url_parts: Vec<&str> = url.split("://").collect();
        if url_parts.len() != 2 {
            panic!("Invalid URL format");
        }
        
        let after_protocol = url_parts[1];
        let host_and_path: Vec<&str> = after_protocol.splitn(2, '/').collect();
        let host = host_and_path[0];
        let path = if host_and_path.len() > 1 {
            format!("/{}", host_and_path[1])
        } else {
            "/".to_string()
        };

        // Split path and query
        let path_parts: Vec<&str> = path.splitn(2, '?').collect();
        let path_only = path_parts[0];
        let query = if path_parts.len() > 1 { path_parts[1] } else { "" };

        // Basic headers that will be signed
        let content_type = "application/x-amz-json-1.0";
        let amz_date = datetime.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = datetime.format("%Y%m%d").to_string();
        
        headers.insert("Content-Type".to_string(), content_type.to_string());
        headers.insert("Host".to_string(), host.to_string());
        headers.insert("X-Amz-Date".to_string(), amz_date.clone());

        if let Some(ref token) = self.session_token {
            headers.insert("X-Amz-Security-Token".to_string(), token.clone());
         }

        // Create the canonical request
        let payload_hash = if let Some(body_str) = body {
            let mut hasher = Sha256::new();
            hasher.update(body_str.as_bytes());
            hex_encode(&hasher.finalize())
        } else {
            let mut hasher = Sha256::new();
            hasher.update(b"");
            hex_encode(&hasher.finalize())
        };

        // Create signed headers (sorted by header name)
        let mut signed_headers_vec: Vec<_> = headers.keys().map(|k| k.to_lowercase()).collect();
        signed_headers_vec.sort();
        let signed_headers = signed_headers_vec.join(";");

        // Create canonical headers
        let mut canonical_headers = String::new();
        for header_name in &signed_headers_vec {
            let header_value = headers.iter()
                .find(|(k, _)| k.to_lowercase() == *header_name)
                .map(|(_, v)| v)
                .unwrap();
            canonical_headers.push_str(&format!("{}:{}\n", header_name, header_value.trim()));
        }

        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method,
            path,
            query,
            canonical_headers,
            signed_headers,
            payload_hash
        );


        // Create the string to sign
        let algorithm = "AWS4-HMAC-SHA256";
        let credential_scope = format!("{}/{}/polly/aws4_request", date_stamp, self.region);
        let string_to_sign = format!(
            "{}\n{}\n{}\n{}",
            algorithm,
            amz_date,
            credential_scope,
            {
                let mut hasher = Sha256::new();
                hasher.update(canonical_request.as_bytes());
                hex_encode(&hasher.finalize())
            }
        );


        // Calculate the signature
        let signature = self.calculate_signature(&string_to_sign, &date_stamp);

        // Create authorization header
        let authorization_header = format!(
            "{} Credential={}/{}, SignedHeaders={}, Signature={}",
            algorithm,
            self.access_key_id,
            credential_scope,
            signed_headers,
            signature
        );

        headers.insert("Authorization".to_string(), authorization_header);

        headers
    }

    /// Calculate AWS Signature Version 4 signature
    fn calculate_signature(&self, string_to_sign: &str, date_stamp: &str) -> String {
        // Step 1: Create the signing key
        let k_date = hmac_sha256(format!("AWS4{}", self.secret_access_key).as_bytes(), date_stamp);
        let k_region = hmac_sha256(&k_date, &self.region);
        let k_service = hmac_sha256(&k_region, "polly");
        let k_signing = hmac_sha256(&k_service, "aws4_request");

        // Step 2: Calculate the signature
        let signature = hmac_sha256(&k_signing, string_to_sign);
        hex_encode(&signature)
    }

    /// Execute a request with retry logic for rate limiting
    fn execute_with_retry<F>(&self, operation: F) -> Result<Response, TtsError>
    where
        F: Fn() -> Result<Response, TtsError>,
    {
        let mut delay = self.rate_limit_config.initial_delay;
        let max_retries = self.rate_limit_config.max_retries;

        trace!("execute_with_retry - Starting with max_retries: {}", max_retries);

        for attempt in 0..=max_retries {
            trace!("execute_with_retry - Attempt {}/{}", attempt + 1, max_retries + 1);

            match operation() {
                Ok(response) => {
                    let status = response.status();
                    trace!("execute_with_retry - Response status: {}", status);

                    if status.as_u16() < 200 || status.as_u16() >= 300 {
                        if (status.as_u16() == 429 || status.as_u16() == 503) && attempt < max_retries {
                            // Rate limited or service unavailable, retry
                            trace!("execute_with_retry - Rate limited ({}), retrying in {:?}", status, delay);
                            std::thread::sleep(delay);
                            delay = std::cmp::min(
                                Duration::from_millis(
                                    (delay.as_millis() as f64 * self.rate_limit_config.backoff_multiplier) as u64,
                                ),
                                self.rate_limit_config.max_delay,
                            );
                            continue;
                        } else {
                            trace!("execute_with_retry - Non-retryable error or max retries reached: {}", status);
                            return Err(tts_error_from_status(status));
                        }
                    }

                    trace!("execute_with_retry - Success on attempt {}", attempt + 1);
                    return Ok(response);
                }
                Err(e) => {
                    error!("execute_with_retry - Error on attempt {}: {:?}", attempt + 1, e);

                    if attempt < max_retries {
                        trace!("execute_with_retry - Retrying after error in {:?}", delay);
                        std::thread::sleep(delay);
                        delay = std::cmp::min(
                            Duration::from_millis(
                                (delay.as_millis() as f64 * self.rate_limit_config.backoff_multiplier) as u64,
                            ),
                            self.rate_limit_config.max_delay,
                        );
                    } else {
                        trace!("execute_with_retry - Max retries exceeded with error: {:?}", e);
                        return Err(e);
                    }
                }
            }
        }

        trace!("execute_with_retry - Max retries exceeded");
        Err(TtsError::InternalError("Max retries exceeded".to_string()))
    }

    /// Describe available voices
    pub fn describe_voices(
        &self,
        params: Option<DescribeVoicesParams>,
    ) -> Result<DescribeVoicesResponse, TtsError> {
        trace!("Describing voices");

        let mut url = format!("{}/v1/voices", self.base_url);
        let mut query_params = Vec::new();

        if let Some(p) = params {
            if let Some(engine) = p.engine {
                query_params.push(format!(
                    "Engine={}",
                    serde_json::to_string(&engine)
                        .unwrap_or_default()
                        .trim_matches('"')
                ));
            }
            if let Some(language_code) = p.language_code {
                query_params.push(format!("LanguageCode={}", language_code));
            }
            if let Some(include_additional_language_codes) = p.include_additional_language_codes {
                query_params.push(format!(
                    "IncludeAdditionalLanguageCodes={}",
                    include_additional_language_codes
                ));
            }
            if let Some(next_token) = p.next_token {
                query_params.push(format!("NextToken={}", next_token));
            }
        }

        if !query_params.is_empty() {
            url = format!("{}?{}", url, query_params.join("&"));
        }


        let response = self.execute_with_retry(|| {
            let datetime = Utc::now();
            let headers = self.create_aws_headers("GET", &url, None, &datetime);
                        
            let mut request = self.create_request(Method::GET, &url, None);
            for (key, value) in &headers {
                request = request.header(key, value);
            }
            
            request.send().map_err(|e| {
                let request_details = format!(
                    "GET {} with headers: {:?}", 
                    url, 
                    headers.iter().map(|(k, v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(", ")
                );
                from_reqwest_error(&format!("Failed to send describe_voices request: {}", request_details), e)
            })
        })?;

        parse_response(response)
    }

    /// Synthesize speech from text
    pub fn synthesize_speech(&self, params: SynthesizeSpeechParams) -> Result<Vec<u8>, TtsError> {
        trace!("Synthesizing speech");
        println!("[DEBUG] AWS Polly synthesize_speech - Starting with params: voice_id={}, text_len={}, format={:?}", 
                params.voice_id, params.text.len(), params.output_format);

        let url = format!("{}/v1/speech", self.base_url);
        let body = serde_json::to_string(&params)
            .map_err(|e| internal_error(format!("Failed to serialize request: {}", e)))?;

        println!("[DEBUG] AWS Polly synthesize_speech - Request body: {}", body);

        let response = self.execute_with_retry(|| {
            let datetime = Utc::now();
            let headers = self.create_aws_headers("POST", &url, Some(&body), &datetime);
            
            println!("[DEBUG] AWS Polly synthesize_speech - Request headers: {:?}", headers);
            
            let mut request = self.create_request(Method::POST, &url, Some(&body));
            for (key, value) in &headers {
                request = request.header(key, value);
            }
            
            request.body(body.clone()).send().map_err(|e| {
                println!("[DEBUG] AWS Polly synthesize_speech - Request failed: {:?}", e);
                from_reqwest_error("Failed to send synthesize_speech request", e)
            })
        })?;

        // Check if response is successful
        let status = response.status();
        println!("[DEBUG] AWS Polly synthesize_speech - Final response status: {}", status);
        
        if status.as_u16() < 200 || status.as_u16() >= 300 {
            return Err(tts_error_from_status(status));
        }

        // Get the audio data
        let audio_data = response.bytes().map_err(|e| {
            println!("[DEBUG] AWS Polly synthesize_speech - Failed to read audio data: {}", e);
            TtsError::InternalError(format!("Failed to read audio data: {}", e))
        })?;
            
        println!("[DEBUG] AWS Polly synthesize_speech - Successfully read {} bytes of audio data", audio_data.len());
        trace!("synthesize_speech - Audio data size: {} bytes", audio_data.len());
        Ok(audio_data.to_vec())
    }

    /// Start speech synthesis task (for long-form content)
    pub fn start_speech_synthesis_task(
        &self,
        params: StartSpeechSynthesisTaskParams,
    ) -> Result<SpeechSynthesisTask, TtsError> {
        trace!("Starting speech synthesis task");

        let url = format!("{}/v1/synthesisTasks", self.base_url);
        let body = serde_json::to_string(&params)
            .map_err(|e| internal_error(format!("Failed to serialize request: {}", e)))?;

        let response = self.execute_with_retry(|| {
            let datetime = Utc::now();
            let headers = self.create_aws_headers("POST", &url, Some(&body), &datetime);
            
            println!("[DEBUG] AWS Polly start_speech_synthesis_task - Request headers: {:?}", headers);
            
            let mut request = self.create_request(Method::POST, &url, Some(&body));
            for (key, value) in &headers {
                request = request.header(key, value);
            }
            
            request.body(body.clone()).send().map_err(|e| {
                println!("[DEBUG] AWS Polly start_speech_synthesis_task - Request failed: {:?}", e);
                from_reqwest_error("Failed to send start_speech_synthesis_task request", e)
            })
        })?;

        parse_response(response)
    }

    /// Get speech synthesis task status
    pub fn get_speech_synthesis_task(
        &self,
        task_id: &str,
    ) -> Result<SpeechSynthesisTask, TtsError> {
       trace!("Getting speech synthesis task: {}", task_id);

        let url = format!("{}/v1/synthesisTasks/{}", self.base_url, task_id);

        let response = self.execute_with_retry(|| {
            let datetime = Utc::now();
            let headers = self.create_aws_headers("GET", &url, None, &datetime);
            
            println!("[DEBUG] AWS Polly get_speech_synthesis_task - Request headers: {:?}", headers);
            
            let mut request = self.create_request(Method::GET, &url, None);
            for (key, value) in &headers {
                request = request.header(key, value);
            }
            
            request.send().map_err(|e| {
                println!("[DEBUG] AWS Polly get_speech_synthesis_task - Request failed: {:?}", e);
                from_reqwest_error("Failed to send get_speech_synthesis_task request", e)
            })
        })?;

        parse_response(response)
    }

    /// List speech synthesis tasks
    pub fn _list_speech_synthesis_tasks(
        &self,
        params: Option<ListSpeechSynthesisTasksParams>,
    ) -> Result<ListSpeechSynthesisTasksResponse, TtsError> {
        trace!("Listing speech synthesis tasks");

        let mut url = format!("{}/v1/synthesisTasks", self.base_url);
        let mut query_params = Vec::new();

        if let Some(p) = params {
            if let Some(max_results) = p.max_results {
                query_params.push(format!("MaxResults={}", max_results));
            }
            if let Some(next_token) = p.next_token {
                query_params.push(format!("NextToken={}", next_token));
            }
            if let Some(status) = p.status {
                query_params.push(format!("Status={}", status));
            }
        }

        if !query_params.is_empty() {
            url = format!("{}?{}", url, query_params.join("&"));
        }

        let response = self.execute_with_retry(|| {
            let datetime = Utc::now();
            let headers = self.create_aws_headers("GET", &url, None, &datetime);
            
            println!("[DEBUG] AWS Polly list_speech_synthesis_tasks - Request headers: {:?}", headers);
            
            let mut request = self.create_request(Method::GET, &url, None);
            for (key, value) in &headers {
                request = request.header(key, value);
            }
            
            request.send().map_err(|e| {
                println!("[DEBUG] AWS Polly list_speech_synthesis_tasks - Request failed: {:?}", e);
                from_reqwest_error("Failed to send list_speech_synthesis_tasks request", e)
            })
        })?;

        parse_response(response)
    }

    /// Put lexicon for custom pronunciations
    pub fn put_lexicon(&self, name: &str, content: &str) -> Result<(), TtsError> {
        trace!("Putting lexicon: {}", name);

        let url = format!("{}/v1/lexicons/{}", self.base_url, name);
        let request_body = PutLexiconRequest {
            content: content.to_string(),
        };
        let body = serde_json::to_string(&request_body)
            .map_err(|e| internal_error(format!("Failed to serialize request: {}", e)))?;

        let response = self.execute_with_retry(|| {
            let datetime = Utc::now();
            let headers = self.create_aws_headers("PUT", &url, Some(&body), &datetime);
            
            println!("[DEBUG] AWS Polly put_lexicon - Request headers: {:?}", headers);
            
            let mut request = self.create_request(Method::PUT, &url, Some(&body));
            for (key, value) in &headers {
                request = request.header(key, value);
            }
            
            request.body(body.clone()).send().map_err(|e| {
                println!("[DEBUG] AWS Polly put_lexicon - Request failed: {:?}", e);
                from_reqwest_error("Failed to send put_lexicon request", e)
            })
        })?;

        if response.status().as_u16() >= 200 && response.status().as_u16() < 300 {
            Ok(())
        } else {
            Err(tts_error_from_status(response.status()))
        }
    }
}

// Request/Response Types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeVoicesParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<Engine>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_additional_language_codes: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeVoicesResponse {
    #[serde(rename = "Voices")]
    pub voices: Vec<Voice>,
    #[serde(rename = "NextToken")]
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(rename = "AdditionalLanguageCodes")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_language_codes: Option<Vec<String>>,
    #[serde(rename = "SupportedEngines")]
    pub supported_engines: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizeSpeechParams {
    #[serde(rename = "Engine")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<Engine>,
    #[serde(rename = "LanguageCode")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(rename = "LexiconNames")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexicon_names: Option<Vec<String>>,
    #[serde(rename = "OutputFormat")]
    pub output_format: OutputFormat,
    #[serde(rename = "SampleRate")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<String>,
    #[serde(rename = "SpeechMarkTypes")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_mark_types: Option<Vec<SpeechMarkType>>,
    #[serde(rename = "Text")]
    pub text: String,
    #[serde(rename = "TextType")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_type: Option<TextType>,
    #[serde(rename = "VoiceId")]
    pub voice_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartSpeechSynthesisTaskParams {
    #[serde(rename = "Engine")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<Engine>,
    #[serde(rename = "LanguageCode")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(rename = "LexiconNames")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexicon_names: Option<Vec<String>>,
    #[serde(rename = "OutputFormat")]
    pub output_format: OutputFormat,
    #[serde(rename = "OutputS3BucketName")]
    pub output_s3_bucket_name: String,
    #[serde(rename = "OutputS3KeyPrefix")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_s3_key_prefix: Option<String>,
    #[serde(rename = "SampleRate")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<String>,
    #[serde(rename = "SnsTopicArn")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sns_topic_arn: Option<String>,
    #[serde(rename = "SpeechMarkTypes")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_mark_types: Option<Vec<SpeechMarkType>>,
    #[serde(rename = "Text")]
    pub text: String,
    #[serde(rename = "TextType")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_type: Option<TextType>,
    #[serde(rename = "VoiceId")]
    pub voice_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechSynthesisTask {
    #[serde(rename = "CreationTime")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub creation_time: Option<f64>,
    #[serde(rename = "Engine")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(rename = "LanguageCode")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(rename = "LexiconNames")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexicon_names: Option<Vec<String>>,
    #[serde(rename = "OutputFormat")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    #[serde(rename = "OutputUri")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_uri: Option<String>,
    #[serde(rename = "RequestCharacters")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_characters: Option<i32>,
    #[serde(rename = "SampleRate")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<String>,
    #[serde(rename = "SnsTopicArn")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sns_topic_arn: Option<String>,
    #[serde(rename = "SpeechMarkTypes")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_mark_types: Option<Vec<String>>,
    #[serde(rename = "TaskId")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(rename = "TaskStatus")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_status: Option<String>,
    #[serde(rename = "TaskStatusReason")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_status_reason: Option<String>,
    #[serde(rename = "TextType")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_type: Option<String>,
    #[serde(rename = "VoiceId")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListSpeechSynthesisTasksParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListSpeechSynthesisTasksResponse {
    #[serde(rename = "NextToken")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
    #[serde(rename = "SynthesisTasks")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synthesis_tasks: Option<Vec<SpeechSynthesisTask>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutLexiconRequest {
    #[serde(rename = "Content")]
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetLexiconResponse {
    #[serde(rename = "Lexicon")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexicon: Option<Lexicon>,
    #[serde(rename = "LexiconAttributes")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexicon_attributes: Option<LexiconAttributes>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lexicon {
    #[serde(rename = "Content")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(rename = "Name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconAttributes {
    #[serde(rename = "Alphabet")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alphabet: Option<String>,
    #[serde(rename = "LanguageCode")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(rename = "LastModified")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_modified: Option<f64>,
    #[serde(rename = "LexemesCount")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexemes_count: Option<i32>,
    #[serde(rename = "LexiconArn")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexicon_arn: Option<String>,
    #[serde(rename = "Size")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListLexiconsParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListLexiconsResponse {
    #[serde(rename = "Lexicons")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexicons: Option<Vec<LexiconDescription>>,
    #[serde(rename = "NextToken")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconDescription {
    #[serde(rename = "Attributes")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<LexiconAttributes>,
    #[serde(rename = "Name")]
    #[serde(skip_serializing_if = "Option::is_none")]
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

    serde_json::from_str(&response_text)
        .map_err(|e| internal_error(format!("Failed to parse AWS Polly response: {}", e)))
}
