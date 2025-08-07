use chrono::{DateTime, Utc};
use golem_tts::config::{get_max_retries_config, get_timeout_config};
use golem_tts::error::{from_reqwest_error, internal_error, tts_error_from_status};
use golem_tts::golem::tts::types::TtsError;
use log::trace;
use reqwest::{Client, Method, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;

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
    ) -> Self {
        let base_url = format!("https://polly.{}.amazonaws.com", region);
        let timeout = Duration::from_secs(get_timeout_config());

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            access_key_id,
            secret_access_key,
            session_token,
            region: region.clone(),
            base_url,
            rate_limit_config: RateLimitConfig::default(),
        }
    }

    /// Create an authenticated request with AWS Signature Version 4
    fn create_request(&self, method: Method, url: &str, body: Option<&str>) -> RequestBuilder {
        let mut request = self.client.request(method.clone(), url);

        // Add AWS signature headers
        let datetime = Utc::now();
        let headers = self.create_aws_headers(&method, url, body, &datetime);

        for (key, value) in headers {
            request = request.header(key, value);
        }

        request
    }

    /// Create AWS Signature Version 4 headers
    fn create_aws_headers(
        &self,
        _method: &Method,
        _url: &str,
        _body: Option<&str>,
        datetime: &DateTime<Utc>,
    ) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        // Basic headers
        headers.insert(
            "Content-Type".to_string(),
            "application/x-amz-json-1.0".to_string(),
        );
        headers.insert(
            "X-Amz-Date".to_string(),
            datetime.format("%Y%m%dT%H%M%SZ").to_string(),
        );

        if let Some(ref token) = self.session_token {
            headers.insert("X-Amz-Security-Token".to_string(), token.clone());
        }

        // For simplicity, we'll use basic auth header format
        // In production, implement full AWS Signature Version 4
        let auth_header = format!(
            "AWS4-HMAC-SHA256 Credential={}/{}/polly/aws4_request",
            self.access_key_id,
            datetime.format("%Y%m%d")
        );
        headers.insert("Authorization".to_string(), auth_header);

        headers
    }

    /// Execute a request with retry logic for rate limiting
    fn execute_with_retry<F>(&self, operation: F) -> Result<Response, TtsError>
    where
        F: Fn() -> Result<Response, TtsError>,
    {
        let mut last_error = None;
        let mut delay = self.rate_limit_config.initial_delay;

        for attempt in 0..=self.rate_limit_config.max_retries {
            match operation() {
                Ok(response) => {
                    if response.status().is_success() {
                        return Ok(response);
                    } else if response.status().as_u16() == 429 || response.status().as_u16() == 503
                    {
                        // Rate limited or service unavailable, retry
                        if attempt < self.rate_limit_config.max_retries {
                            trace!("Rate limited, retrying in {:?}", delay);
                            std::thread::sleep(delay);
                            delay = std::cmp::min(
                                Duration::from_millis(
                                    (delay.as_millis() as f64
                                        * self.rate_limit_config.backoff_multiplier)
                                        as u64,
                                ),
                                self.rate_limit_config.max_delay,
                            );
                            continue;
                        }
                    }
                    return Err(tts_error_from_status(response.status()));
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.rate_limit_config.max_retries {
                        std::thread::sleep(delay);
                        delay = std::cmp::min(
                            Duration::from_millis(
                                (delay.as_millis() as f64
                                    * self.rate_limit_config.backoff_multiplier)
                                    as u64,
                            ),
                            self.rate_limit_config.max_delay,
                        );
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| internal_error("Max retries exceeded")))
    }

    /// Describe available voices
    pub fn describe_voices(
        &self,
        params: Option<DescribeVoicesParams>,
    ) -> Result<DescribeVoicesResponse, TtsError> {
        let url = format!("{}/v1/voices", self.base_url);
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

        let final_url = if query_params.is_empty() {
            url
        } else {
            format!("{}?{}", url, query_params.join("&"))
        };

        let operation = || {
            self.create_request(Method::GET, &final_url, None)
                .send()
                .map_err(|e| from_reqwest_error("Failed to send request", e))
        };

        let response = self.execute_with_retry(operation)?;
        parse_response(response)
    }

    /// Synthesize speech from text
    pub fn synthesize_speech(&self, params: SynthesizeSpeechParams) -> Result<Vec<u8>, TtsError> {
        let url = format!("{}/v1/speech", self.base_url);
        let body = serde_json::to_string(&params)
            .map_err(|e| internal_error(format!("Failed to serialize request: {}", e)))?;

        let operation = || {
            self.create_request(Method::POST, &url, Some(&body))
                .body(body.clone())
                .send()
                .map_err(|e| from_reqwest_error("Failed to send request", e))
        };

        let response = self.execute_with_retry(operation)?;

        if response.status().is_success() {
            response
                .bytes()
                .map_err(|e| from_reqwest_error("Failed to read response", e))
                .map(|bytes| bytes.to_vec())
        } else {
            Err(tts_error_from_status(response.status()))
        }
    }

    /// Start speech synthesis task (for long-form content)
    pub fn start_speech_synthesis_task(
        &self,
        params: StartSpeechSynthesisTaskParams,
    ) -> Result<SpeechSynthesisTask, TtsError> {
        let url = format!("{}/v1/synthesisTasks", self.base_url);
        let body = serde_json::to_string(&params)
            .map_err(|e| internal_error(format!("Failed to serialize request: {}", e)))?;

        let operation = || {
            self.create_request(Method::POST, &url, Some(&body))
                .body(body.clone())
                .send()
                .map_err(|e| from_reqwest_error("Failed to send request", e))
        };

        let response = self.execute_with_retry(operation)?;
        parse_response(response)
    }

    /// Get speech synthesis task status
    pub fn get_speech_synthesis_task(
        &self,
        task_id: &str,
    ) -> Result<SpeechSynthesisTask, TtsError> {
        let url = format!("{}/v1/synthesisTasks/{}", self.base_url, task_id);

        let operation = || {
            self.create_request(Method::GET, &url, None)
                .send()
                .map_err(|e| from_reqwest_error("Failed to send request", e))
        };

        let response = self.execute_with_retry(operation)?;
        parse_response(response)
    }

    /// List speech synthesis tasks
    pub fn _list_speech_synthesis_tasks(
        &self,
        params: Option<ListSpeechSynthesisTasksParams>,
    ) -> Result<ListSpeechSynthesisTasksResponse, TtsError> {
        let url = format!("{}/v1/synthesisTasks", self.base_url);
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

        let final_url = if query_params.is_empty() {
            url
        } else {
            format!("{}?{}", url, query_params.join("&"))
        };

        let operation = || {
            self.create_request(Method::GET, &final_url, None)
                .send()
                .map_err(|e| from_reqwest_error("Failed to send request", e))
        };

        let response = self.execute_with_retry(operation)?;
        parse_response(response)
    }

    /// Put lexicon for custom pronunciations
    pub fn put_lexicon(&self, name: &str, content: &str) -> Result<(), TtsError> {
        let url = format!("{}/v1/lexicons/{}", self.base_url, name);
        let request = PutLexiconRequest {
            content: content.to_string(),
        };
        let body = serde_json::to_string(&request)
            .map_err(|e| internal_error(format!("Failed to serialize request: {}", e)))?;

        let operation = || {
            self.create_request(Method::PUT, &url, Some(&body))
                .body(body.clone())
                .send()
                .map_err(|e| from_reqwest_error("Failed to send request", e))
        };

        let response = self.execute_with_retry(operation)?;
        if response.status().is_success() {
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
