use golem_tts::error::{from_reqwest_error, internal_error, tts_error_from_status};
use golem_tts::golem::tts::types::TtsError;
use log::trace;
use reqwest::{Client, Method, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// The ElevenLabs TTS API client for managing voices and performing text-to-speech
/// Based on https://elevenlabs.io/docs/api-reference/
#[derive(Clone)]
pub struct ElevenLabsTtsApi {
    client: Client,
    api_key: String,
    base_url: String,
}

impl ElevenLabsTtsApi {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");

        let base_url = "https://api.elevenlabs.io".to_string();

        Self {
            api_key,
            client,
            base_url,
        }
    }

    fn create_request(&self, method: Method, url: &str) -> RequestBuilder {
        self.client
            .request(method, url)
            .header("xi-api-key", &self.api_key)
            .header("Content-Type", "application/json")
    }

    /// Get a list of available voices
    pub fn list_voices(&self, params: Option<ListVoicesParams>) -> Result<ListVoicesResponse, TtsError> {
        trace!("Listing voices");

        let mut url = format!("{}/v2/voices", self.base_url);
        
        if let Some(params) = params {
            let mut query_params = Vec::new();
            
            if let Some(page_size) = params.page_size {
                query_params.push(format!("page_size={}", page_size));
            }
            if let Some(search) = params.search {
                query_params.push(format!("search={}", urlencoding::encode(&search)));
            }
            if let Some(sort) = params.sort {
                query_params.push(format!("sort={}", sort));
            }
            if let Some(sort_direction) = params.sort_direction {
                query_params.push(format!("sort_direction={}", sort_direction));
            }
            if let Some(voice_type) = params.voice_type {
                query_params.push(format!("voice_type={}", voice_type));
            }
            if let Some(category) = params.category {
                query_params.push(format!("category={}", category));
            }
            if let Some(next_page_token) = params.next_page_token {
                query_params.push(format!("next_page_token={}", urlencoding::encode(&next_page_token)));
            }
            if let Some(include_total_count) = params.include_total_count {
                query_params.push(format!("include_total_count={}", include_total_count));
            }
            
            if !query_params.is_empty() {
                url.push('?');
                url.push_str(&query_params.join("&"));
            }
        }

        let response = self
            .create_request(Method::GET, &url)
            .send()
            .map_err(|e| internal_error(format!("Failed to list voices: {e}")))?;

        parse_response(response)
    }

    /// Get a specific voice by ID
    pub fn get_voice(&self, voice_id: &str) -> Result<Voice, TtsError> {
        trace!("Getting voice: {voice_id}");

        let url = format!("{}/v1/voices/{}", self.base_url, voice_id);

        let response = self
            .create_request(Method::GET, &url)
            .send()
            .map_err(|e| internal_error(format!("Failed to get voice: {e}")))?;

        parse_response(response)
    }

    /// Convert text to speech
    pub fn text_to_speech(
        &self,
        voice_id: &str,
        request: &TextToSpeechRequest,
        params: Option<TextToSpeechParams>,
    ) -> Result<Vec<u8>, TtsError> {
        trace!("Converting text to speech with voice: {voice_id}");

        let mut url = format!("{}/v1/text-to-speech/{}", self.base_url, voice_id);
        
        if let Some(params) = params {
            let mut query_params = Vec::new();
            
            if let Some(enable_logging) = params.enable_logging {
                query_params.push(format!("enable_logging={}", enable_logging));
            }
            if let Some(optimize_streaming_latency) = params.optimize_streaming_latency {
                query_params.push(format!("optimize_streaming_latency={}", optimize_streaming_latency));
            }
            if let Some(output_format) = params.output_format {
                query_params.push(format!("output_format={}", output_format));
            }
            
            if !query_params.is_empty() {
                url.push('?');
                url.push_str(&query_params.join("&"));
            }
        }

        let response = self
            .create_request(Method::POST, &url)
            .json(request)
            .send()
            .map_err(|e| internal_error(format!("Failed to synthesize speech: {e}")))?;

        if response.status().is_success() {
            let audio_data = response
                .bytes()
                .map_err(|err| from_reqwest_error("Failed to read audio response", err))?;
            Ok(audio_data.to_vec())
        } else {
            let status = response.status();
            let error_body = response
                .text()
                .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

            trace!("Received {status} response from ElevenLabs API: {error_body:?}");
            Err(tts_error_from_status(status))
        }
    }

    /// Stream text to speech (returns response body for streaming)
    pub fn text_to_speech_stream(
        &self,
        voice_id: &str,
        request: &TextToSpeechRequest,
        params: Option<TextToSpeechParams>,
    ) -> Result<reqwest::Response, TtsError> {
        trace!("Streaming text to speech with voice: {voice_id}");

        let mut url = format!("{}/v1/text-to-speech/{}/stream", self.base_url, voice_id);
        
        if let Some(params) = params {
            let mut query_params = Vec::new();
            
            if let Some(enable_logging) = params.enable_logging {
                query_params.push(format!("enable_logging={}", enable_logging));
            }
            if let Some(optimize_streaming_latency) = params.optimize_streaming_latency {
                query_params.push(format!("optimize_streaming_latency={}", optimize_streaming_latency));
            }
            if let Some(output_format) = params.output_format {
                query_params.push(format!("output_format={}", output_format));
            }
            
            if !query_params.is_empty() {
                url.push('?');
                url.push_str(&query_params.join("&"));
            }
        }

        let response = self
            .create_request(Method::POST, &url)
            .json(request)
            .send()
            .map_err(|e| internal_error(format!("Failed to start streaming synthesis: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(tts_error_from_status(status));
        }

        Ok(response)
    }

    /// Get available models
    pub fn get_models(&self) -> Result<Vec<Model>, TtsError> {
        trace!("Getting available models");

        let url = format!("{}/v1/models", self.base_url);

        let response = self
            .create_request(Method::GET, &url)
            .send()
            .map_err(|e| internal_error(format!("Failed to get models: {e}")))?;

        parse_response(response)
    }

    /// Get user subscription info
    pub fn get_user_subscription(&self) -> Result<UserSubscription, TtsError> {
        trace!("Getting user subscription info");

        let url = format!("{}/v1/user/subscription", self.base_url);

        let response = self
            .create_request(Method::GET, &url)
            .send()
            .map_err(|e| internal_error(format!("Failed to get user subscription: {e}")))?;

        parse_response(response)
    }

    /// Create a voice clone
    pub fn create_voice(
        &self,
        request: &CreateVoiceRequest,
    ) -> Result<Voice, TtsError> {
        trace!("Creating voice clone: {}", request.name);

        let url = format!("{}/v1/voices/add", self.base_url);

        // For voice creation, we need to send multipart/form-data
        let mut form = reqwest::multipart::Form::new()
            .text("name", request.name.clone())
            .text("description", request.description.clone().unwrap_or_default());

        // Add audio files
        for (i, sample) in request.files.iter().enumerate() {
            let part = reqwest::multipart::Part::bytes(sample.data.clone())
                .file_name(format!("sample_{}.wav", i))
                .mime_str("audio/wav")
                .map_err(|e| internal_error(format!("Failed to create multipart: {e}")))?;
            form = form.part(format!("files[{}]", i), part);
        }

        if let Some(labels) = &request.labels {
            form = form.text("labels", labels.clone());
        }

        let response = self
            .client
            .request(Method::POST, &url)
            .header("xi-api-key", &self.api_key)
            .multipart(form)
            .send()
            .map_err(|e| internal_error(format!("Failed to create voice: {e}")))?;

        parse_response(response)
    }

    /// Delete a voice
    pub fn delete_voice(&self, voice_id: &str) -> Result<(), TtsError> {
        trace!("Deleting voice: {voice_id}");

        let url = format!("{}/v1/voices/{}", self.base_url, voice_id);

        let response = self
            .create_request(Method::DELETE, &url)
            .send()
            .map_err(|e| internal_error(format!("Failed to delete voice: {e}")))?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(tts_error_from_status(response.status()))
        }
    }
}

// Request/Response Types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListVoicesParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort_direction: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_total_count: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListVoicesResponse {
    pub voices: Vec<Voice>,
    pub has_more: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    pub voice_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub category: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settings: Option<VoiceSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samples: Option<Vec<VoiceSample>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high_quality_base_model_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_for_tiers: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_boost: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_speaker_boost: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSample {
    pub sample_id: String,
    pub file_name: String,
    pub mime_type: String,
    pub size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextToSpeechRequest {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_settings: Option<VoiceSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pronunciation_dictionary_locators: Option<Vec<PronunciationDictionaryLocator>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_request_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_request_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apply_text_normalization: Option<String>, // "auto", "on", "off"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apply_language_text_normalization: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_pvc_as_ivc: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationDictionaryLocator {
    pub pronunciation_dictionary_id: String,
    pub version_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextToSpeechParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_logging: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimize_streaming_latency: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub model_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub can_be_finetuned: bool,
    pub can_do_text_to_speech: bool,
    pub can_do_voice_conversion: bool,
    pub can_use_style: bool,
    pub can_use_speaker_boost: bool,
    pub serves_pro_voices: bool,
    pub token_cost_factor: f32,
    pub languages: Vec<LanguageInfo>,
    pub max_characters_request_free_tier: u32,
    pub max_characters_request_subscribed_tier: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    pub language_id: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSubscription {
    pub tier: String,
    pub character_count: u32,
    pub character_limit: u32,
    pub can_extend_character_limit: bool,
    pub allowed_to_extend_character_limit: bool,
    pub next_character_count_reset_unix: u64,
    pub voice_limit: u32,
    pub max_voice_add_edits: u32,
    pub voice_add_edit_counter: u32,
    pub professional_voice_limit: u32,
    pub can_extend_voice_limit: bool,
    pub can_use_instant_voice_cloning: bool,
    pub can_use_professional_voice_cloning: bool,
    pub currency: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct CreateVoiceRequest {
    pub name: String,
    pub description: Option<String>,
    pub files: Vec<AudioFile>,
    pub labels: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AudioFile {
    pub data: Vec<u8>,
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, TtsError> {
    let status = response.status();

    trace!("Received response from ElevenLabs API: {response:?}");

    if status.is_success() {
        let body = response
            .json::<T>()
            .map_err(|err| from_reqwest_error("Failed to decode response body", err))?;

        trace!("Received response from ElevenLabs API: {body:?}");

        Ok(body)
    } else {
        let error_body = response
            .text()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

        trace!("Received {status} response from ElevenLabs API: {error_body:?}");

        Err(tts_error_from_status(status))
    }
}
