use crate::client::{DeepgramTtsApi, Model, get_available_models, RateLimitConfig};
use crate::conversions::{
    deepgram_model_to_voice_info, synthesis_options_to_tts_request, audio_data_to_synthesis_result,
    models_to_language_info, estimate_audio_duration, validate_text_input,
};
use golem_tts::config::with_config_key;
use golem_tts::durability::{DurableTts, ExtendedGuest};
use golem_tts::golem::tts::types::{
    AudioChunk, AudioFormat, LanguageCode, SynthesisResult, TextInput, TimingInfo, TtsError, VoiceGender, VoiceQuality, VoiceSettings,
};
use golem_tts::golem::tts::voices::{
    Guest as VoicesGuest, GuestVoice, GuestVoiceResults, LanguageInfo, Voice, VoiceFilter, VoiceInfo, VoiceResults,
};
use golem_tts::golem::tts::synthesis::{
    Guest as SynthesisGuest, SynthesisOptions, ValidationResult,
};
use golem_tts::golem::tts::streaming::{
    Guest as StreamingGuest, GuestSynthesisStream, GuestVoiceConversionStream, StreamStatus, SynthesisStream, VoiceConversionStream,
};
use golem_tts::golem::tts::advanced::{
    Guest as AdvancedGuest, GuestPronunciationLexicon, GuestLongFormOperation, AudioSample, VoiceDesignParams, 
    PronunciationLexicon, PronunciationEntry, LongFormOperation, LongFormResult, OperationStatus,
};
use golem_rust::wasm_rpc::Pollable;
use std::cell::{Cell, RefCell};

mod client;
mod conversions;

// Deepgram Voice Resource Implementation
struct DeepgramVoiceImpl {
    model_data: Model,
    client: DeepgramTtsApi,
}

impl DeepgramVoiceImpl {
    fn new(model_data: Model, client: DeepgramTtsApi) -> Self {
        Self { model_data, client }
    }
}

impl GuestVoice for DeepgramVoiceImpl {
    fn get_id(&self) -> String {
        self.model_data.voice_id.clone()
    }

    fn get_name(&self) -> String {
        self.model_data.name.clone()
    }

    fn get_provider_id(&self) -> Option<String> {
        Some("Deepgram".to_string())
    }

    fn get_language(&self) -> LanguageCode {
        conversions::normalize_language_code(&self.model_data.language)
    }

    fn get_additional_languages(&self) -> Vec<LanguageCode> {
        // Deepgram voices are typically monolingual
        vec![]
    }

    fn get_gender(&self) -> VoiceGender {
        conversions::parse_gender(&self.model_data.gender)
    }

    fn get_quality(&self) -> VoiceQuality {
        conversions::infer_quality_from_model(&self.model_data.voice_id)
    }

    fn get_description(&self) -> Option<String> {
        Some(format!("{} voice with {} accent, {}. Characteristics: {}. Suitable for: {}",
            self.model_data.gender,
            self.model_data.accent,
            self.model_data.age,
            self.model_data.characteristics.join(", "),
            self.model_data.use_cases.join(", ")
        ))
    }

    fn supports_ssml(&self) -> bool {
        false // Deepgram does not support SSML
    }

    fn get_sample_rates(&self) -> Vec<u32> {
        vec![8000, 16000, 22050, 24000, 32000, 48000]
    }

    fn get_supported_formats(&self) -> Vec<AudioFormat> {
        vec![
            AudioFormat::Mp3,
            AudioFormat::Wav,
            AudioFormat::Pcm,
            AudioFormat::OggOpus,
            AudioFormat::Aac,
            AudioFormat::Flac,
            AudioFormat::Mulaw,
            AudioFormat::Alaw,
        ]
    }

    fn update_settings(&self, _settings: VoiceSettings) -> Result<(), TtsError> {
        // Deepgram doesn't support voice settings updates
        Err(TtsError::UnsupportedOperation("Deepgram does not support voice settings updates".to_string()))
    }

    fn delete(&self) -> Result<(), TtsError> {
        // Deepgram voices cannot be deleted (they are predefined)
        Err(TtsError::UnsupportedOperation("Deepgram voices cannot be deleted".to_string()))
    }

    fn clone(&self) -> Result<Voice, TtsError> {
        // Deepgram doesn't support voice cloning
        Err(TtsError::UnsupportedOperation("Deepgram does not support voice cloning".to_string()))
    }

    fn preview(&self, text: String) -> Result<Vec<u8>, TtsError> {
        // Generate a short preview using the voice with enhanced client
        let (request, params) = synthesis_options_to_tts_request(text, None);
        let mut params = params.unwrap();
        params.model = Some(self.model_data.voice_id.clone());
        
        // Use enhanced client method - we only need the audio data for preview
        self.client.text_to_speech(&request, Some(&params))
    }
}

// Deepgram Voice Results Implementation
struct DeepgramVoiceResults {
    voices: RefCell<Vec<VoiceInfo>>,
    current_index: Cell<usize>,
}

impl DeepgramVoiceResults {
    fn new(voices: Vec<VoiceInfo>) -> Self {
        Self {
            voices: RefCell::new(voices),
            current_index: Cell::new(0),
        }
    }
}

impl GuestVoiceResults for DeepgramVoiceResults {
    fn has_more(&self) -> bool {
        self.current_index.get() < self.voices.borrow().len()
    }

    fn get_next(&self) -> Result<Vec<VoiceInfo>, TtsError> {
        let voices = self.voices.borrow();
        let current = self.current_index.get();
        
        if current >= voices.len() {
            return Ok(vec![]);
        }
        
        // Return all remaining voices (Deepgram has a manageable number)
        let remaining: Vec<VoiceInfo> = voices[current..].to_vec();
        self.current_index.set(voices.len());
        
        Ok(remaining)
    }

    fn get_total_count(&self) -> Option<u32> {
        Some(self.voices.borrow().len() as u32)
    }
}

// Enhanced Synthesis Stream Implementation with better chunking and error handling
#[warn(dead_code)]
struct DeepgramSynthesisStream {
    client: DeepgramTtsApi,
    current_request: RefCell<Option<crate::client::TextToSpeechRequest>>,
    params: RefCell<Option<crate::client::TextToSpeechParams>>,
    response_stream: RefCell<Option<reqwest::Response>>,
    chunk_buffer: RefCell<Vec<u8>>,
    bytes_streamed: Cell<usize>,
    total_chunks_received: Cell<u32>,
    finished: Cell<bool>,
    sequence_number: Cell<u32>,
    stream_started: Cell<bool>,
}

impl DeepgramSynthesisStream {
    fn new(voice_id: String, client: DeepgramTtsApi, options: Option<SynthesisOptions>) -> Self {
        let (request, params) = synthesis_options_to_tts_request(String::new(), options);
        let mut params = params.unwrap();
        params.model = Some(voice_id.clone());
        
        Self {
            client,
            current_request: RefCell::new(Some(request)),
            params: RefCell::new(Some(params)),
            response_stream: RefCell::new(None),
            chunk_buffer: RefCell::new(Vec::new()),
            bytes_streamed: Cell::new(0),
            total_chunks_received: Cell::new(0),
            finished: Cell::new(false),
            sequence_number: Cell::new(0),
            stream_started: Cell::new(false),
        }
    }
    
    /// Get streaming progress information
    #[allow(dead_code)]
    fn get_progress(&self) -> (usize, u32) {
        (self.bytes_streamed.get(), self.total_chunks_received.get())
    }
}

impl GuestSynthesisStream for DeepgramSynthesisStream {
    fn send_text(&self, input: TextInput) -> Result<(), TtsError> {
        if self.finished.get() {
            return Err(TtsError::InvalidConfiguration("Stream already finished".to_string()));
        }

        // Update the request with new text
        if let Some(mut request) = self.current_request.borrow_mut().take() {
            request.text = input.content;
            self.current_request.borrow_mut().replace(request);
        }

        Ok(())
    }

    fn finish(&self) -> Result<(), TtsError> {
        if self.stream_started.get() {
            return Ok(()); // Already started
        }
        
        // Start the streaming request if we have text
        if let Some(request) = self.current_request.borrow().as_ref() {
            if !request.text.is_empty() {
                match self.client.text_to_speech_stream(request, self.params.borrow().as_ref()) {
                    Ok(response) => {
                        *self.response_stream.borrow_mut() = Some(response);
                        self.stream_started.set(true);
                    }
                    Err(e) => {
                        self.finished.set(true);
                        return Err(e);
                    }
                }
            }
        }
        
        Ok(())
    }

    fn receive_chunk(&self) -> Result<Option<AudioChunk>, TtsError> {
        if self.finished.get() {
            return Ok(None);
        }
        
        // Start stream if not started and we have pending audio
        if !self.stream_started.get() && self.has_pending_audio() {
            self.finish()?;
        }
        
        // If we have a response stream, try to read data from it
        if let Some(response) = self.response_stream.borrow_mut().take() {
            // Read response in manageable chunks (similar to ElevenLabs approach)
            const CHUNK_SIZE: usize = 8192; // 8KB chunks for better streaming
            
            // Note: response.bytes() consumes the response, so we read all data at once
            match response.bytes() {
                Ok(bytes) => {
                    if bytes.is_empty() {
                        self.finished.set(true);
                        return Ok(None);
                    }
                    
                    let mut current_buffer = self.chunk_buffer.borrow_mut();
                    current_buffer.extend_from_slice(&bytes);
                    
                    // If we have enough data for a chunk or this is the final data
                    if current_buffer.len() >= CHUNK_SIZE || bytes.len() < CHUNK_SIZE {
                        let chunk_data: Vec<u8> = if current_buffer.len() <= CHUNK_SIZE {
                            current_buffer.drain(..).collect()
                        } else {
                            current_buffer.drain(..CHUNK_SIZE).collect()
                        };
                        
                        let sequence = self.sequence_number.get();
                        self.sequence_number.set(sequence + 1);
                        self.bytes_streamed.set(self.bytes_streamed.get() + chunk_data.len());
                        self.total_chunks_received.set(self.total_chunks_received.get() + 1);
                        
                        let is_final = bytes.len() < CHUNK_SIZE && current_buffer.is_empty();
                        if is_final {
                            self.finished.set(true);
                        }
                        
                        let chunk = AudioChunk {
                            data: chunk_data,
                            sequence_number: sequence,
                            is_final,
                            timing_info: None, // Deepgram doesn't provide timing info in streaming
                        };
                        
                        return Ok(Some(chunk));
                    }
                    
                   
                    Ok(None)
                }
                Err(e) => {
                    self.finished.set(true);
                    Err(TtsError::NetworkError(format!("Failed to read response: {}", e)))
                }
            }
        } else {
            // No more data to process
            if self.stream_started.get() && self.chunk_buffer.borrow().is_empty() {
                self.finished.set(true);
            }
            Ok(None)
        }
    }

    fn has_pending_audio(&self) -> bool {
        !self.finished.get() && (
            self.response_stream.borrow().is_some() || 
            !self.chunk_buffer.borrow().is_empty() ||
            (!self.stream_started.get() && self.current_request.borrow().as_ref().map_or(false, |r| !r.text.is_empty()))
        )
    }

    fn get_status(&self) -> StreamStatus {
        if self.finished.get() {
            StreamStatus::Finished
        } else if self.stream_started.get() || self.response_stream.borrow().is_some() {
            StreamStatus::Processing
        } else {
            StreamStatus::Ready
        }
    }

    fn close(&self) {
        self.finished.set(true);
        self.stream_started.set(false);
        *self.response_stream.borrow_mut() = None;
        self.chunk_buffer.borrow_mut().clear();
    }
}

// Voice Conversion Stream (not supported by Deepgram)
struct DeepgramVoiceConversionStream {
    _voice_id: String,
}

impl DeepgramVoiceConversionStream {
    fn new(voice_id: String, _client: DeepgramTtsApi) -> Self {
        Self { _voice_id: voice_id }
    }
}

impl GuestVoiceConversionStream for DeepgramVoiceConversionStream {
    fn send_audio(&self, _audio_data: Vec<u8>) -> Result<(), TtsError> {
        Err(TtsError::UnsupportedOperation("Deepgram does not support voice conversion".to_string()))
    }

    fn receive_converted(&self) -> Result<Option<AudioChunk>, TtsError> {
        Err(TtsError::UnsupportedOperation("Deepgram does not support voice conversion".to_string()))
    }

    fn finish(&self) -> Result<(), TtsError> {
        Err(TtsError::UnsupportedOperation("Deepgram does not support voice conversion".to_string()))
    }

    fn close(&self) {
        // No-op
    }
}

// Pronunciation Lexicon (not supported by Deepgram)
struct DeepgramPronunciationLexicon {
    _name: String,
}

impl DeepgramPronunciationLexicon {
    fn new(name: String, _language: LanguageCode, _entries: Option<Vec<PronunciationEntry>>) -> Self {
        Self { _name: name }
    }
}

impl GuestPronunciationLexicon for DeepgramPronunciationLexicon {
    fn get_name(&self) -> String {
        self._name.clone()
    }

    fn get_language(&self) -> LanguageCode {
        "en".to_string()
    }

    fn get_entry_count(&self) -> u32 {
        0
    }

    fn add_entry(&self, _word: String, _pronunciation: String) -> Result<(), TtsError> {
        Err(TtsError::UnsupportedOperation("Deepgram does not support pronunciation lexicons".to_string()))
    }

    fn remove_entry(&self, _word: String) -> Result<(), TtsError> {
        Err(TtsError::UnsupportedOperation("Deepgram does not support pronunciation lexicons".to_string()))
    }

    fn export_content(&self) -> Result<String, TtsError> {
        Err(TtsError::UnsupportedOperation("Deepgram does not support pronunciation lexicons".to_string()))
    }
}

// Long Form Operation (basic implementation)
struct DeepgramLongFormOperation {
    content: String,
    voice_id: String,
    client: DeepgramTtsApi,
    status: Cell<OperationStatus>,
    progress: Cell<f32>,
    audio_chunks: RefCell<Option<Vec<Vec<u8>>>>,
}

impl DeepgramLongFormOperation {
    fn new(content: String, _output_location: String, voice_id: String, client: DeepgramTtsApi, _chapter_breaks: Option<Vec<u32>>) -> Self {
        Self {
            content,
            voice_id,
            client,
            status: Cell::new(OperationStatus::Pending),
            progress: Cell::new(0.0),
            audio_chunks: RefCell::new(None),
        }
    }

    fn process_long_form(&self) -> Result<(), TtsError> {
        self.status.set(OperationStatus::Processing);
        
        // Split content into chunks that respect Deepgram's character limits
        let chunks = self.split_content_intelligently(&self.content, 1000); // Aura-2 limit
        let mut audio_chunks = Vec::new();
        
        for (i, chunk) in chunks.iter().enumerate() {
            let (request, mut params) = synthesis_options_to_tts_request(chunk.clone(), None);
            if let Some(ref mut p) = params {
                p.model = Some(self.voice_id.clone());
            }
            
            // Use enhanced client method with metadata
            match self.client.text_to_speech_with_metadata(&request, params.as_ref()) {
                Ok(tts_response) => {
                    audio_chunks.push(tts_response.audio_data);
                    self.progress.set((i + 1) as f32 / chunks.len() as f32);
                }
                Err(e) => {
                    self.status.set(OperationStatus::Failed);
                    return Err(e);
                }
            }
        }
        
        *self.audio_chunks.borrow_mut() = Some(audio_chunks);
        self.status.set(OperationStatus::Completed);
        self.progress.set(1.0);
        
        Ok(())
    }

    fn split_content_intelligently(&self, content: &str, max_chunk_size: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let sentences: Vec<&str> = content.split(". ").collect();
        let mut current_chunk = String::new();
        
        for sentence in sentences {
            let sentence_with_period = if sentence.ends_with('.') {
                sentence.to_string()
            } else {
                format!("{}.", sentence)
            };
            
            if current_chunk.len() + sentence_with_period.len() + 1 <= max_chunk_size {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(&sentence_with_period);
            } else {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                }
                current_chunk = sentence_with_period;
            }
        }
        
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }
        
        chunks
    }
}

impl GuestLongFormOperation for DeepgramLongFormOperation {
    fn get_status(&self) -> OperationStatus {
        self.status.get()
    }

    fn get_progress(&self) -> f32 {
        self.progress.get()
    }

    fn cancel(&self) -> Result<(), TtsError> {
        self.status.set(OperationStatus::Cancelled);
        Ok(())
    }

    fn get_result(&self) -> Result<LongFormResult, TtsError> {
        if self.status.get() != OperationStatus::Completed {
            return Err(TtsError::InvalidConfiguration("Operation not completed".to_string()));
        }
        
        if let Some(chunks) = self.audio_chunks.borrow().as_ref() {
            // Concatenate all audio chunks
            let mut combined_audio = Vec::new();
            for chunk in chunks {
                combined_audio.extend_from_slice(chunk);
            }
            
            let result = LongFormResult {
                output_location: "deepgram-synthesis".to_string(),
                total_duration: estimate_audio_duration(&combined_audio, 24000),
                chapter_durations: None,
                metadata: audio_data_to_synthesis_result(combined_audio.clone(), &self.content, "linear16", 24000).metadata,
            };
            
            Ok(result)
        } else {
            Err(TtsError::InternalError("No audio data available".to_string()))
        }
    }
}

// Main Deepgram Component
// 
// Environment Variables:
// - DEEPGRAM_API_KEY: Required API key for Deepgram service
// - DEEPGRAM_API_VERSION: Optional API version (defaults to "v1")
struct DeepgramComponent;

impl DeepgramComponent {
    const ENV_VAR_NAME: &'static str = "DEEPGRAM_API_KEY";

    fn create_client() -> Result<DeepgramTtsApi, TtsError> {
        with_config_key(Self::ENV_VAR_NAME, Err, |api_key| {
            Ok(DeepgramTtsApi::new(api_key.to_string()))
        })
    }

    /// Create client with custom rate limiting configuration
    fn create_client_with_rate_limit(rate_limit_config: RateLimitConfig) -> Result<DeepgramTtsApi, TtsError> {
        with_config_key(Self::ENV_VAR_NAME, Err, |api_key| {
            Ok(DeepgramTtsApi::new(api_key.to_string()).with_rate_limit_config(rate_limit_config))
        })
    }

    /// Create client optimized for batch operations (more aggressive retries)
    fn create_batch_client() -> Result<DeepgramTtsApi, TtsError> {
        let batch_config = RateLimitConfig {
            max_retries: 5, // More retries for batch operations
            initial_delay: std::time::Duration::from_millis(500),
            max_delay: std::time::Duration::from_secs(60), // Longer max delay
            backoff_multiplier: 1.5, // Gentler backoff for batch
        };
        Self::create_client_with_rate_limit(batch_config)
    }

    /// Create client optimized for streaming (faster recovery)
    fn create_streaming_client() -> Result<DeepgramTtsApi, TtsError> {
        let streaming_config = RateLimitConfig {
            max_retries: 3, // Fewer retries for real-time streaming
            initial_delay: std::time::Duration::from_millis(200),
            max_delay: std::time::Duration::from_secs(5), // Short max delay
            backoff_multiplier: 2.0, // Faster backoff for streaming
        };
        Self::create_client_with_rate_limit(streaming_config)
    }
}

impl VoicesGuest for DeepgramComponent {
    type Voice = DeepgramVoiceImpl;
    type VoiceResults = DeepgramVoiceResults;

    fn list_voices(filter: Option<VoiceFilter>) -> Result<VoiceResults, TtsError> {
        let client = Self::create_client()?;
        let models = get_available_models();
        
        // Use enhanced client filtering if we have filter criteria
        if let Some(f) = filter.as_ref() {
            // Convert VoiceFilter to VoiceFilters for enhanced filtering
            let mut voice_filters = crate::client::VoiceFilters::new();
            
            if let Some(lang) = &f.language {
                voice_filters = voice_filters.with_language(lang.clone());
            }
            
            if let Some(gender) = f.gender {
                let gender_str = match gender {
                    VoiceGender::Male => "masculine",
                    VoiceGender::Female => "feminine", 
                    VoiceGender::Neutral => "neutral",
                };
                voice_filters = voice_filters.with_gender(gender_str.to_string());
            }
            
            if let Some(quality) = f.quality {
                let _quality_filter = match quality {
                    VoiceQuality::Standard => crate::client::VoiceQuality::Standard,
                    VoiceQuality::Premium => crate::client::VoiceQuality::Premium,
                    VoiceQuality::Neural => crate::client::VoiceQuality::Professional,
                    VoiceQuality::Studio => crate::client::VoiceQuality::Professional,
                };
                voice_filters = voice_filters.with_version(crate::client::ModelVersion::Aura2);
            }
            
            if let Some(query) = &f.search_query {
                voice_filters = voice_filters.with_search(query.clone());
            }
            
            // Use enhanced client filtering
            let filtered_response = client.get_models_filtered(&voice_filters)?;
            let voice_infos: Vec<VoiceInfo> = filtered_response.models.into_iter()
                .map(deepgram_model_to_voice_info)
                .collect();
                
            return Ok(VoiceResults::new(DeepgramVoiceResults::new(voice_infos)));
        }
        
        // Fallback to basic filtering for simple cases
        let mut voice_infos: Vec<VoiceInfo> = models.into_iter()
            .map(deepgram_model_to_voice_info)
            .collect();

        // Apply basic filter if provided
        if let Some(f) = filter {
            voice_infos.retain(|voice| {
                let mut matches = true;

                if let Some(gender) = f.gender {
                    matches = matches && voice.gender == gender;
                }

                if let Some(quality) = f.quality {
                    matches = matches && voice.quality == quality;
                }

                if let Some(lang) = &f.language {
                    matches = matches && voice.language == *lang;
                }

                if let Some(provider) = &f.provider {
                    matches = matches && voice.provider.contains(provider);
                }

                if let Some(query) = &f.search_query {
                    let query_lower = query.to_lowercase();
                    matches = matches && (
                        voice.name.to_lowercase().contains(&query_lower) ||
                        voice.description.as_ref().map_or(false, |d| d.to_lowercase().contains(&query_lower)) ||
                        voice.use_cases.iter().any(|uc| uc.to_lowercase().contains(&query_lower))
                    );
                }

                matches
            });
        }

        Ok(VoiceResults::new(DeepgramVoiceResults::new(voice_infos)))
    }

    fn get_voice(voice_id: String) -> Result<Voice, TtsError> {
        let client = Self::create_client()?;
        let models = get_available_models();
        
        if let Some(model) = models.into_iter().find(|m| m.voice_id == voice_id) {
            Ok(Voice::new(DeepgramVoiceImpl::new(model, client)))
        } else {
            Err(TtsError::VoiceNotFound(format!("Voice '{}' not found", voice_id)))
        }
    }

    fn search_voices(query: String, filter: Option<VoiceFilter>) -> Result<Vec<VoiceInfo>, TtsError> {
        let client = Self::create_client()?;
        
        // Use enhanced client search functionality
        let search_results = client.search_models(&query)?;
        let mut voice_infos: Vec<VoiceInfo> = search_results.into_iter()
            .map(deepgram_model_to_voice_info)
            .collect();
        
        // Apply additional filters if provided
        if let Some(f) = filter {
            voice_infos.retain(|voice| {
                let mut matches = true;

                if let Some(gender) = f.gender {
                    matches = matches && voice.gender == gender;
                }

                if let Some(quality) = f.quality {
                    matches = matches && voice.quality == quality;
                }

                if let Some(lang) = &f.language {
                    matches = matches && voice.language == *lang;
                }

                if let Some(provider) = &f.provider {
                    matches = matches && voice.provider.contains(provider);
                }

                matches
            });
        }
        
        Ok(voice_infos)
    }

    fn list_languages() -> Result<Vec<LanguageInfo>, TtsError> {
        let models = get_available_models();
        Ok(models_to_language_info(models))
    }
}

impl SynthesisGuest for DeepgramComponent {
    fn synthesize(
        input: TextInput,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<SynthesisResult, TtsError> {
        let client = Self::create_client()?;
        let voice_id = voice.get::<DeepgramVoiceImpl>().get_id();
        
        let (request, mut params) = synthesis_options_to_tts_request(input.content.clone(), options);
        if let Some(ref mut p) = params {
            p.model = Some(voice_id);
        }
        
        // Use enhanced client method with metadata
        let tts_response = client.text_to_speech_with_metadata(&request, params.as_ref())?;
        let encoding = params.as_ref().and_then(|p| p.encoding.as_ref()).unwrap_or(&"linear16".to_string()).clone();
        let sample_rate = params.as_ref().and_then(|p| p.sample_rate).unwrap_or(24000);
        
        // Create synthesis result with enhanced metadata
        let mut synthesis_result = audio_data_to_synthesis_result(tts_response.audio_data, &input.content, &encoding, sample_rate);
        
        // Enhance metadata with information from TTS response
        synthesis_result.metadata.provider_info = Some(format!("Deepgram TTS - Model: {}, Characters: {}", 
            tts_response.metadata.dg_model_name,
            tts_response.metadata.dg_char_count));
        
        Ok(synthesis_result)
    }

    fn synthesize_batch(
        inputs: Vec<TextInput>,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<Vec<SynthesisResult>, TtsError> {
        let mut results = Vec::new();
        // Use batch-optimized client for better rate limiting
        let client = Self::create_batch_client()?;
        let voice_id = voice.get::<DeepgramVoiceImpl>().get_id();
        
        for input in inputs {
            let (request, mut params) = synthesis_options_to_tts_request(input.content.clone(), options.clone());
            if let Some(ref mut p) = params {
                p.model = Some(voice_id.clone());
            }
            
            // Use enhanced client method with metadata for each request
            match client.text_to_speech_with_metadata(&request, params.as_ref()) {
                Ok(tts_response) => {
                    let encoding = params.as_ref().and_then(|p| p.encoding.as_ref()).unwrap_or(&"linear16".to_string()).clone();
                    let sample_rate = params.as_ref().and_then(|p| p.sample_rate).unwrap_or(24000);
                    
                    let mut synthesis_result = audio_data_to_synthesis_result(tts_response.audio_data, &input.content, &encoding, sample_rate);
                    
                    // Enhance metadata with information from TTS response
                    synthesis_result.metadata.provider_info = Some(format!("Deepgram TTS - Model: {}, Characters: {}", 
                        tts_response.metadata.dg_model_name,
                        tts_response.metadata.dg_char_count));
                    
                    results.push(synthesis_result);
                }
                Err(e) => {
                    // For batch processing, we could continue with other items, but for now fail fast
                    return Err(e);
                }
            }
        }
        
        Ok(results)
    }

    fn get_timing_marks(
        _input: TextInput,
        _voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
    ) -> Result<Vec<TimingInfo>, TtsError> {
        // Deepgram doesn't provide timing marks
        Err(TtsError::UnsupportedOperation("Timing marks not supported by Deepgram".to_string()))
    }

    fn validate_input(
        input: TextInput,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
    ) -> Result<ValidationResult, TtsError> {
        let voice_id = voice.get::<DeepgramVoiceImpl>().get_id();
        
        // Enhanced validation with Deepgram-specific rules
        let mut _is_valid = true;
        let mut messages = Vec::new();
        
        // Basic text validation
        if input.content.is_empty() {
            _is_valid = false;
            messages.push("Text input cannot be empty".to_string());
        }
        
        // Deepgram character limits (vary by model version)
        let char_limit = if voice_id.starts_with("aura-2") { 2000 } else { 1000 };
        if input.content.len() > char_limit {
            _is_valid = false;
            messages.push(format!("Text exceeds {} character limit for {}", char_limit, voice_id));
        }
        
        // Check for unsupported characters (basic validation)
        if input.content.chars().any(|c| c.is_control() && c != '\n' && c != '\r' && c != '\t') {
            messages.push("Warning: Text contains control characters that may not be processed correctly".to_string());
        }
        
        let _message = if messages.is_empty() {
            None
        } else {
            Some(messages.join("; "))
        };
        
        Ok(validate_text_input(&input.content, Some(&voice_id)))
    }
}

impl StreamingGuest for DeepgramComponent {
    type SynthesisStream = DeepgramSynthesisStream;
    type VoiceConversionStream = DeepgramVoiceConversionStream;

    fn create_stream(
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<SynthesisStream, TtsError> {
        // Use streaming-optimized client for better real-time performance
        let client = Self::create_streaming_client()?;
        let voice_id = voice.get::<DeepgramVoiceImpl>().get_id();
        
        let stream = DeepgramSynthesisStream::new(voice_id, client, options);
        Ok(SynthesisStream::new(stream))
    }

    fn create_voice_conversion_stream(
        target_voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Result<VoiceConversionStream, TtsError> {
        let client = Self::create_client()?;
        let voice_id = target_voice.get::<DeepgramVoiceImpl>().get_id();
        
        let stream = DeepgramVoiceConversionStream::new(voice_id, client);
        Ok(VoiceConversionStream::new(stream))
    }
}

impl AdvancedGuest for DeepgramComponent {
    type PronunciationLexicon = DeepgramPronunciationLexicon;
    type LongFormOperation = DeepgramLongFormOperation;

    fn create_voice_clone(
        _name: String,
        _audio_samples: Vec<AudioSample>,
        _description: Option<String>,
    ) -> Result<Voice, TtsError> {
        Err(TtsError::UnsupportedOperation("Deepgram does not support voice cloning".to_string()))
    }

    fn design_voice(
        _name: String,
        _characteristics: VoiceDesignParams,
    ) -> Result<Voice, TtsError> {
        Err(TtsError::UnsupportedOperation("Deepgram does not support voice design".to_string()))
    }

    fn convert_voice(
        _input_audio: Vec<u8>,
        _target_voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        _preserve_timing: Option<bool>,
    ) -> Result<Vec<u8>, TtsError> {
        Err(TtsError::UnsupportedOperation("Deepgram does not support voice conversion".to_string()))
    }

    fn generate_sound_effect(
        _description: String,
        _duration_seconds: Option<f32>,
        _style_influence: Option<f32>,
    ) -> Result<Vec<u8>, TtsError> {
        Err(TtsError::UnsupportedOperation("Deepgram does not support sound effect generation".to_string()))
    }

    fn create_lexicon(
        name: String,
        language: LanguageCode,
        entries: Option<Vec<PronunciationEntry>>,
    ) -> Result<PronunciationLexicon, TtsError> {
        let lexicon = DeepgramPronunciationLexicon::new(name, language, entries);
        Ok(PronunciationLexicon::new(lexicon))
    }

    fn synthesize_long_form(
        content: String,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        output_location: String,
        chapter_breaks: Option<Vec<u32>>,
    ) -> Result<LongFormOperation, TtsError> {
        // Use batch-optimized client for long-form synthesis
        let client = Self::create_batch_client()?;
        let voice_id = voice.get::<DeepgramVoiceImpl>().get_id();
        
        let operation = DeepgramLongFormOperation::new(content, output_location, voice_id, client, chapter_breaks);
        
        // Start processing immediately
        if let Err(e) = operation.process_long_form() {
            return Err(e);
        }
        
        Ok(LongFormOperation::new(operation))
    }
}

impl ExtendedGuest for DeepgramComponent {
    fn unwrapped_synthesis_stream(
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Self::SynthesisStream {
        let client = Self::create_streaming_client().unwrap_or_else(|_| {
            // Fallback to default client for unwrapped method
            DeepgramTtsApi::new("dummy".to_string())
        });
        let voice_id = voice.get::<DeepgramVoiceImpl>().get_id();
        
        DeepgramSynthesisStream::new(voice_id, client, options)
    }

    fn unwrapped_voice_conversion_stream(
        target_voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Self::VoiceConversionStream {
        let client = Self::create_client().unwrap_or_else(|_| {
            // Fallback client for unwrapped method
            DeepgramTtsApi::new("dummy".to_string())
        });
        let voice_id = target_voice.get::<DeepgramVoiceImpl>().get_id();
        
        DeepgramVoiceConversionStream::new(voice_id, client)
    }

    fn subscribe_synthesis_stream(_stream: &Self::SynthesisStream) -> Pollable {
        golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
    }

    fn subscribe_voice_conversion_stream(_stream: &Self::VoiceConversionStream) -> Pollable {
        golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
    }
}

type DurableDeepgramComponent = DurableTts<DeepgramComponent>;

golem_tts::export_tts!(DurableDeepgramComponent with_types_in golem_tts);