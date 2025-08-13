use crate::client::{ElevenLabsTtsApi, Voice as ElevenLabsVoice};
use crate::conversions::{
    audio_data_to_synthesis_result, create_validation_result, create_voice_request_from_samples,
    elevenlabs_voice_to_voice_info, estimate_audio_duration, models_to_language_info,
    synthesis_options_to_tts_request, voice_design_params_to_create_request,
    voice_filter_to_list_params,
};
use golem_rust::wasm_rpc::Pollable;
use golem_tts::config::with_config_key;
use golem_tts::durability::{DurableTts, ExtendedGuest};
use golem_tts::golem::tts::advanced::{
    AudioSample, Guest as AdvancedGuest, GuestLongFormOperation, GuestPronunciationLexicon,
    LongFormOperation, LongFormResult, OperationStatus, PronunciationEntry, PronunciationLexicon,
    VoiceDesignParams,
};
use golem_tts::golem::tts::streaming::{
    Guest as StreamingGuest, GuestSynthesisStream, GuestVoiceConversionStream, StreamStatus,
    SynthesisStream, VoiceConversionStream,
};
use golem_tts::golem::tts::synthesis::{
    Guest as SynthesisGuest, SynthesisOptions, ValidationResult,
};
use golem_tts::golem::tts::types::{
    AudioChunk, AudioFormat, LanguageCode, SynthesisResult, TextInput, TimingInfo, TtsError,
    VoiceGender, VoiceQuality, VoiceSettings,
};
use golem_tts::golem::tts::voices::{
    Guest as VoicesGuest, GuestVoice, GuestVoiceResults, LanguageInfo, Voice, VoiceFilter,
    VoiceInfo, VoiceResults,
};
use log::trace;
use std::cell::{Cell, RefCell};

mod client;
mod conversions;

// ElevenLabs Voice Resource Implementation
struct ElevenLabsVoiceImpl {
    voice_data: ElevenLabsVoice,
    client: ElevenLabsTtsApi,
}

impl ElevenLabsVoiceImpl {
    fn new(voice_data: ElevenLabsVoice, client: ElevenLabsTtsApi) -> Self {
        Self { voice_data, client }
    }
}

impl GuestVoice for ElevenLabsVoiceImpl {
    fn get_id(&self) -> String {
        self.voice_data.voice_id.clone()
    }

    fn get_name(&self) -> String {
        self.voice_data.name.clone()
    }

    fn get_provider_id(&self) -> Option<String> {
        Some("elevenlabs".to_string())
    }

    fn get_language(&self) -> LanguageCode {
        "en-US".to_string() // ElevenLabs default
    }

    fn get_additional_languages(&self) -> Vec<LanguageCode> {
        vec![] // ElevenLabs doesn't provide explicit language info per voice
    }

    fn get_gender(&self) -> VoiceGender {
        conversions::infer_gender_from_name(&self.voice_data.name).unwrap_or(VoiceGender::Neutral)
    }

    fn get_quality(&self) -> VoiceQuality {
        conversions::infer_quality_from_category(&self.voice_data.category)
            .unwrap_or(VoiceQuality::Standard)
    }

    fn get_description(&self) -> Option<String> {
        self.voice_data.description.clone()
    }

    fn supports_ssml(&self) -> bool {
        true // ElevenLabs supports SSML
    }

    fn get_sample_rates(&self) -> Vec<u32> {
        vec![22050, 44100] // Common ElevenLabs sample rates
    }

    fn get_supported_formats(&self) -> Vec<AudioFormat> {
        vec![AudioFormat::Mp3, AudioFormat::Wav, AudioFormat::Pcm]
    }

    fn update_settings(&self, _settings: VoiceSettings) -> Result<(), TtsError> {
        // ElevenLabs doesn't support updating voice settings directly
        Err(TtsError::UnsupportedOperation(
            "Voice settings update not supported by ElevenLabs".to_string(),
        ))
    }

    fn delete(&self) -> Result<(), TtsError> {
        match self.client.delete_voice(&self.voice_data.voice_id) {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn clone(&self) -> Result<Voice, TtsError> {
        // ElevenLabs doesn't have a direct clone API, so we simulate it
        Err(TtsError::UnsupportedOperation(
            "Voice cloning not directly supported by ElevenLabs API".to_string(),
        ))
    }

    fn preview(&self, text: String) -> Result<Vec<u8>, TtsError> {
        let params = crate::client::TextToSpeechParams {
            enable_logging: Some(false),
            optimize_streaming_latency: Some(0),
            output_format: Some("mp3_22050_32".to_string()),
        };

        // Check if model supports language_code parameter
        let model_version = self.client.get_model_version();
        let supports_language_code = !model_version.contains("multilingual");

        let request = crate::client::TextToSpeechRequest {
            text,
            model_id: Some(model_version.to_string()),
            language_code: if supports_language_code { Some("en".to_string()) } else { None }, // Only set for models that support it
            voice_settings: self.voice_data.settings.clone(),
            pronunciation_dictionary_locators: None,
            seed: None,
            previous_text: None,
            next_text: None,
            previous_request_ids: None,
            next_request_ids: None,
            apply_text_normalization: Some("auto".to_string()),
            apply_language_text_normalization: Some(false), // Disable to avoid compatibility issues
            use_pvc_as_ivc: Some(false),
        };

        self.client
            .text_to_speech(&self.voice_data.voice_id, &request, Some(params))
    }
}

// ElevenLabs Voice Results Implementation
struct ElevenLabsVoiceResults {
    voices: RefCell<Vec<VoiceInfo>>,
    current_index: Cell<usize>,
    has_more: Cell<bool>,
    total_count: Option<u32>,
}

impl ElevenLabsVoiceResults {
    fn new(voices: Vec<VoiceInfo>, total_count: Option<u32>) -> Self {
        let has_more = !voices.is_empty();
        Self {
            voices: RefCell::new(voices),
            current_index: Cell::new(0),
            has_more: Cell::new(has_more),
            total_count,
        }
    }
}

impl GuestVoiceResults for ElevenLabsVoiceResults {
    fn has_more(&self) -> bool {
        self.has_more.get()
    }

    fn get_next(&self) -> Result<Vec<VoiceInfo>, TtsError> {
        let voices = self.voices.borrow();
        let current_idx = self.current_index.get();

        if current_idx >= voices.len() {
            self.has_more.set(false);
            return Ok(vec![]);
        }

        // Return next batch of voices (simulating pagination)
        const BATCH_SIZE: usize = 10;
        let end_idx = std::cmp::min(current_idx + BATCH_SIZE, voices.len());
        let batch = voices[current_idx..end_idx].to_vec();

        self.current_index.set(end_idx);
        self.has_more.set(end_idx < voices.len());

        Ok(batch)
    }

    fn get_total_count(&self) -> Option<u32> {
        self.total_count
    }
}

// Synthesis Stream Implementation with native ElevenLabs streaming
struct ElevenLabsSynthesisStream {
    voice_id: String,
    client: ElevenLabsTtsApi,
    current_request: RefCell<Option<crate::client::TextToSpeechRequest>>,
    params: RefCell<Option<crate::client::TextToSpeechParams>>,
    response_stream: RefCell<Option<reqwest::Response>>,
    chunk_buffer: RefCell<Vec<u8>>,
    bytes_streamed: Cell<usize>,
    total_chunks_received: Cell<u32>,
    finished: Cell<bool>,
    sequence_number: Cell<u32>,
}

impl ElevenLabsSynthesisStream {
    fn new(voice_id: String, client: ElevenLabsTtsApi, options: Option<SynthesisOptions>) -> Self {
        let (request, params) =
            conversions::synthesis_options_to_tts_request(options, client.get_model_version());

        Self {
            voice_id,
            client,
            current_request: RefCell::new(Some(request)),
            params: RefCell::new(params),
            response_stream: RefCell::new(None),
            chunk_buffer: RefCell::new(Vec::new()),
            bytes_streamed: Cell::new(0),
            total_chunks_received: Cell::new(0),
            finished: Cell::new(false),
            sequence_number: Cell::new(0),
        }
    }

    /// Get streaming progress information
    #[allow(dead_code)]
    fn get_progress(&self) -> (usize, u32) {
        (self.bytes_streamed.get(), self.total_chunks_received.get())
    }
}

impl GuestSynthesisStream for ElevenLabsSynthesisStream {
    fn send_text(&self, input: TextInput) -> Result<(), TtsError> {
        if self.finished.get() {
            return Err(TtsError::InternalError("Stream is finished".to_string()));
        }

        // Update the request with new text
        {
            let mut request_ref = self.current_request.borrow_mut();
            if let Some(mut request) = request_ref.take() {
                println!("[DEBUG] ElevenLabs send_text - Updating request with new text: '{}'", input.content);
                request.text = input.content;
                *request_ref = Some(request);
                println!("[DEBUG] ElevenLabs send_text - Successfully updated request");
            } else {
                println!("[DEBUG] ElevenLabs send_text - Warning: No current request found to update");
            }
        }

        Ok(())
    }

    fn finish(&self) -> Result<(), TtsError> {
        // Start the streaming request if we have text
        if let Some(request) = self.current_request.borrow().as_ref() {
            if !request.text.is_empty() {
                match self.client.text_to_speech_stream(
                    &self.voice_id,
                    request,
                    self.params.borrow().clone(),
                ) {
                    Ok(response) => {
                        self.response_stream.borrow_mut().replace(response);
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

        // If we have a response stream, try to read data from it
        if let Some(response) = self.response_stream.borrow_mut().take() {
            // In a synchronous context, we read the entire response and simulate chunking
            // In a real async environment, this would use proper streaming APIs
            match response.bytes() {
                Ok(bytes) => {
                    if !bytes.is_empty() {
                        // Simulate chunking by breaking the response into smaller pieces
                        const CHUNK_SIZE: usize = 4096; // 4KB chunks for realistic streaming feel
                        let mut current_buffer = self.chunk_buffer.borrow_mut();

                        // If this is the first call, store all the data
                        if current_buffer.is_empty() {
                            current_buffer.extend_from_slice(&bytes);
                        }

                        // Check if we have enough data for a chunk
                        if current_buffer.len() >= CHUNK_SIZE {
                            // Take a chunk from the buffer
                            let chunk_data: Vec<u8> = current_buffer.drain(0..CHUNK_SIZE).collect();

                            // Update streaming statistics
                            let current_bytes = self.bytes_streamed.get();
                            let current_chunks = self.total_chunks_received.get();
                            self.bytes_streamed.set(current_bytes + chunk_data.len());
                            self.total_chunks_received.set(current_chunks + 1);

                            let seq = self.sequence_number.get();
                            self.sequence_number.set(seq + 1);

                            // Determine if this is the final chunk
                            let is_final = current_buffer.is_empty();

                            // Create audio chunk with incremental data
                            let chunk = AudioChunk {
                                data: chunk_data.clone(),
                                sequence_number: seq,
                                is_final,
                                timing_info: Some(TimingInfo {
                                    start_time_seconds: (current_bytes as f32) / 12000.0, // Rough timing calculation
                                    end_time_seconds: Some(estimate_audio_duration(
                                        &chunk_data,
                                        22050,
                                    )),
                                    text_offset: None,
                                    mark_type: None,
                                }),
                            };

                            // If not final, put the response back for next chunk
                            if !is_final {
                                // In real implementation, we'd keep the response stream alive
                                // For now, we'll continue processing from buffer
                            } else {
                                self.finished.set(true);
                            }

                            return Ok(Some(chunk));
                        } else if !current_buffer.is_empty() {
                            // Send remaining data as final chunk
                            let final_data = current_buffer.clone();
                            current_buffer.clear();

                            let current_bytes = self.bytes_streamed.get();
                            let current_chunks = self.total_chunks_received.get();
                            self.bytes_streamed.set(current_bytes + final_data.len());
                            self.total_chunks_received.set(current_chunks + 1);

                            let seq = self.sequence_number.get();
                            let final_chunk = AudioChunk {
                                data: final_data.clone(),
                                sequence_number: seq,
                                is_final: true,
                                timing_info: Some(TimingInfo {
                                    start_time_seconds: (current_bytes as f32) / 12000.0,
                                    end_time_seconds: Some(estimate_audio_duration(
                                        &final_data,
                                        22050,
                                    )),
                                    text_offset: None,
                                    mark_type: None,
                                }),
                            };

                            self.finished.set(true);
                            return Ok(Some(final_chunk));
                        }
                    }

                    // No data received, mark as finished
                    self.finished.set(true);
                    Ok(None)
                }
                Err(e) => {
                    self.finished.set(true);
                    Err(TtsError::NetworkError(format!("Stream read error: {}", e)))
                }
            }
        } else {
            self.finished.set(true);
            Ok(None)
        }
    }

    fn has_pending_audio(&self) -> bool {
        !self.finished.get()
            && (self.response_stream.borrow().is_some()
                || self
                    .current_request
                    .borrow()
                    .as_ref()
                    .is_some_and(|r| !r.text.is_empty()))
    }

    fn get_status(&self) -> StreamStatus {
        if self.finished.get() {
            StreamStatus::Finished
        } else if self.response_stream.borrow().is_some() {
            StreamStatus::Processing
        } else {
            StreamStatus::Ready
        }
    }

    fn close(&self) {
        self.finished.set(true);
        self.response_stream.borrow_mut().take();
    }
}

// Voice Conversion Stream with ElevenLabs speech-to-speech
struct ElevenLabsVoiceConversionStream {
    voice_id: String,
    client: ElevenLabsTtsApi,
    audio_buffer: RefCell<Vec<u8>>,
    finished: Cell<bool>,
    sequence_number: Cell<u32>,
}

impl ElevenLabsVoiceConversionStream {
    fn new(voice_id: String, client: ElevenLabsTtsApi) -> Self {
        Self {
            voice_id,
            client,
            audio_buffer: RefCell::new(Vec::new()),
            finished: Cell::new(false),
            sequence_number: Cell::new(0),
        }
    }
}

impl GuestVoiceConversionStream for ElevenLabsVoiceConversionStream {
    fn send_audio(&self, audio_data: Vec<u8>) -> Result<(), TtsError> {
        if self.finished.get() {
            return Err(TtsError::InternalError("Stream is finished".to_string()));
        }

        // Accumulate audio data for processing
        self.audio_buffer
            .borrow_mut()
            .extend_from_slice(&audio_data);
        Ok(())
    }

    fn receive_converted(&self) -> Result<Option<AudioChunk>, TtsError> {
        if self.finished.get() {
            return Ok(None);
        }

        // Process accumulated audio data using speech-to-speech
        let audio_data = self.audio_buffer.borrow().clone();
        if !audio_data.is_empty() {
            let request = crate::client::SpeechToSpeechRequest {
                audio_data,
                model_id: Some("eleven_english_sts_v2".to_string()),
                voice_settings: None,
                seed: None,
            };

            match self.client.speech_to_speech(&self.voice_id, &request, None) {
                Ok(converted_audio) => {
                    let seq = self.sequence_number.get();
                    self.sequence_number.set(seq + 1);

                    let chunk = AudioChunk {
                        data: converted_audio,
                        sequence_number: seq,
                        is_final: true,
                        timing_info: None,
                    };

                    // Clear buffer after processing
                    self.audio_buffer.borrow_mut().clear();
                    self.finished.set(true);

                    return Ok(Some(chunk));
                }
                Err(e) => {
                    self.finished.set(true);
                    return Err(e);
                }
            }
        }

        Ok(None)
    }

    fn finish(&self) -> Result<(), TtsError> {
        self.finished.set(true);
        Ok(())
    }

    fn close(&self) {
        self.finished.set(true);
        self.audio_buffer.borrow_mut().clear();
    }
}

// Pronunciation Lexicon Implementation (placeholder)
struct ElevenLabsPronunciationLexicon {
    name: String,
    language: LanguageCode,
    entries: RefCell<Vec<PronunciationEntry>>,
}

impl ElevenLabsPronunciationLexicon {
    fn new(name: String, language: LanguageCode, entries: Option<Vec<PronunciationEntry>>) -> Self {
        Self {
            name,
            language,
            entries: RefCell::new(entries.unwrap_or_default()),
        }
    }
}

impl GuestPronunciationLexicon for ElevenLabsPronunciationLexicon {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn get_language(&self) -> LanguageCode {
        self.language.clone()
    }

    fn get_entry_count(&self) -> u32 {
        self.entries.borrow().len() as u32
    }

    fn add_entry(&self, word: String, pronunciation: String) -> Result<(), TtsError> {
        self.entries.borrow_mut().push(PronunciationEntry {
            word,
            pronunciation,
            part_of_speech: None,
        });
        Ok(())
    }

    fn remove_entry(&self, word: String) -> Result<(), TtsError> {
        self.entries.borrow_mut().retain(|entry| entry.word != word);
        Ok(())
    }

    fn export_content(&self) -> Result<String, TtsError> {
        let entries = self.entries.borrow();
        let mut content = format!(
            "# Pronunciation Lexicon: {}\n# Language: {}\n\n",
            self.name, self.language
        );

        for entry in entries.iter() {
            content.push_str(&format!("{}: {}\n", entry.word, entry.pronunciation));
        }

        Ok(content)
    }
}

// Long Form Operation Implementation with ElevenLabs batch processing
struct ElevenLabsLongFormOperation {
    content: String,
    output_location: String,
    voice_id: String,
    client: ElevenLabsTtsApi,
    status: Cell<OperationStatus>,
    progress: Cell<f32>,
    audio_chunks: RefCell<Option<Vec<Vec<u8>>>>,
}

impl ElevenLabsLongFormOperation {
    fn new(
        content: String,
        output_location: String,
        voice_id: String,
        client: ElevenLabsTtsApi,
        _chapter_breaks: Option<Vec<u32>>,
    ) -> Self {
        Self {
            content,
            output_location,
            voice_id,
            client,
            status: Cell::new(OperationStatus::Processing),
            progress: Cell::new(0.0),
            audio_chunks: RefCell::new(None),
        }
    }

    fn process_long_form(&self) -> Result<(), TtsError> {
        // Use the batch processing functionality
        let max_chunk_size = 4500; // Conservative limit for ElevenLabs
        let chunks = self.client.synthesize_long_form_batch(
            &self.voice_id,
            &self.content,
            None, // Use default synthesis options
            max_chunk_size,
        )?;

        self.audio_chunks.borrow_mut().replace(chunks);
        self.status.set(OperationStatus::Completed);
        self.progress.set(1.0);
        Ok(())
    }
}

impl GuestLongFormOperation for ElevenLabsLongFormOperation {
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
            // Try to process if not completed yet
            if self.status.get() == OperationStatus::Processing {
                self.process_long_form()?;
            } else {
                return Err(TtsError::InternalError(
                    "Operation not completed".to_string(),
                ));
            }
        }

        let audio_chunks = self.audio_chunks.borrow();
        let chunks = audio_chunks
            .as_ref()
            .ok_or_else(|| TtsError::InternalError("Audio chunks not available".to_string()))?;

        // Calculate total audio size
        let total_audio_size: usize = chunks.iter().map(|chunk| chunk.len()).sum();

        // Estimate duration (assuming MP3 at ~128kbps, approximately 16KB per second)
        let estimated_duration = (total_audio_size as f64) / 16000.0;

        Ok(LongFormResult {
            output_location: self.output_location.clone(),
            total_duration: estimated_duration as f32,
            chapter_durations: None,
            metadata: golem_tts::golem::tts::types::SynthesisMetadata {
                duration_seconds: estimated_duration as f32,
                character_count: self.content.len() as u32,
                word_count: self.content.split_whitespace().count() as u32,
                audio_size_bytes: total_audio_size as u32,
                request_id: format!("elevenlabs-long-form-{}", self.voice_id),
                provider_info: Some("elevenlabs".to_string()),
            },
        })
    }
}

// Main ElevenLabs Component
struct ElevenLabsComponent;

impl ElevenLabsComponent {
    const ENV_VAR_NAME: &'static str = "ELEVENLABS_API_KEY";
    const MODEL_VERSION_ENV_VAR: &'static str = "ELEVENLABS_MODEL_VERSION";

    fn create_client() -> Result<ElevenLabsTtsApi, TtsError> {
        with_config_key(Self::ENV_VAR_NAME, Err, |api_key| {
            let model_version = std::env::var(Self::MODEL_VERSION_ENV_VAR)
                .unwrap_or_else(|_| "eleven_multilingual_v2".to_string());
            Ok(ElevenLabsTtsApi::new(api_key.to_string(), model_version))
        })
    }

    /// Validate synthesis input and options for proper error handling
    fn validate_synthesis_input(
        input: &TextInput,
        options: Option<&SynthesisOptions>,
    ) -> Result<(), TtsError> {
        // Validate empty text
        if input.content.trim().is_empty() {
            return Err(TtsError::InvalidText("Text content cannot be empty".to_string()));
        }

        // Validate text length (ElevenLabs has limits)
        if input.content.len() > 5000 {
            return Err(TtsError::InvalidText("Text exceeds maximum length of 5000 characters".to_string()));
        }

        // Validate SSML content if specified
        if input.text_type == golem_tts::golem::tts::types::TextType::Ssml {
            if let Err(msg) = Self::validate_ssml_content(&input.content) {
                return Err(TtsError::InvalidSsml(msg));
            }
        }

        // Validate language code if specified
        if let Some(ref language) = input.language {
            if !Self::is_supported_language(language) {
                return Err(TtsError::UnsupportedLanguage(
                    format!("Language '{}' is not supported by ElevenLabs", language)
                ));
            }
        }

        // Validate voice settings if specified
        if let Some(opts) = options {
            if let Some(ref voice_settings) = opts.voice_settings {
                if let Err(msg) = Self::validate_voice_settings(voice_settings) {
                    return Err(TtsError::InvalidConfiguration(msg));
                }
            }
        }

        Ok(())
    }

    /// Validate SSML content for basic structure
    fn validate_ssml_content(content: &str) -> Result<(), String> {
        // Basic SSML validation - check for unmatched tags
        let mut tag_stack = Vec::new();
        let mut chars = content.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '<' {
                // Parse tag
                let mut tag = String::new();
                let mut is_closing = false;
                let mut is_self_closing = false;
                
                // Check if it's a closing tag
                if chars.peek() == Some(&'/') {
                    is_closing = true;
                    chars.next(); // consume '/'
                }

                // Read tag name and attributes
                let mut full_tag_content = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch == '>' {
                        break;
                    }
                    if ch == ' ' && tag.is_empty() {
                        // We've read the tag name, now read the rest
                        tag = full_tag_content.clone();
                    }
                    full_tag_content.push(chars.next().unwrap());
                }

                // If we didn't hit a space, the entire content is the tag name
                if tag.is_empty() {
                    tag = full_tag_content.clone();
                }

                // Check if it's self-closing (ends with '/')
                if full_tag_content.ends_with('/') {
                    is_self_closing = true;
                    // Remove the trailing '/' from tag name if it got included
                    if tag.ends_with('/') {
                        tag = tag[..tag.len()-1].to_string();
                    }
                }

                // Skip to end of tag
                while let Some(ch) = chars.next() {
                    if ch == '>' {
                        break;
                    }
                }

                if is_closing {
                    if let Some(expected_tag) = tag_stack.pop() {
                        if expected_tag != tag {
                            return Err(format!("Unmatched closing tag: </{}>", tag));
                        }
                    } else {
                        return Err(format!("Unmatched closing tag: </{}>", tag));
                    }
                } else if !tag.is_empty() && !tag.starts_with('!') && !tag.starts_with('?') {
                    // Only track opening tags that aren't self-closing, XML declarations, or comments
                    if !is_self_closing {
                        tag_stack.push(tag);
                    }
                }
            }
        }

        if !tag_stack.is_empty() {
            return Err(format!("Unclosed tags: {:?}", tag_stack));
        }

        Ok(())
    }

    /// Check if a language is supported by ElevenLabs
    fn is_supported_language(language: &str) -> bool {
        let supported_languages = [
            "en", "en-US", "en-GB", "en-AU", "en-CA",
            "es", "es-ES", "es-MX", "es-AR",
            "fr", "fr-FR", "fr-CA",
            "de", "de-DE", "de-AT", "de-CH",
            "it", "it-IT",
            "pt", "pt-PT", "pt-BR",
            "pl", "pl-PL",
            "zh", "zh-CN", "zh-TW",
            "ja", "ja-JP",
            "hi", "hi-IN",
            "ko", "ko-KR",
            "nl", "nl-NL",
            "tr", "tr-TR",
            "sv", "sv-SE",
            "da", "da-DK",
            "no", "no-NO",
            "fi", "fi-FI",
        ];

        supported_languages.contains(&language)
    }

    /// Validate voice settings parameters
    fn validate_voice_settings(settings: &golem_tts::golem::tts::types::VoiceSettings) -> Result<(), String> {
        // Check speed (should be between 0.25 and 4.0 for most TTS systems)
        if let Some(speed) = settings.speed {
            if speed < 0.1 || speed > 5.0 {
                return Err(format!("Speed value {} is out of valid range (0.1-5.0)", speed));
            }
        }

        // Check pitch (should be reasonable range)
        if let Some(pitch) = settings.pitch {
            if pitch < -50.0 || pitch > 50.0 {
                return Err(format!("Pitch value {} is out of valid range (-50.0 to 50.0)", pitch));
            }
        }

        // Check stability (ElevenLabs specific: 0.0-1.0)
        if let Some(stability) = settings.stability {
            if stability < 0.0 || stability > 1.0 {
                return Err(format!("Stability value {} is out of valid range (0.0-1.0)", stability));
            }
        }

        // Check similarity (ElevenLabs specific: 0.0-1.0)
        if let Some(similarity) = settings.similarity {
            if similarity < 0.0 || similarity > 1.0 {
                return Err(format!("Similarity value {} is out of valid range (0.0-1.0)", similarity));
            }
        }

        Ok(())
    }
}

impl VoicesGuest for ElevenLabsComponent {
    type Voice = ElevenLabsVoiceImpl;
    type VoiceResults = ElevenLabsVoiceResults;

    fn list_voices(filter: Option<VoiceFilter>) -> Result<VoiceResults, TtsError> {
        let client = Self::create_client()?;
        let params = voice_filter_to_list_params(filter);

        match client.list_voices(params) {
            Ok(response) => {
                let voices: Vec<VoiceInfo> = response
                    .voices
                    .into_iter()
                    .map(elevenlabs_voice_to_voice_info)
                    .collect();

                Ok(VoiceResults::new(ElevenLabsVoiceResults::new(
                    voices,
                    response.total_count,
                )))
            }
            Err(e) => Err(e),
        }
    }

    fn get_voice(voice_id: String) -> Result<Voice, TtsError> {
        let client = Self::create_client()?;

        match client.get_voice(&voice_id) {
            Ok(voice_data) => {
                let voice_impl = ElevenLabsVoiceImpl::new(voice_data, client);
                Ok(Voice::new(voice_impl))
            }
            Err(e) => Err(e),
        }
    }

    fn search_voices(
        query: String,
        filter: Option<VoiceFilter>,
    ) -> Result<Vec<VoiceInfo>, TtsError> {
        // ElevenLabs doesn't have a dedicated search API, so we use list with search query
        let mut search_filter = filter.unwrap_or(VoiceFilter {
            language: None,
            gender: None,
            quality: None,
            supports_ssml: None,
            provider: None,
            search_query: None,
        });
        search_filter.search_query = Some(query);

        let client = Self::create_client()?;
        let params = voice_filter_to_list_params(Some(search_filter));

        match client.list_voices(params) {
            Ok(response) => {
                let voices = response
                    .voices
                    .into_iter()
                    .map(elevenlabs_voice_to_voice_info)
                    .collect();
                Ok(voices)
            }
            Err(e) => Err(e),
        }
    }

    fn list_languages() -> Result<Vec<LanguageInfo>, TtsError> {
        let client = Self::create_client()?;

        match client.get_models() {
            Ok(models) => Ok(models_to_language_info(models)),
            Err(e) => Err(e),
        }
    }
}

impl SynthesisGuest for ElevenLabsComponent {
    fn synthesize(
        input: TextInput,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<SynthesisResult, TtsError> {
        // Validate input before processing
        Self::validate_synthesis_input(&input, options.as_ref())?;

        let client = Self::create_client()?;
        let voice_id = voice.get::<ElevenLabsVoiceImpl>().get_id();

        let (mut request, params) =
            synthesis_options_to_tts_request(options, client.get_model_version());
        request.text = input.content;

        // Handle language from TextInput with model compatibility
        if let Some(language) = input.language {
            let model_version = client.get_model_version();
            let supports_language_code = !model_version.contains("multilingual");
            
            if supports_language_code {
                request.language_code = Some(language);
            }
            // For multilingual models, ignore the language parameter as it's not supported
        }

        match client.text_to_speech(&voice_id, &request, params) {
            Ok(audio_data) => Ok(audio_data_to_synthesis_result(audio_data, &request.text)),
            Err(e) => Err(e),
        }
    }

    fn synthesize_batch(
        inputs: Vec<TextInput>,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<Vec<SynthesisResult>, TtsError> {
        // Validate all inputs first
        for input in &inputs {
            Self::validate_synthesis_input(input, options.as_ref())?;
        }

        let mut results = Vec::new();
        let client = Self::create_client()?;
        let voice_id = voice.get::<ElevenLabsVoiceImpl>().get_id();

        for input in inputs {
            // For long content, use intelligent chunking
            if input.content.len() > 4500 {
                trace!(
                    "Using long-form batch processing for content with {} characters",
                    input.content.len()
                );

                let audio_chunks = client.synthesize_long_form_batch(
                    &voice_id,
                    &input.content,
                    None, // Use default synthesis options for now
                    4500, // Conservative chunk size
                )?;

                // Combine all chunks into a single result
                let combined_audio: Vec<u8> = audio_chunks.into_iter().flatten().collect();
                let result = audio_data_to_synthesis_result(combined_audio, &input.content);
                results.push(result);
            } else {
                // Use regular synthesis for shorter content
                let (mut request, params) =
                    synthesis_options_to_tts_request(options.clone(), client.get_model_version());
                request.text = input.content.clone();

                // Handle language from TextInput with model compatibility
                if let Some(language) = input.language.clone() {
                    let model_version = client.get_model_version();
                    let supports_language_code = !model_version.contains("multilingual");
                    
                    if supports_language_code {
                        request.language_code = Some(language);
                    }
                    // For multilingual models, ignore the language parameter as it's not supported
                }

                match client.text_to_speech(&voice_id, &request, params) {
                    Ok(audio_data) => {
                        let result = audio_data_to_synthesis_result(audio_data, &input.content);
                        results.push(result);
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        Ok(results)
    }

    fn get_timing_marks(
        _input: TextInput,
        _voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
    ) -> Result<Vec<TimingInfo>, TtsError> {
        // ElevenLabs doesn't provide timing marks without synthesis
        Err(TtsError::UnsupportedOperation(
            "Timing marks not supported by ElevenLabs".to_string(),
        ))
    }

    fn validate_input(
        input: TextInput,
        _voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
    ) -> Result<ValidationResult, TtsError> {
        // Basic validation for ElevenLabs
        let is_valid = !input.content.is_empty() && input.content.len() <= 5000; // ElevenLabs limit
        let message = if !is_valid {
            Some("Text is empty or exceeds 5000 character limit".to_string())
        } else {
            None
        };

        Ok(create_validation_result(is_valid, message))
    }
}

impl StreamingGuest for ElevenLabsComponent {
    type SynthesisStream = ElevenLabsSynthesisStream;
    type VoiceConversionStream = ElevenLabsVoiceConversionStream;

    fn create_stream(
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<SynthesisStream, TtsError> {
        let client = Self::create_client()?;
        let voice_id = voice.get::<ElevenLabsVoiceImpl>().get_id();

        let stream = ElevenLabsSynthesisStream::new(voice_id, client, options);
        Ok(SynthesisStream::new(stream))
    }

    fn create_voice_conversion_stream(
        target_voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Result<VoiceConversionStream, TtsError> {
        let client = Self::create_client()?;
        let voice_id = target_voice.get::<ElevenLabsVoiceImpl>().get_id();

        let stream = ElevenLabsVoiceConversionStream::new(voice_id, client);
        Ok(VoiceConversionStream::new(stream))
    }
}

impl AdvancedGuest for ElevenLabsComponent {
    type PronunciationLexicon = ElevenLabsPronunciationLexicon;
    type LongFormOperation = ElevenLabsLongFormOperation;

    fn create_voice_clone(
        name: String,
        audio_samples: Vec<AudioSample>,
        description: Option<String>,
    ) -> Result<Voice, TtsError> {
        let client = Self::create_client()?;
        let request = create_voice_request_from_samples(name, description, audio_samples);

        match client.create_voice(&request) {
            Ok(voice_data) => {
                let voice_impl = ElevenLabsVoiceImpl::new(voice_data, client);
                Ok(Voice::new(voice_impl))
            }
            Err(e) => Err(e),
        }
    }

    fn design_voice(_name: String, characteristics: VoiceDesignParams) -> Result<Voice, TtsError> {
        let client = Self::create_client()?;
        let request = voice_design_params_to_create_request(characteristics);

        match client.create_voice(&request) {
            Ok(voice_data) => {
                let voice_impl = ElevenLabsVoiceImpl::new(voice_data, client);
                Ok(Voice::new(voice_impl))
            }
            Err(e) => Err(e),
        }
    }

    fn convert_voice(
        input_audio: Vec<u8>,
        target_voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        _preserve_timing: Option<bool>,
    ) -> Result<Vec<u8>, TtsError> {
        let client = Self::create_client()?;
        let voice_id = target_voice.get::<ElevenLabsVoiceImpl>().get_id();

        let request = crate::client::SpeechToSpeechRequest {
            audio_data: input_audio,
            model_id: Some("eleven_english_sts_v2".to_string()),
            voice_settings: None,
            seed: None,
        };

        match client.speech_to_speech(&voice_id, &request, None) {
            Ok(converted_audio) => Ok(converted_audio),
            Err(e) => Err(e),
        }
    }

    fn generate_sound_effect(
        description: String,
        duration_seconds: Option<f32>,
        style_influence: Option<f32>,
    ) -> Result<Vec<u8>, TtsError> {
        let client = Self::create_client()?;

        let request = crate::client::SoundEffectRequest {
            text: description,
            duration_seconds: duration_seconds.map(|d| d as f64),
            prompt_influence: style_influence.map(|s| s as f64),
        };

        let params = crate::client::SoundEffectParams {
            output_format: Some("mp3_22050_32".to_string()),
        };

        match client.create_sound_effect(&request, Some(params)) {
            Ok(audio_data) => Ok(audio_data),
            Err(e) => Err(e),
        }
    }

    fn create_lexicon(
        name: String,
        language: LanguageCode,
        entries: Option<Vec<PronunciationEntry>>,
    ) -> Result<PronunciationLexicon, TtsError> {
        let lexicon = ElevenLabsPronunciationLexicon::new(name, language, entries);
        Ok(PronunciationLexicon::new(lexicon))
    }

    fn synthesize_long_form(
        content: String,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        output_location: String,
        chapter_breaks: Option<Vec<u32>>,
    ) -> Result<LongFormOperation, TtsError> {
        let client = Self::create_client()?;
        let voice_id = voice.get::<ElevenLabsVoiceImpl>().get_id();

        let operation = ElevenLabsLongFormOperation::new(
            content,
            output_location,
            voice_id,
            client,
            chapter_breaks,
        );
        Ok(LongFormOperation::new(operation))
    }
}

impl ExtendedGuest for ElevenLabsComponent {
    fn unwrapped_synthesis_stream(
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Self::SynthesisStream {
        let client = Self::create_client().unwrap_or_else(|_| {
            // Fallback client for unwrapped method
            ElevenLabsTtsApi::new("dummy".to_string(), "eleven_multilingual_v2".to_string())
        });
        let voice_id = voice.get::<ElevenLabsVoiceImpl>().get_id();

        ElevenLabsSynthesisStream::new(voice_id, client, options)
    }

    fn unwrapped_voice_conversion_stream(
        target_voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Self::VoiceConversionStream {
        let client = Self::create_client().unwrap_or_else(|_| {
            // Fallback client for unwrapped method
            ElevenLabsTtsApi::new("dummy".to_string(), "eleven_multilingual_v2".to_string())
        });
        let voice_id = target_voice.get::<ElevenLabsVoiceImpl>().get_id();

        ElevenLabsVoiceConversionStream::new(voice_id, client)
    }

    fn subscribe_synthesis_stream(_stream: &Self::SynthesisStream) -> Pollable {
        golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
    }

    fn subscribe_voice_conversion_stream(_stream: &Self::VoiceConversionStream) -> Pollable {
        golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
    }
}

type DurableElevenLabsComponent = DurableTts<ElevenLabsComponent>;

golem_tts::export_tts!(DurableElevenLabsComponent with_types_in golem_tts);
