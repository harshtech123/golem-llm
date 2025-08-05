use crate::client::{ElevenLabsTtsApi, ListVoicesParams, TextToSpeechParams, Voice as ClientVoice};
use crate::conversions::{
    convert_synthesis_request, convert_voice_filter_from_guest, convert_voice_info_to_guest,
    convert_audio_format_to_string,
};
use golem_rust::wasm_rpc::Pollable;
use golem_tts::config::with_config_keys;
use golem_tts::durability::{DurableTts, ExtendedGuest};
use golem_tts::exports::golem::tts::advanced::Guest as AdvancedGuest;
use golem_tts::exports::golem::tts::streaming::{Guest as StreamingGuest, GuestSynthesisStream, GuestVoiceConversionStream};
use golem_tts::exports::golem::tts::synthesis::Guest as SynthesisGuest;
use golem_tts::exports::golem::tts::voices::Guest as VoicesGuest;
use golem_tts::exports::golem::tts::types::{
    AudioChunk, AudioConfig, AudioEffects, AudioFormat, LanguageCode, SynthesisResult,
    TextInput, TimingInfo, TtsError, VoiceGender, VoiceQuality, VoiceSettings,
};
use golem_tts::exports::golem::tts::voices::{
    LanguageInfo, Voice, VoiceFilter, VoiceInfo, VoiceResults,
};
use golem_tts::exports::golem::tts::synthesis::{SynthesisOptions, ValidationResult};
use golem_tts::exports::golem::tts::streaming::{StreamStatus, SynthesisStream, VoiceConversionStream};
use golem_tts::exports::golem::tts::advanced::{
    AudioSample, VoiceDesignParams, PronunciationLexicon, PronunciationEntry,
    LongFormOperation, LongFormResult, OperationStatus,
};
use std::cell::{Cell, RefCell};
use futures_util::StreamExt;
use log::trace;

mod bindings;
mod client;
mod conversions;

// Streaming implementation for ElevenLabs
struct ElevenLabsSynthesisStream {
    client: ElevenLabsTtsApi,
    voice_id: String,
    request: crate::client::TextToSpeechRequest,
    finished: Cell<bool>,
    buffer: RefCell<Vec<u8>>,
    chunk_size: usize,
    position: Cell<usize>,
}

impl ElevenLabsSynthesisStream {
    pub fn new(
        client: ElevenLabsTtsApi,
        voice_id: String,
        request: crate::client::TextToSpeechRequest,
    ) -> Self {
        Self {
            client,
            voice_id,
            request,
            finished: Cell::new(false),
            buffer: RefCell::new(Vec::new()),
            chunk_size: 4096, // 4KB chunks
            position: Cell::new(0),
        }
    }

    pub fn subscribe(&self) -> Pollable {
        golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
    }

    fn ensure_data_loaded(&self) -> Result<(), TtsError> {
        if self.buffer.borrow().is_empty() && !self.finished.get() {
            match self.client.text_to_speech(&self.voice_id, &self.request, None) {
                Ok(audio_data) => {
                    *self.buffer.borrow_mut() = audio_data;
                }
                Err(e) => {
                    self.finished.set(true);
                    return Err(e);
                }
            }
        }
        Ok(())
    }
}

impl GuestSynthesisStream for ElevenLabsSynthesisStream {
    fn send_text(&self, _input: TextInput) -> Result<(), TtsError> {
        // ElevenLabs API doesn't support streaming text input - we synthesize everything at once
        Err(TtsError::UnsupportedOperation("Streaming text input not supported".to_string()))
    }

    fn finish(&self) -> Result<(), TtsError> {
        self.finished.set(true);
        Ok(())
    }

    fn receive_chunk(&self) -> Result<Option<AudioChunk>, TtsError> {
        if self.finished.get() {
            return Ok(None);
        }

        self.ensure_data_loaded()?;

        let buffer = self.buffer.borrow();
        let current_pos = self.position.get();
        
        if current_pos >= buffer.len() {
            self.finished.set(true);
            return Ok(None);
        }

        let end_pos = std::cmp::min(current_pos + self.chunk_size, buffer.len());
        let chunk_data = buffer[current_pos..end_pos].to_vec();
        
        self.position.set(end_pos);
        
        if end_pos >= buffer.len() {
            self.finished.set(true);
        }

        Ok(Some(AudioChunk {
            data: chunk_data,
            sequence_number: current_pos as u32 / self.chunk_size as u32,
            is_final: self.finished.get(),
            timing_info: None,
        }))
    }

    fn has_pending_audio(&self) -> bool {
        !self.finished.get() && (self.buffer.borrow().is_empty() || self.position.get() < self.buffer.borrow().len())
    }

    fn get_status(&self) -> StreamStatus {
        if self.finished.get() {
            StreamStatus::Finished
        } else {
            StreamStatus::Processing
        }
    }

    fn close(&self) {
        self.finished.set(true);
    }
}

// Voice conversion stream (placeholder - ElevenLabs doesn't have direct voice conversion streaming)
struct ElevenLabsVoiceConversionStream {
    finished: Cell<bool>,
}

impl ElevenLabsVoiceConversionStream {
    pub fn new() -> Self {
        Self {
            finished: Cell::new(true), // Mark as finished since this feature isn't supported
        }
    }

    pub fn subscribe(&self) -> Pollable {
        golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
    }
}

impl GuestVoiceConversionStream for ElevenLabsVoiceConversionStream {
    fn send_audio(&self, _audio_data: Vec<u8>) -> Result<(), TtsError> {
        Err(TtsError::UnsupportedOperation("Voice conversion streaming not supported by ElevenLabs".to_string()))
    }

    fn receive_converted(&self) -> Result<Option<AudioChunk>, TtsError> {
        Ok(None)
    }

    fn finish(&self) -> Result<(), TtsError> {
        Ok(())
    }

    fn close(&self) {
        // Already finished
    }
}

struct ElevenLabsComponent;

impl ElevenLabsComponent {
    const API_KEY_ENV_VAR: &'static str = "ELEVENLABS_API_KEY";

    fn create_client() -> Result<ElevenLabsTtsApi, TtsError> {
        with_config_keys(&[Self::API_KEY_ENV_VAR], |keys| {
            if keys.is_empty() {
                return Err(TtsError::MissingCredentials(
                    "Missing ElevenLabs API key".to_string(),
                ));
            }

            let api_key = keys[0].clone();
            Ok(ElevenLabsTtsApi::new(api_key))
        })
    }
}

impl VoicesGuest for ElevenLabsComponent {
    type Voice = Voice;
    type VoiceResults = VoiceResults;

    fn list_voices(filter: Option<VoiceFilter>) -> Result<VoiceResults, TtsError> {
        let client = Self::create_client()?;
        
        let params = filter.map(convert_voice_filter_from_guest);
        
        match client.list_voices(params) {
            Ok(response) => {
                // Create a simple implementation that doesn't use handle
                // The voice-results resource would normally manage pagination,
                // but for simplicity we'll return all results at once
                let voice_infos: Vec<VoiceInfo> = response
                    .voices
                    .into_iter()
                    .map(convert_voice_info_to_guest)
                    .collect();

                // Return a VoiceResults that contains the results directly
                // This is a simplified implementation
                Ok(VoiceResults::new(voice_infos))
            }
            Err(e) => Err(e),
        }
    }

    fn get_voice(voice_id: String) -> Result<Voice, TtsError> {
        let client = Self::create_client()?;
        
        match client.get_voice(&voice_id) {
            Ok(client_voice) => {
                let voice_info = convert_voice_info_to_guest(client_voice);
                Ok(Voice::new(voice_info))
            }
            Err(e) => Err(e),
        }
    }

    fn search_voices(query: String, filter: Option<VoiceFilter>) -> Result<Vec<VoiceInfo>, TtsError> {
        let mut search_filter = filter.unwrap_or_default();
        search_filter.search_query = Some(query);
        let voices_result = Self::list_voices(Some(search_filter))?;
        // Extract voices from the results - this would normally be done through the resource methods
        // For simplicity, we'll assume we can get all voices at once
        Ok(vec![]) // Placeholder - would need to implement proper voice extraction
    }

    fn list_languages() -> Result<Vec<LanguageInfo>, TtsError> {
        // ElevenLabs supports multiple languages, but we'll return a basic set
        Ok(vec![
            LanguageInfo {
                code: "en".to_string(),
                name: "English".to_string(),
                native_name: "English".to_string(),
                voice_count: 50, // Estimated count
            },
            LanguageInfo {
                code: "es".to_string(),
                name: "Spanish".to_string(),
                native_name: "Español".to_string(),
                voice_count: 20,
            },
            LanguageInfo {
                code: "fr".to_string(),
                name: "French".to_string(),
                native_name: "Français".to_string(),
                voice_count: 15,
            },
            LanguageInfo {
                code: "de".to_string(),
                name: "German".to_string(),
                native_name: "Deutsch".to_string(),
                voice_count: 15,
            },
            LanguageInfo {
                code: "it".to_string(),
                name: "Italian".to_string(),
                native_name: "Italiano".to_string(),
                voice_count: 10,
            },
        ])
    }
}

impl SynthesisGuest for ElevenLabsComponent {
    fn synthesize(
        text: TextInput,
        voice: crate::exports::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<SynthesisResult, TtsError> {
        let client = Self::create_client()?;
        let request = convert_synthesis_request(text, options.clone())?;

        let params = options.as_ref().map(|opts| TextToSpeechParams {
            enable_logging: Some(false),
            optimize_streaming_latency: opts.streaming_config.as_ref().map(|_| 2),
            output_format: opts.audio_config.as_ref().map(|config| {
                convert_audio_format_to_string(&config.format)
            }),
        });

        match client.text_to_speech(&voice.info.id, &request, params) {
            Ok(audio_data) => Ok(SynthesisResult {
                audio_data,
                format: options
                    .and_then(|opts| opts.audio_config)
                    .map(|config| config.format)
                    .unwrap_or(AudioFormat::Mp3),
                sample_rate: 44100,
                duration_ms: None,
                metadata: None,
            }),
            Err(e) => Err(e),
        }
    }

    fn synthesize_batch(
        texts: Vec<TextInput>,
        voice: crate::exports::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<Vec<SynthesisResult>, TtsError> {
        let mut results = Vec::new();
        
        for text in texts {
            let result = Self::synthesize(text, voice, options.clone())?;
            results.push(result);
        }
        
        Ok(results)
    }

    fn get_timing_marks(
        _text: TextInput,
        _voice: crate::exports::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Result<Vec<TimingInfo>, TtsError> {
        // ElevenLabs doesn't provide timing marks in their standard API
        Err(TtsError::UnsupportedFeature("Timing marks not supported by ElevenLabs".to_string()))
    }

    fn validate_input(
        text: TextInput,
        _voice: crate::exports::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Result<ValidationResult, TtsError> {
        let mut warnings = Vec::new();
        let mut is_valid = true;

        if text.content.is_empty() {
            is_valid = false;
        } else if text.content.len() > 5000 {
            warnings.push("Text exceeds recommended length of 5000 characters".to_string());
        }

        if text.content.chars().any(|c| !c.is_ascii()) {
            warnings.push("Text contains non-ASCII characters which may affect pronunciation".to_string());
        }

        Ok(ValidationResult {
            is_valid,
            warnings,
            estimated_duration_ms: Some((text.content.len() as f64 * 80.0) as u32), // Rough estimate
            character_count: text.content.len() as u32,
        })
    }
}

impl StreamingGuest for ElevenLabsComponent {
    type SynthesisStream = ElevenLabsSynthesisStream;
    type VoiceConversionStream = ElevenLabsVoiceConversionStream;

    fn create_stream(
        text: TextInput,
        voice: crate::exports::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<SynthesisStream, TtsError> {
        let client = Self::create_client()?;
        let request = convert_synthesis_request(text, options)?;
        
        let stream = ElevenLabsSynthesisStream::new(client, voice.info.id.clone(), request);
        Ok(SynthesisStream::new(stream))
    }

    fn create_voice_conversion_stream(
        _target_voice: crate::exports::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Result<VoiceConversionStream, TtsError> {
        let stream = ElevenLabsVoiceConversionStream::new();
        Ok(VoiceConversionStream::new(stream))
    }
}

impl AdvancedGuest for ElevenLabsComponent {
    fn create_voice_clone(
        name: String,
        samples: Vec<AudioSample>,
        description: Option<String>,
        _params: Option<VoiceDesignParams>,
    ) -> Result<Voice, TtsError> {
        let client = Self::create_client()?;
        
        let audio_files: Vec<crate::client::AudioFile> = samples
            .into_iter()
            .map(|sample| crate::client::AudioFile { data: sample.data })
            .collect();

        let request = crate::client::CreateVoiceRequest {
            name: name.clone(),
            description,
            files: audio_files,
            labels: None,
        };

        match client.create_voice(&request) {
            Ok(client_voice) => {
                let voice_info = convert_voice_info_to_guest(client_voice);
                Ok(Voice { info: voice_info })
            }
            Err(e) => Err(e),
        }
    }

    fn design_voice(
        _params: VoiceDesignParams,
    ) -> Result<Voice, TtsError> {
        Err(TtsError::UnsupportedFeature("Voice design not supported by ElevenLabs".to_string()))
    }

    fn convert_voice(
        _source_audio: AudioSample,
        _target_voice: crate::exports::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Result<SynthesisResult, TtsError> {
        Err(TtsError::UnsupportedFeature("Voice conversion not supported in this implementation".to_string()))
    }

    fn generate_sound_effect(
        _description: String,
        _duration_ms: Option<u32>,
        _options: Option<SynthesisOptions>,
    ) -> Result<SynthesisResult, TtsError> {
        Err(TtsError::UnsupportedFeature("Sound effect generation not supported by ElevenLabs".to_string()))
    }

    fn create_lexicon(
        _name: String,
        _entries: Vec<PronunciationEntry>,
        _description: Option<String>,
    ) -> Result<PronunciationLexicon, TtsError> {
        Err(TtsError::UnsupportedFeature("Custom lexicons not supported in this implementation".to_string()))
    }

    fn synthesize_long_form(
        _text: TextInput,
        _voice: crate::exports::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Result<LongFormOperation, TtsError> {
        Err(TtsError::UnsupportedFeature("Long form synthesis not supported in this implementation".to_string()))
    }

    fn get_long_form_result(
        _operation_id: String,
    ) -> Result<LongFormResult, TtsError> {
        Err(TtsError::UnsupportedFeature("Long form synthesis not supported in this implementation".to_string()))
    }
}

impl ExtendedGuest for ElevenLabsComponent {
    fn unwrapped_synthesis_stream(
        voice: crate::exports::golem::tts::voices::VoiceBorrow<'_>,
        text: TextInput,
        options: Option<SynthesisOptions>,
    ) -> Self::SynthesisStream {
        let client = Self::create_client()
            .unwrap_or_else(|_| ElevenLabsTtsApi::new("dummy".to_string()));

        let request = convert_synthesis_request(text, options)
            .unwrap_or_else(|_| crate::client::TextToSpeechRequest {
                text: "dummy".to_string(),
                model_id: None,
                language_code: None,
                voice_settings: None,
                pronunciation_dictionary_locators: None,
                seed: None,
                previous_text: None,
                next_text: None,
                previous_request_ids: None,
                next_request_ids: None,
                apply_text_normalization: None,
                apply_language_text_normalization: None,
                use_pvc_as_ivc: None,
            });

        ElevenLabsSynthesisStream::new(client, voice.info.id.clone(), request)
    }

    fn unwrapped_voice_conversion_stream(
        _target_voice: crate::exports::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Self::VoiceConversionStream {
        ElevenLabsVoiceConversionStream::new()
    }

    fn subscribe_synthesis_stream(stream: &Self::SynthesisStream) -> Pollable {
        stream.subscribe()
    }

    fn subscribe_voice_conversion_stream(stream: &Self::VoiceConversionStream) -> Pollable {
        stream.subscribe()
    }
}

type DurableElevenLabsComponent = DurableTts<ElevenLabsComponent>;

golem_tts::export_tts!(DurableElevenLabsComponent with_types_in golem_tts);