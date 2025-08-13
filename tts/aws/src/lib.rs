use crate::client::{AwsPollyTtsApi, Voice as AwsVoice};
use crate::conversions::{
    audio_data_to_synthesis_result, aws_voice_to_voice_info, get_polly_language_info,
    polly_format_to_audio_format, synthesis_options_to_polly_params, validate_polly_input,
    voice_filter_to_describe_params,
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

// AWS Polly Voice Resource Implementation
struct PollyVoiceImpl {
    voice_data: AwsVoice,
    client: AwsPollyTtsApi,
}

impl PollyVoiceImpl {
    fn new(voice_data: AwsVoice, client: AwsPollyTtsApi) -> Self {
        Self { voice_data, client }
    }
}

impl GuestVoice for PollyVoiceImpl {
    fn get_id(&self) -> String {
        self.voice_data.id.clone()
    }

    fn get_name(&self) -> String {
        self.voice_data.name.clone()
    }

    fn get_provider_id(&self) -> Option<String> {
        Some("aws-polly".to_string())
    }

    fn get_language(&self) -> LanguageCode {
        self.voice_data.language_code.clone()
    }

    fn get_additional_languages(&self) -> Vec<LanguageCode> {
        self.voice_data
            .additional_language_codes
            .clone()
            .unwrap_or_default()
    }

    fn get_gender(&self) -> VoiceGender {
        match self.voice_data.gender.to_lowercase().as_str() {
            "male" => VoiceGender::Male,
            "female" => VoiceGender::Female,
            _ => VoiceGender::Neutral,
        }
    }

    fn get_quality(&self) -> VoiceQuality {
        if self
            .voice_data
            .supported_engines
            .contains(&"neural".to_string())
        {
            VoiceQuality::Neural
        } else if self
            .voice_data
            .supported_engines
            .contains(&"generative".to_string())
        {
            VoiceQuality::Studio
        } else {
            VoiceQuality::Standard
        }
    }

    fn get_description(&self) -> Option<String> {
        Some(format!(
            "{} voice from AWS Polly",
            self.voice_data.language_name
        ))
    }

    fn supports_ssml(&self) -> bool {
        true // AWS Polly supports SSML for all voices
    }

    fn get_sample_rates(&self) -> Vec<u32> {
        // AWS Polly supports these specific sample rates for all engines
        vec![8000, 16000, 22050]
    }

    fn get_supported_formats(&self) -> Vec<AudioFormat> {
        vec![AudioFormat::Mp3, AudioFormat::Pcm, AudioFormat::OggOpus]
    }

    fn update_settings(&self, _settings: VoiceSettings) -> Result<(), TtsError> {
        // AWS Polly doesn't support updating voice settings in the same way
        Err(TtsError::UnsupportedOperation(
            "Voice settings cannot be permanently updated in AWS Polly".to_string(),
        ))
    }

    fn delete(&self) -> Result<(), TtsError> {
        // AWS Polly voices are managed by AWS and cannot be deleted
        Err(TtsError::UnsupportedOperation(
            "AWS Polly voices cannot be deleted".to_string(),
        ))
    }

    fn clone(&self) -> Result<Voice, TtsError> {
        // AWS Polly doesn't support voice cloning in the traditional sense
        Err(TtsError::UnsupportedOperation(
            "Voice cloning not supported for AWS Polly voices".to_string(),
        ))
    }

    fn preview(&self, text: String) -> Result<Vec<u8>, TtsError> {
        // Generate a short preview using the voice
        let preview_text = if text.len() > 100 {
            format!("{}...", &text[..97])
        } else {
            text
        };

        let params =
            synthesis_options_to_polly_params(None, self.voice_data.id.clone(), preview_text);

        self.client.synthesize_speech(params)
    }
}

// AWS Polly Voice Results Implementation
struct PollyVoiceResults {
    voices: RefCell<Vec<VoiceInfo>>,
    current_index: Cell<usize>,
    has_more: Cell<bool>,
    total_count: Option<u32>,
}

impl PollyVoiceResults {
    fn new(voices: Vec<VoiceInfo>, total_count: Option<u32>) -> Self {
        let has_voices = !voices.is_empty();
        Self {
            voices: RefCell::new(voices),
            current_index: Cell::new(0),
            has_more: Cell::new(has_voices), // Set to true if we have voices to return
            total_count,
        }
    }
}

impl GuestVoiceResults for PollyVoiceResults {
    fn has_more(&self) -> bool {
        self.has_more.get()
    }

    fn get_next(&self) -> Result<Vec<VoiceInfo>, TtsError> {
        let voices = self.voices.borrow();
        let start_index = self.current_index.get();
        let batch_size = 10; // Return 10 voices at a time
        let end_index = std::cmp::min(start_index + batch_size, voices.len());

        trace!("PollyVoiceResults::get_next - start: {}, end: {}, total: {}", start_index, end_index, voices.len());

        if start_index >= voices.len() {
            trace!("No more voices to return");
            return Ok(Vec::new());
        }

        let batch = voices[start_index..end_index].to_vec();
        trace!("Returning batch of {} voices", batch.len());
        
        self.current_index.set(end_index);
        self.has_more.set(end_index < voices.len());

        Ok(batch)
    }

    fn get_total_count(&self) -> Option<u32> {
        self.total_count
    }
}

// Synthesis Stream Implementation (simplified for AWS Polly)
struct PollySynthesisStream {
    voice_id: String,
    client: AwsPollyTtsApi,
    text_buffer: RefCell<String>,
    finished: Cell<bool>,
    audio_queue: RefCell<Vec<AudioChunk>>,
    sequence_number: Cell<u32>,
    synthesis_options: RefCell<Option<SynthesisOptions>>,
}

impl PollySynthesisStream {
    fn new(voice_id: String, client: AwsPollyTtsApi, options: Option<SynthesisOptions>) -> Self {
        Self {
            voice_id,
            client,
            text_buffer: RefCell::new(String::new()),
            finished: Cell::new(false),
            audio_queue: RefCell::new(Vec::new()),
            sequence_number: Cell::new(0),
            synthesis_options: RefCell::new(options),
        }
    }
}

impl GuestSynthesisStream for PollySynthesisStream {
    fn send_text(&self, input: TextInput) -> Result<(), TtsError> {
        if self.finished.get() {
            return Err(TtsError::InvalidConfiguration(
                "Stream already finished".to_string(),
            ));
        }

        // For AWS Polly, we'll buffer text and synthesize in chunks
        let mut buffer = self.text_buffer.borrow_mut();
        buffer.push_str(&input.content);

        // If buffer is getting large or contains sentence endings, process it
        if buffer.len() > 1000
            || buffer.ends_with('.')
            || buffer.ends_with('!')
            || buffer.ends_with('?')
        {
            let text_to_process = buffer.clone();
            buffer.clear();

            let params = synthesis_options_to_polly_params(
                self.synthesis_options.borrow().clone(),
                self.voice_id.clone(),
                text_to_process,
            );

            let audio_data = self.client.synthesize_speech(params)?;
            let sequence = self.sequence_number.get();
            self.sequence_number.set(sequence + 1);

            let chunk = AudioChunk {
                data: audio_data,
                sequence_number: sequence,
                is_final: false,
                timing_info: None,
            };

            self.audio_queue.borrow_mut().push(chunk);
        }

        Ok(())
    }

    fn finish(&self) -> Result<(), TtsError> {
        if self.finished.get() {
            return Ok(());
        }

        // Process any remaining text in buffer
        let remaining_text = {
            let mut buffer = self.text_buffer.borrow_mut();
            let text = buffer.clone();
            buffer.clear();
            text
        };

        if !remaining_text.is_empty() {
            let params = synthesis_options_to_polly_params(
                self.synthesis_options.borrow().clone(),
                self.voice_id.clone(),
                remaining_text,
            );

            let audio_data = self.client.synthesize_speech(params)?;
            let sequence = self.sequence_number.get();
            self.sequence_number.set(sequence + 1);

            let chunk = AudioChunk {
                data: audio_data,
                sequence_number: sequence,
                is_final: true,
                timing_info: None,
            };

            self.audio_queue.borrow_mut().push(chunk);
        }

        self.finished.set(true);
        Ok(())
    }

    fn receive_chunk(&self) -> Result<Option<AudioChunk>, TtsError> {
        let mut queue = self.audio_queue.borrow_mut();
        Ok(queue.pop())
    }

    fn has_pending_audio(&self) -> bool {
        !self.audio_queue.borrow().is_empty()
    }

    fn get_status(&self) -> StreamStatus {
        if self.finished.get() && self.audio_queue.borrow().is_empty() {
            StreamStatus::Finished
        } else if self.finished.get() {
            StreamStatus::Processing
        } else {
            StreamStatus::Ready
        }
    }

    fn close(&self) {
        self.finished.set(true);
        self.audio_queue.borrow_mut().clear();
        self.text_buffer.borrow_mut().clear();
    }
}

// Voice Conversion Stream (not supported by AWS Polly directly)
struct PollyVoiceConversionStream {
    #[allow(dead_code)]
    voice_id: String,
    finished: Cell<bool>,
}

impl PollyVoiceConversionStream {
    fn new(voice_id: String, _client: AwsPollyTtsApi) -> Self {
        Self {
            voice_id,
            finished: Cell::new(false),
        }
    }
}

impl GuestVoiceConversionStream for PollyVoiceConversionStream {
    fn send_audio(&self, _audio_data: Vec<u8>) -> Result<(), TtsError> {
        Err(TtsError::UnsupportedOperation(
            "Voice conversion not supported by AWS Polly".to_string(),
        ))
    }

    fn receive_converted(&self) -> Result<Option<AudioChunk>, TtsError> {
        Err(TtsError::UnsupportedOperation(
            "Voice conversion not supported by AWS Polly".to_string(),
        ))
    }

    fn finish(&self) -> Result<(), TtsError> {
        self.finished.set(true);
        Ok(())
    }

    fn close(&self) {
        self.finished.set(true);
    }
}

// Pronunciation Lexicon Implementation
struct PollyPronunciationLexicon {
    name: String,
    language: LanguageCode,
    entries: RefCell<Vec<PronunciationEntry>>,
    client: AwsPollyTtsApi,
}

impl PollyPronunciationLexicon {
    fn new(
        name: String,
        language: LanguageCode,
        entries: Option<Vec<PronunciationEntry>>,
        client: AwsPollyTtsApi,
    ) -> Self {
        Self {
            name,
            language,
            entries: RefCell::new(entries.unwrap_or_default()),
            client,
        }
    }
}

impl GuestPronunciationLexicon for PollyPronunciationLexicon {
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
        let entry = PronunciationEntry {
            word,
            pronunciation,
            part_of_speech: None,
        };
        self.entries.borrow_mut().push(entry);

        // Sync with AWS Polly lexicon
        self.sync_lexicon()?;
        Ok(())
    }

    fn remove_entry(&self, word: String) -> Result<(), TtsError> {
        self.entries.borrow_mut().retain(|entry| entry.word != word);

        // Sync with AWS Polly lexicon
        self.sync_lexicon()?;
        Ok(())
    }

    fn export_content(&self) -> Result<String, TtsError> {
        let entries = self.entries.borrow();
        let mut lexicon_content = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<lexicon version="1.0" xmlns="http://www.w3.org/2005/01/pronunciation-lexicon" alphabet="ipa" xml:lang="{}">
"#,
            self.language
        );

        for entry in entries.iter() {
            lexicon_content.push_str(&format!(
                r#"  <lexeme>
    <grapheme>{}</grapheme>
    <phoneme>{}</phoneme>
  </lexeme>
"#,
                entry.word, entry.pronunciation
            ));
        }

        lexicon_content.push_str("</lexicon>");
        Ok(lexicon_content)
    }
}

impl PollyPronunciationLexicon {
    fn sync_lexicon(&self) -> Result<(), TtsError> {
        let content = self.export_content()?;
        self.client.put_lexicon(&self.name, &content)?;
        Ok(())
    }
}

// Long Form Operation Implementation using AWS Polly synthesis tasks
struct PollyLongFormOperation {
    task_id: String,
    client: AwsPollyTtsApi,
    status: Cell<OperationStatus>,
    progress: Cell<f32>,
}

impl PollyLongFormOperation {
    fn new(task_id: String, client: AwsPollyTtsApi) -> Self {
        Self {
            task_id,
            client,
            status: Cell::new(OperationStatus::Pending),
            progress: Cell::new(0.0),
        }
    }

    fn update_status(&self) -> Result<(), TtsError> {
        let task = self.client.get_speech_synthesis_task(&self.task_id)?;

        let status = match task.task_status.as_deref() {
            Some("scheduled") => OperationStatus::Pending,
            Some("inProgress") => OperationStatus::Processing,
            Some("completed") => OperationStatus::Completed,
            Some("failed") => OperationStatus::Failed,
            _ => OperationStatus::Pending,
        };

        self.status.set(status);

        // AWS Polly doesn't provide detailed progress, so we estimate
        let progress = match status {
            OperationStatus::Pending => 0.0,
            OperationStatus::Processing => 0.5,
            OperationStatus::Completed => 1.0,
            OperationStatus::Failed => 0.0,
            OperationStatus::Cancelled => 0.0,
        };
        self.progress.set(progress);

        Ok(())
    }
}

impl GuestLongFormOperation for PollyLongFormOperation {
    fn get_status(&self) -> OperationStatus {
        // Update status from AWS before returning
        let _ = self.update_status();
        self.status.get()
    }

    fn get_progress(&self) -> f32 {
        self.progress.get()
    }

    fn cancel(&self) -> Result<(), TtsError> {
        // AWS Polly doesn't support canceling synthesis tasks once started
        Err(TtsError::UnsupportedOperation(
            "Cannot cancel AWS Polly synthesis tasks".to_string(),
        ))
    }

    fn get_result(&self) -> Result<LongFormResult, TtsError> {
        let task = self.client.get_speech_synthesis_task(&self.task_id)?;

        if let Some(output_uri) = task.output_uri {
            Ok(LongFormResult {
                output_location: output_uri,
                total_duration: 0.0, // Would need to calculate from the output
                chapter_durations: None,
                metadata: golem_tts::golem::tts::types::SynthesisMetadata {
                    duration_seconds: 0.0,
                    character_count: task.request_characters.unwrap_or(0) as u32,
                    word_count: 0,       // Would need to calculate
                    audio_size_bytes: 0, // Would need to get from S3 object
                    request_id: self.task_id.clone(),
                    provider_info: Some("AWS Polly".to_string()),
                },
            })
        } else {
            Err(TtsError::InternalError(
                "Task result not available".to_string(),
            ))
        }
    }
}

// Main AWS Polly Component
struct AwsPollyComponent;

impl AwsPollyComponent {
    const ACCESS_KEY_ENV_VAR: &'static str = "AWS_ACCESS_KEY_ID";
    const SECRET_KEY_ENV_VAR: &'static str = "AWS_SECRET_ACCESS_KEY";
    const REGION_ENV_VAR: &'static str = "AWS_REGION";
    const SESSION_TOKEN_ENV_VAR: &'static str = "AWS_SESSION_TOKEN";

    fn create_client() -> Result<AwsPollyTtsApi, TtsError> {
        with_config_key(Self::ACCESS_KEY_ENV_VAR, Err, |access_key_id| {
            with_config_key(Self::SECRET_KEY_ENV_VAR, Err, |secret_access_key| {
                let region =
                    std::env::var(Self::REGION_ENV_VAR).unwrap_or_else(|_| "us-east-1".to_string());
                let session_token = std::env::var(Self::SESSION_TOKEN_ENV_VAR).ok();

                Ok(AwsPollyTtsApi::new(
                    access_key_id.to_string(),
                    secret_access_key.to_string(),
                    region,
                    session_token,
                )?)
            })
        })
    }

    /// Validate synthesis input and options for proper error handling
    fn validate_synthesis_input(
        input: &TextInput,
        options: Option<&SynthesisOptions>,
    ) -> Result<(), TtsError> {
        use golem_tts::golem::tts::types::TextType;
        
        // Validate empty text
        if input.content.trim().is_empty() {
            return Err(TtsError::InvalidText("Text content cannot be empty".to_string()));
        }

        // Validate text length (AWS Polly has limits)
        if input.content.len() > 3000 {
            return Err(TtsError::InvalidText("Text exceeds AWS Polly limit of 3000 characters".to_string()));
        }

        // Validate SSML content if specified
        if input.text_type == TextType::Ssml {
            if let Err(msg) = Self::validate_ssml_content(&input.content) {
                return Err(TtsError::InvalidSsml(msg));
            }
        }

        // Validate language code if specified
        if let Some(ref language) = input.language {
            if !Self::is_supported_language(language) {
                return Err(TtsError::UnsupportedLanguage(format!(
                    "Language '{}' is not supported by AWS Polly", language
                )));
            }
        }

        // Validate voice settings if specified
        if let Some(opts) = options {
            if let Some(ref voice_settings) = opts.voice_settings {
                Self::validate_voice_settings(voice_settings)?;
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
    /// Check if a language is supported by AWS Polly
    fn is_supported_language(language: &str) -> bool {
        let supported_languages = [
            "en-US", "en-GB", "en-AU", "en-IN",
            "es-ES", "es-MX", "es-US",
            "fr-FR", "fr-CA",
            "de-DE", "it-IT",
            "pt-PT", "pt-BR",
            "ja-JP", "ko-KR",
            "zh-CN", "cmn-CN",
            "ar", "hi-IN", "ru-RU",
            "nl-NL", "pl-PL", "sv-SE",
            "nb-NO", "da-DK", "tr-TR",
            "ro-RO", "cy-GB", "is-IS"
        ];
        supported_languages.contains(&language)
    }

    /// Validate voice settings for AWS Polly limits
    fn validate_voice_settings(settings: &VoiceSettings) -> Result<(), TtsError> {
        // Validate speed (0.25x to 4.0x)
        if let Some(speed) = settings.speed {
            if speed < 0.25 || speed > 4.0 {
                return Err(TtsError::InvalidConfiguration(
                    "Speed must be between 0.25 and 4.0".to_string()
                ));
            }
        }

        // Validate pitch (-20dB to +20dB in semitones, roughly -10.0 to +10.0)
        if let Some(pitch) = settings.pitch {
            if pitch < -10.0 || pitch > 10.0 {
                return Err(TtsError::InvalidConfiguration(
                    "Pitch must be between -10.0 and +10.0".to_string()
                ));
            }
        }

        // Validate volume (-20dB to +20dB, roughly -20.0 to +20.0)
        if let Some(volume) = settings.volume {
            if volume < -20.0 || volume > 20.0 {
                return Err(TtsError::InvalidConfiguration(
                    "Volume must be between -20.0 and +20.0".to_string()
                ));
            }
        }

        // Validate stability (0.0 to 1.0) - AWS Polly doesn't directly support this
        if let Some(stability) = settings.stability {
            if stability < 0.0 || stability > 1.0 {
                return Err(TtsError::InvalidConfiguration(
                    "Stability must be between 0.0 and 1.0".to_string()
                ));
            }
        }

        // Validate similarity (0.0 to 1.0) - AWS Polly doesn't directly support this
        if let Some(similarity) = settings.similarity {
            if similarity < 0.0 || similarity > 1.0 {
                return Err(TtsError::InvalidConfiguration(
                    "Similarity must be between 0.0 and 1.0".to_string()
                ));
            }
        }

        // Validate style (0.0 to 1.0) - AWS Polly doesn't directly support this
        if let Some(style) = settings.style {
            if style < 0.0 || style > 1.0 {
                return Err(TtsError::InvalidConfiguration(
                    "Style must be between 0.0 and 1.0".to_string()
                ));
            }
        }

        Ok(())
    }
}

impl VoicesGuest for AwsPollyComponent {
    type Voice = PollyVoiceImpl;
    type VoiceResults = PollyVoiceResults;

    fn list_voices(filter: Option<VoiceFilter>) -> Result<VoiceResults, TtsError> {
        let client = Self::create_client()?;
        let params = voice_filter_to_describe_params(filter);

        let response = client.describe_voices(params)?;
        trace!("AWS describe_voices returned {} voices", response.voices.len());
        
        let voice_infos: Vec<VoiceInfo> = response
            .voices
            .into_iter()
            .enumerate()
            .map(|(i, voice)| {
                trace!("Converting voice {}: {} ({})", i, voice.name, voice.id);
                aws_voice_to_voice_info(voice)
            })
            .collect();

        trace!("Converted to {} VoiceInfo objects", voice_infos.len());
        let total_count = Some(voice_infos.len() as u32);
        Ok(VoiceResults::new(PollyVoiceResults::new(
            voice_infos,
            total_count,
        )))
    }

    fn get_voice(voice_id: String) -> Result<Voice, TtsError> {
        let client = Self::create_client()?;
        let response = client.describe_voices(None)?;

        let voice_data = response
            .voices
            .into_iter()
            .find(|v| v.id == voice_id)
            .ok_or_else(|| TtsError::VoiceNotFound(voice_id.clone()))?;

        Ok(Voice::new(PollyVoiceImpl::new(voice_data, client)))
    }

    fn search_voices(
        query: String,
        filter: Option<VoiceFilter>,
    ) -> Result<Vec<VoiceInfo>, TtsError> {
        let client = Self::create_client()?;
        let params = voice_filter_to_describe_params(filter);

        let response = client.describe_voices(params)?;
        let voice_infos: Vec<VoiceInfo> = response
            .voices
            .into_iter()
            .filter(|voice| {
                voice.name.to_lowercase().contains(&query.to_lowercase())
                    || voice
                        .language_name
                        .to_lowercase()
                        .contains(&query.to_lowercase())
                    || voice
                        .language_code
                        .to_lowercase()
                        .contains(&query.to_lowercase())
            })
            .map(aws_voice_to_voice_info)
            .collect();

        Ok(voice_infos)
    }

    fn list_languages() -> Result<Vec<LanguageInfo>, TtsError> {
        Ok(get_polly_language_info())
    }
}

impl SynthesisGuest for AwsPollyComponent {
    fn synthesize(
        input: TextInput,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<SynthesisResult, TtsError> {
        // Validate input before processing
        Self::validate_synthesis_input(&input, options.as_ref())?;
        
        let client = Self::create_client()?;
        let voice_id = voice.get::<PollyVoiceImpl>().get_id();

        let params = synthesis_options_to_polly_params(options, voice_id, input.content.clone());
        let audio_data = client.synthesize_speech(params.clone())?;

        let format = polly_format_to_audio_format(&params.output_format);
        let sample_rate = params
            .sample_rate
            .and_then(|s| s.parse().ok())
            .unwrap_or(22050);

        Ok(audio_data_to_synthesis_result(
            audio_data,
            &input.content,
            &format,
            sample_rate,
        ))
    }

    fn synthesize_batch(
        inputs: Vec<TextInput>,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<Vec<SynthesisResult>, TtsError> {
        let client = Self::create_client()?;
        let voice_id = voice.get::<PollyVoiceImpl>().get_id();
        let mut results = Vec::new();

        for input in inputs {
            // Validate each input before processing
            Self::validate_synthesis_input(&input, options.as_ref())?;
            
            let params = synthesis_options_to_polly_params(
                options.clone(),
                voice_id.clone(),
                input.content.clone(),
            );
            let audio_data = client.synthesize_speech(params.clone())?;

            let format = polly_format_to_audio_format(&params.output_format);
            let sample_rate = params
                .sample_rate
                .and_then(|s| s.parse().ok())
                .unwrap_or(22050);

            results.push(audio_data_to_synthesis_result(
                audio_data,
                &input.content,
                &format,
                sample_rate,
            ));
        }

        Ok(results)
    }

    fn get_timing_marks(
        input: TextInput,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
    ) -> Result<Vec<TimingInfo>, TtsError> {
        // AWS Polly supports speech marks, but this would require a separate API call
        // For now, return empty timing info
        let voice_id = voice.get::<PollyVoiceImpl>().get_id();
        trace!(
            "Getting timing marks for voice: {}, text: {}",
            voice_id,
            input.content
        );
        Ok(Vec::new())
    }

    fn validate_input(
        input: TextInput,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
    ) -> Result<ValidationResult, TtsError> {
        let voice_id = voice.get::<PollyVoiceImpl>().get_id();
        Ok(validate_polly_input(&input.content, &voice_id))
    }
}

impl StreamingGuest for AwsPollyComponent {
    type SynthesisStream = PollySynthesisStream;
    type VoiceConversionStream = PollyVoiceConversionStream;

    fn create_stream(
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Result<SynthesisStream, TtsError> {
        let client = Self::create_client()?;
        let voice_id = voice.get::<PollyVoiceImpl>().get_id();

        Ok(SynthesisStream::new(PollySynthesisStream::new(
            voice_id, client, options,
        )))
    }

    fn create_voice_conversion_stream(
        target_voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Result<VoiceConversionStream, TtsError> {
        let client = Self::create_client()?;
        let voice_id = target_voice.get::<PollyVoiceImpl>().get_id();

        Ok(VoiceConversionStream::new(PollyVoiceConversionStream::new(
            voice_id, client,
        )))
    }
}

impl AdvancedGuest for AwsPollyComponent {
    type PronunciationLexicon = PollyPronunciationLexicon;
    type LongFormOperation = PollyLongFormOperation;

    fn create_voice_clone(
        _name: String,
        _audio_samples: Vec<AudioSample>,
        _description: Option<String>,
    ) -> Result<golem_tts::golem::tts::voices::Voice, TtsError> {
        Err(TtsError::UnsupportedOperation(
            "Voice cloning not supported by AWS Polly".to_string(),
        ))
    }

    fn design_voice(
        _name: String,
        _characteristics: VoiceDesignParams,
    ) -> Result<golem_tts::golem::tts::voices::Voice, TtsError> {
        Err(TtsError::UnsupportedOperation(
            "Voice design not supported by AWS Polly".to_string(),
        ))
    }

    fn convert_voice(
        _input_audio: Vec<u8>,
        _target_voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        _preserve_timing: Option<bool>,
    ) -> Result<Vec<u8>, TtsError> {
        Err(TtsError::UnsupportedOperation(
            "Voice conversion not supported by AWS Polly".to_string(),
        ))
    }

    fn generate_sound_effect(
        _description: String,
        _duration_seconds: Option<f32>,
        _style_influence: Option<f32>,
    ) -> Result<Vec<u8>, TtsError> {
        Err(TtsError::UnsupportedOperation(
            "Sound effect generation not supported by AWS Polly".to_string(),
        ))
    }

    fn create_lexicon(
        name: String,
        language: LanguageCode,
        entries: Option<Vec<PronunciationEntry>>,
    ) -> Result<PronunciationLexicon, TtsError> {
        let client = Self::create_client()?;
        let lexicon = PollyPronunciationLexicon::new(name, language, entries, client);

        // Create empty lexicon in AWS Polly
        lexicon.sync_lexicon()?;

        Ok(PronunciationLexicon::new(lexicon))
    }

    fn synthesize_long_form(
        content: String,
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        output_location: String,
        _chapter_breaks: Option<Vec<u32>>,
    ) -> Result<LongFormOperation, TtsError> {
        let client = Self::create_client()?;
        let voice_id = voice.get::<PollyVoiceImpl>().get_id();
        let voice_language = voice.get::<PollyVoiceImpl>().get_language();

        // Extract S3 bucket name from output location
        let bucket_name = if output_location.starts_with("s3://") {
            output_location
                .strip_prefix("s3://")
                .and_then(|path| path.split('/').next())
                .unwrap_or("default-bucket")
                .to_string()
        } else {
            "default-bucket".to_string()
        };

        let params = crate::client::StartSpeechSynthesisTaskParams {
            engine: Some(crate::client::Engine::Neural),
            language_code: Some(voice_language),
            lexicon_names: None,
            output_format: crate::client::OutputFormat::Mp3,
            output_s3_bucket_name: bucket_name,
            output_s3_key_prefix: Some("polly-output".to_string()),
            sample_rate: None,
            sns_topic_arn: None,
            speech_mark_types: None,
            text: content,
            text_type: Some(crate::client::TextType::Text),
            voice_id,
        };

        let task = client.start_speech_synthesis_task(params)?;
        let task_id = task
            .task_id
            .ok_or_else(|| TtsError::InternalError("No task ID returned".to_string()))?;

        Ok(LongFormOperation::new(PollyLongFormOperation::new(
            task_id, client,
        )))
    }
}

impl ExtendedGuest for AwsPollyComponent {
    fn unwrapped_synthesis_stream(
        voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        options: Option<SynthesisOptions>,
    ) -> Self::SynthesisStream {
        let client = Self::create_client().unwrap_or_else(|_| {
            // Fallback client for unwrapped method
            AwsPollyTtsApi::new(
                "dummy".to_string(),
                "dummy".to_string(),
                "us-east-1".to_string(),
                None,
            ).unwrap_or_else(|_| panic!("Failed to create fallback client"))
        });
        let voice_id = voice.get::<PollyVoiceImpl>().get_id();

        PollySynthesisStream::new(voice_id, client, options)
    }

    fn unwrapped_voice_conversion_stream(
        target_voice: golem_tts::golem::tts::voices::VoiceBorrow<'_>,
        _options: Option<SynthesisOptions>,
    ) -> Self::VoiceConversionStream {
        let client = Self::create_client().unwrap_or_else(|_| {
            // Fallback client for unwrapped method
            AwsPollyTtsApi::new(
                "dummy".to_string(),
                "dummy".to_string(),
                "us-east-1".to_string(),
                None,
            ).unwrap_or_else(|_| panic!("Failed to create fallback client"))
        });
        let voice_id = target_voice.get::<PollyVoiceImpl>().get_id();

        PollyVoiceConversionStream::new(voice_id, client)
    }

    fn subscribe_synthesis_stream(_stream: &Self::SynthesisStream) -> Pollable {
        golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
    }

    fn subscribe_voice_conversion_stream(_stream: &Self::VoiceConversionStream) -> Pollable {
        golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
    }
}

type DurableAwsPollyComponent = DurableTts<AwsPollyComponent>;

golem_tts::export_tts!(DurableAwsPollyComponent with_types_in golem_tts);
