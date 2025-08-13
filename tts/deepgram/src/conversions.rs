use crate::client::{Model, TextToSpeechParams, TextToSpeechRequest};
use golem_tts::golem::tts::synthesis::{SynthesisOptions, ValidationResult};
use golem_tts::golem::tts::types::{
    AudioFormat, SynthesisMetadata, SynthesisResult, VoiceGender, VoiceQuality, TtsError, TextType, VoiceSettings,
};
use golem_tts::golem::tts::voices::{LanguageInfo, VoiceInfo};

/// Estimate audio duration in seconds based on audio data size
/// This is a rough estimation for audio data
pub fn estimate_audio_duration(audio_data: &[u8], sample_rate: u32) -> f32 {
    if audio_data.is_empty() {
        return 0.0;
    }

    // For different encoding formats, use appropriate calculations
    // This is a simplified estimation
    let bytes_per_second = match sample_rate {
        8000 => 16000,  // 8kHz * 2 bytes (16-bit)
        16000 => 32000, // 16kHz * 2 bytes (16-bit)
        22050 => 44100, // 22.05kHz * 2 bytes (16-bit)
        24000 => 48000, // 24kHz * 2 bytes (16-bit)
        48000 => 96000, // 48kHz * 2 bytes (16-bit)
        _ => 48000,     // Default fallback
    };

    (audio_data.len() as f32) / (bytes_per_second as f32)
}

pub fn deepgram_model_to_voice_info(model: Model) -> VoiceInfo {
    let gender = parse_gender(&model.gender);
    let quality = infer_quality_from_model(&model.voice_id);

    // Parse language code
    let language = normalize_language_code(&model.language);

    // Deepgram voices are always custom/proprietary
    let is_custom = false;
    let is_cloned = false;

    VoiceInfo {
        id: model.voice_id.clone(),
        name: model.name.clone(),
        language: language.clone(),
        additional_languages: vec![], // Deepgram voices are typically monolingual
        gender,
        quality,
        description: Some(format!(
            "{} voice with {} accent, {}. Characteristics: {}. Suitable for: {}",
            model.gender,
            model.accent,
            model.age,
            model.characteristics.join(", "),
            model.use_cases.join(", ")
        )),
        provider: "Deepgram".to_string(),
        sample_rate: get_default_sample_rate(),
        is_custom,
        is_cloned,
        preview_url: None, // Deepgram doesn't provide preview URLs
        use_cases: model.use_cases.clone(),
    }
}

/// Parse gender from string
pub fn parse_gender(gender_str: &str) -> VoiceGender {
    match gender_str.to_lowercase().as_str() {
        "feminine" | "female" => VoiceGender::Female,
        "masculine" | "male" => VoiceGender::Male,
        _ => VoiceGender::Neutral,
    }
}

/// Infer quality from model name
pub fn infer_quality_from_model(voice_id: &str) -> VoiceQuality {
    if voice_id.starts_with("aura-2-") {
        VoiceQuality::Premium // Aura-2 is the newer, higher quality model
    } else {
        VoiceQuality::Standard // Aura-1 and others are standard quality
    }
}

/// Normalize language codes to standard format
pub fn normalize_language_code(code: &str) -> String {
    match code.to_lowercase().as_str() {
        "en-us" | "en-gb" | "en-au" | "en-ph" | "en-ie" => "en".to_string(),
        "es-es" | "es-mx" | "es-co" | "es-419" => "es".to_string(),
        _ => code.to_lowercase().chars().take(2).collect(),
    }
}

/// Get default sample rate for Deepgram
fn get_default_sample_rate() -> u32 {
    24000 // Deepgram's default sample rate
}

/// Validate SSML content
pub fn validate_ssml(content: &str) -> Result<(), TtsError> {
    // Since Deepgram doesn't support SSML, we should reject it
    if content.trim_start().starts_with('<') && content.contains("speak") {
        return Err(TtsError::InvalidSsml(
            "Deepgram TTS does not support SSML markup".to_string(),
        ));
    }
    Ok(())
}

/// Validate language code
pub fn validate_language_code(language: &str) -> Result<(), TtsError> {
    // Define supported language codes for Deepgram
    let supported_languages = [
        "en", "en-us", "en-gb", "en-au", "en-ph", "en-ie",
        "es", "es-es", "es-mx", "es-co", "es-419",
    ];
    
    let normalized = language.to_lowercase();
    let is_supported = supported_languages.iter().any(|&lang| {
        normalized == lang || normalized.starts_with(&format!("{}-", lang))
    });
    
    if !is_supported {
        return Err(TtsError::UnsupportedLanguage(
            format!("Language '{}' is not supported by Deepgram TTS", language)
        ));
    }
    
    Ok(())
}

/// Validate voice settings
pub fn validate_voice_settings(settings: &VoiceSettings) -> Result<(), TtsError> {
    let mut errors = Vec::new();
    
    // Deepgram doesn't support voice settings modifications
    if settings.speed.is_some() {
        errors.push("Speed adjustment not supported".to_string());
    }
    if settings.pitch.is_some() {
        errors.push("Pitch adjustment not supported".to_string());
    }
    if settings.volume.is_some() {
        errors.push("Volume adjustment not supported".to_string());
    }
    if settings.stability.is_some() {
        errors.push("Stability adjustment not supported".to_string());
    }
    if settings.similarity.is_some() {
        errors.push("Similarity adjustment not supported".to_string());
    }
    if settings.style.is_some() {
        errors.push("Style adjustment not supported".to_string());
    }
    
    if !errors.is_empty() {
        return Err(TtsError::InvalidConfiguration(
            format!("Deepgram TTS does not support voice settings: {}", errors.join(", "))
        ));
    }
    
    Ok(())
}

pub fn synthesis_options_to_tts_request(
    text: String,
    options: Option<SynthesisOptions>,
) -> Result<(TextToSpeechRequest, Option<TextToSpeechParams>), TtsError> {
    let request = TextToSpeechRequest { text: text.clone() };

    let default_params = TextToSpeechParams {
        model: Some(get_default_model()),
        encoding: Some("linear16".to_string()),
        container: Some("wav".to_string()),
        sample_rate: Some(24000),
        bit_rate: None,
    };

    if let Some(opts) = options {
        let mut params = default_params;

        // Validate voice settings if provided
        if let Some(ref voice_settings) = opts.voice_settings {
            validate_voice_settings(voice_settings)?;
        }

        // Map audio config
        if let Some(audio_config) = opts.audio_config {
            let (encoding, container, default_sample_rate, default_bit_rate) =
                audio_format_to_deepgram_params(audio_config.format);
            
            params.encoding = Some(encoding);
            params.container = container;
            
            // For some formats, sample rate is fixed and cannot be configured
            // For fixed-rate formats, DON'T send sample_rate parameter to Deepgram
            match audio_config.format {
                AudioFormat::Mp3 | AudioFormat::Aac => {
                    // Fixed sample rate formats - DO NOT send sample_rate parameter
                    params.sample_rate = None;
                },
                AudioFormat::OggOpus => {
                    // Fixed sample rate for Opus - DO NOT send sample_rate parameter  
                    params.sample_rate = None;
                },
                AudioFormat::Wav | AudioFormat::Pcm | AudioFormat::Flac => {
                    // For uncompressed formats, validate user sample rate against supported rates
                    if let Some(user_rate) = audio_config.sample_rate {
                        // Deepgram supports: 8000, 16000, 24000, 32000, 48000 for WAV/PCM/FLAC
                        let supported_rates = [8000, 16000, 24000, 32000, 48000];
                        if supported_rates.contains(&user_rate) {
                            params.sample_rate = Some(user_rate);
                        } else {
                            // Fall back to closest supported rate
                            params.sample_rate = Some(24000); // Safe default
                        }
                    } else {
                        params.sample_rate = Some(default_sample_rate);
                    }
                },
                _ => {
                    // Use default for other formats
                    params.sample_rate = Some(default_sample_rate);
                }
            }
            
            // For bit rate, use default values for compressed formats (ignore user settings for now)
            match audio_config.format {
                AudioFormat::Mp3 | AudioFormat::OggOpus | AudioFormat::Aac => {
                    // Use the conservative default bit rates we defined
                    params.bit_rate = default_bit_rate;
                },
                _ => {
                    // Uncompressed formats don't use bit rate
                    params.bit_rate = None;
                }
            }
        }

        // Model version mapping
        if let Some(model_version) = opts.model_version {
            params.model = Some(model_version);
        }

        Ok((request, Some(params)))
    } else {
        Ok((request, Some(default_params)))
    }
}

/// Get the default Deepgram model
fn get_default_model() -> String {
    std::env::var("DEEPGRAM_MODEL").unwrap_or_else(|_| "aura-2-asteria-en".to_string())
}

/// Convert AudioFormat to Deepgram parameters
/// Based on Deepgram TTS API documentation: https://developers.deepgram.com/docs/text-to-speech
/// 
/// Audio Format Combinations from official docs:
/// - MP3: encoding=mp3, no container, fixed sample_rate=22050, bit_rate=32000|48000 (default)
/// - Opus: encoding=opus, container=ogg, fixed sample_rate=48000, bit_rate=12000 (default), range: >=4000 and <=650000
/// - AAC: encoding=aac, no container, fixed sample_rate=22050, bit_rate=48000 (default), range: >=4000 and <=192000
/// - Linear16: encoding=linear16, container=wav|none, sample_rate configurable (8k,16k,24k,32k,48k), no bit_rate
fn audio_format_to_deepgram_params(
    format: AudioFormat,
) -> (String, Option<String>, u32, Option<u32>) {
    match format {
        // MP3: Use Deepgram's default bit_rate of 48000 instead of 32000
        AudioFormat::Mp3 => ("mp3".to_string(), None, 22050, Some(48000)),
        // WAV: encoding=linear16, container=wav, sample_rate configurable (8k,16k,24k,32k,48k), no bit_rate
        AudioFormat::Wav => ("linear16".to_string(), Some("wav".to_string()), 24000, None),
        // PCM: encoding=linear16, no container (raw audio), sample_rate configurable, no bit_rate
        AudioFormat::Pcm => (
            "linear16".to_string(),
            None, // No container for raw PCM
            24000,
            None,
        ),
        // Opus: Use Deepgram's default bit_rate of 12000 instead of 32000
        AudioFormat::OggOpus => (
            "opus".to_string(),
            Some("ogg".to_string()),
            48000,
            Some(12000), // Deepgram's documented default
        ),
        // AAC: Use Deepgram's default bit_rate of 48000 instead of 64000
        AudioFormat::Aac => ("aac".to_string(), None, 22050, Some(48000)), // Deepgram's documented default
        // FLAC: encoding=flac, no container, sample_rate configurable, no bit_rate
        AudioFormat::Flac => ("flac".to_string(), None, 48000, None),
        // MULAW: encoding=mulaw, container=wav, sample_rate=8000 or 16000, no bit_rate
        AudioFormat::Mulaw => ("mulaw".to_string(), Some("wav".to_string()), 8000, None),
        // ALAW: encoding=alaw, container=wav, sample_rate=8000 or 16000, no bit_rate
        AudioFormat::Alaw => ("alaw".to_string(), Some("wav".to_string()), 8000, None),
    }
}

pub fn audio_data_to_synthesis_result(
    audio_data: Vec<u8>,
    text: &str,
    encoding: &str,
    sample_rate: u32,
) -> SynthesisResult {
    let audio_size = audio_data.len() as u32;
    let character_count = text.chars().count() as u32;
    let word_count = text.split_whitespace().count() as u32;

    // Estimate duration
    let duration_seconds = estimate_audio_duration(&audio_data, sample_rate);

    let metadata = SynthesisMetadata {
        duration_seconds,
        character_count,
        word_count,
        audio_size_bytes: audio_size,
        request_id: format!("deepgram-{}", chrono::Utc::now().timestamp()),
        provider_info: Some(format!(
            "Deepgram TTS - Encoding: {}, Sample Rate: {}Hz",
            encoding, sample_rate
        )),
    };

    SynthesisResult {
        audio_data,
        metadata,
    }
}

pub fn models_to_language_info(models: Vec<Model>) -> Vec<LanguageInfo> {
    let mut language_map = std::collections::HashMap::new();

    for model in models {
        let lang_code = normalize_language_code(&model.language);

        if !language_map.contains_key(&lang_code) {
            let lang_info = LanguageInfo {
                code: lang_code.clone(),
                name: get_language_name(&lang_code),
                native_name: get_native_language_name(&lang_code),
                voice_count: 0,
            };
            language_map.insert(lang_code.clone(), lang_info);
        }

        if let Some(info) = language_map.get_mut(&lang_code) {
            info.voice_count += 1;
        }
    }

    let mut languages: Vec<LanguageInfo> = language_map.into_values().collect();
    languages.sort_by(|a, b| a.name.cmp(&b.name));
    languages
}

fn get_language_name(language_code: &str) -> String {
    match language_code {
        "en" => "English".to_string(),
        "es" => "Spanish".to_string(),
        _ => language_code.to_uppercase(),
    }
}

fn get_native_language_name(language_code: &str) -> String {
    match language_code {
        "en" => "English".to_string(),
        "es" => "Español".to_string(),
        _ => get_language_name(language_code),
    }
}

/// Validate text input for Deepgram TTS
pub fn validate_text_input(text: &str, model: Option<&str>) -> ValidationResult {
    let character_count = text.chars().count();
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    if text.trim().is_empty() {
        errors.push("Text cannot be empty".to_string());
    }

    // Check for potentially problematic characters
    if text.contains('\0') {
        warnings.push("Text contains null characters which may cause issues".to_string());
    }

    // For very long text, add a warning but don't error (we'll handle chunking automatically)
    let max_chars = get_max_chars_for_model(model);
    if character_count > max_chars {
        warnings.push(format!(
            "Text length ({} characters) exceeds single request limit ({}). Will be automatically chunked.",
            character_count, max_chars
        ));
    }

    let is_valid = errors.is_empty();
    let estimated_duration = if is_valid {
        // Rough estimation: ~150 words per minute
        let word_count = text.split_whitespace().count();
        Some((word_count as f32 / 150.0) * 60.0)
    } else {
        None
    };

    ValidationResult {
        is_valid,
        character_count: character_count as u32,
        estimated_duration,
        warnings,
        errors,
    }
}

/// Get maximum characters for a Deepgram model
pub fn get_max_chars_for_model(model: Option<&str>) -> usize {
    if let Some(m) = model {
        if m.starts_with("aura-2-") {
            1000 // Aura-2 limit
        } else {
            2000 // Aura-1 limit
        }
    } else {
        1000 // Default to stricter limit
    }
}

/// Intelligently split text into chunks suitable for Deepgram TTS
/// Following Deepgram's text chunking best practices
pub fn split_text_intelligently(text: &str, max_chunk_size: usize) -> Vec<String> {
    if text.len() <= max_chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    // First, try to split by paragraphs (double newlines)
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    
    for paragraph in paragraphs {
        // If the paragraph itself is too long, split by sentences
        if paragraph.len() > max_chunk_size {
            let sentences = split_by_sentences(paragraph);
            for sentence in sentences {
                if sentence.len() > max_chunk_size {
                    // If even a single sentence is too long, split by clauses
                    let clauses = split_by_clauses(&sentence, max_chunk_size);
                    for clause in clauses {
                        if current_chunk.len() + clause.len() + 1 <= max_chunk_size {
                            if !current_chunk.is_empty() {
                                current_chunk.push(' ');
                            }
                            current_chunk.push_str(&clause);
                        } else {
                            if !current_chunk.is_empty() {
                                chunks.push(current_chunk.trim().to_string());
                                current_chunk.clear();
                            }
                            current_chunk = clause;
                        }
                    }
                } else {
                    // Normal sentence processing
                    if current_chunk.len() + sentence.len() + 1 <= max_chunk_size {
                        if !current_chunk.is_empty() {
                            current_chunk.push(' ');
                        }
                        current_chunk.push_str(&sentence);
                    } else {
                        if !current_chunk.is_empty() {
                            chunks.push(current_chunk.trim().to_string());
                            current_chunk.clear();
                        }
                        current_chunk = sentence;
                    }
                }
            }
        } else {
            // Paragraph fits within limits
            if current_chunk.len() + paragraph.len() + 2 <= max_chunk_size {
                if !current_chunk.is_empty() {
                    current_chunk.push_str("\n\n");
                }
                current_chunk.push_str(paragraph);
            } else {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk.clear();
                }
                current_chunk = paragraph.to_string();
            }
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    // Ensure no empty chunks
    chunks.into_iter().filter(|chunk| !chunk.trim().is_empty()).collect()
}

/// Split text by sentences, preserving sentence boundaries
fn split_by_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current_sentence = String::new();
    let mut chars = text.chars().peekable();
    
    while let Some(ch) = chars.next() {
        current_sentence.push(ch);
        
        // Check for sentence endings
        if matches!(ch, '.' | '!' | '?') {
            // Look ahead to see if this is actually the end of a sentence
            if let Some(&next_char) = chars.peek() {
                if next_char.is_whitespace() || next_char.is_ascii_uppercase() {
                    sentences.push(current_sentence.trim().to_string());
                    current_sentence.clear();
                }
            } else {
                // End of text
                sentences.push(current_sentence.trim().to_string());
                current_sentence.clear();
            }
        }
    }
    
    if !current_sentence.trim().is_empty() {
        sentences.push(current_sentence.trim().to_string());
    }
    
    sentences.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

/// Split text by clauses (commas, semicolons) when sentences are too long
fn split_by_clauses(text: &str, max_size: usize) -> Vec<String> {
    if text.len() <= max_size {
        return vec![text.to_string()];
    }
    
    let mut clauses = Vec::new();
    let mut current_clause = String::new();
    let mut chars = text.chars().peekable();
    
    while let Some(ch) = chars.next() {
        current_clause.push(ch);
        
        // Check for clause boundaries
        if matches!(ch, ',' | ';' | ':') && current_clause.len() <= max_size {
            if let Some(&next_char) = chars.peek() {
                if next_char.is_whitespace() {
                    clauses.push(current_clause.trim().to_string());
                    current_clause.clear();
                }
            }
        } else if current_clause.len() >= max_size {
            // Force split if we've reached the limit
            if let Some(last_space) = current_clause.rfind(' ') {
                if last_space > max_size / 2 { // Don't split too early
                    let (first_part, second_part) = current_clause.split_at(last_space);
                    clauses.push(first_part.trim().to_string());
                    current_clause = second_part.trim().to_string();
                }
            } else {
                // No good split point, just cut at the limit
                clauses.push(current_clause.trim().to_string());
                current_clause.clear();
            }
        }
    }
    
    if !current_clause.trim().is_empty() {
        clauses.push(current_clause.trim().to_string());
    }
    
    clauses.into_iter().filter(|c| !c.trim().is_empty()).collect()
}

/// Comprehensive validation for synthesis request
pub fn validate_synthesis_request(
    text: &str,
    text_type: TextType,
    language: Option<&str>,
    options: Option<&SynthesisOptions>,
) -> Result<(), TtsError> {
    // Validate empty text
    if text.trim().is_empty() {
        return Err(TtsError::InvalidText("Text cannot be empty".to_string()));
    }
    
    // Validate SSML
    if text_type == TextType::Ssml {
        validate_ssml(text)?;
    }
    
    // Validate language if provided
    if let Some(lang) = language {
        validate_language_code(lang)?;
    }
    
    // Validate voice settings if provided
    if let Some(opts) = options {
        if let Some(ref voice_settings) = opts.voice_settings {
            validate_voice_settings(voice_settings)?;
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gender() {
        assert_eq!(parse_gender("feminine"), VoiceGender::Female);
        assert_eq!(parse_gender("masculine"), VoiceGender::Male);
        assert_eq!(parse_gender("neutral"), VoiceGender::Neutral);
    }

    #[test]
    fn test_normalize_language_code() {
        assert_eq!(normalize_language_code("en-us"), "en");
        assert_eq!(normalize_language_code("es-mx"), "es");
        assert_eq!(normalize_language_code("en-gb"), "en");
    }

    #[test]
    fn test_audio_format_conversion() {
        let (encoding, container, sample_rate, _) =
            audio_format_to_deepgram_params(AudioFormat::Mp3);
        assert_eq!(encoding, "mp3");
        assert_eq!(container, None);
        assert_eq!(sample_rate, 22050);
    }

    #[test]
    fn test_validate_text_input() {
        let result = validate_text_input("Hello, world!", Some("aura-2-thalia-en"));
        assert!(result.is_valid);
        assert_eq!(result.character_count, 13);

        let long_text = "a".repeat(1001);
        let result = validate_text_input(&long_text, Some("aura-2-thalia-en"));
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }
}
