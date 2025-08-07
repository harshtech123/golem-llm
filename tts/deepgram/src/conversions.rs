use crate::client::{Model, TextToSpeechParams, TextToSpeechRequest};
use golem_tts::golem::tts::synthesis::{SynthesisOptions, ValidationResult};
use golem_tts::golem::tts::types::{
    AudioFormat, SynthesisMetadata, SynthesisResult, VoiceGender, VoiceQuality,
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

pub fn synthesis_options_to_tts_request(
    text: String,
    options: Option<SynthesisOptions>,
) -> (TextToSpeechRequest, Option<TextToSpeechParams>) {
    let request = TextToSpeechRequest { text };

    let default_params = TextToSpeechParams {
        model: Some(get_default_model()),
        encoding: Some("linear16".to_string()),
        container: Some("wav".to_string()),
        sample_rate: Some(24000),
        bit_rate: None,
    };

    if let Some(opts) = options {
        let mut params = default_params;

        // Map audio config
        if let Some(audio_config) = opts.audio_config {
            let (encoding, container, sample_rate, bit_rate) =
                audio_format_to_deepgram_params(audio_config.format);
            params.encoding = Some(encoding);
            params.container = container;
            params.sample_rate = Some(sample_rate);
            params.bit_rate = bit_rate;
        }

        // Model version mapping
        if let Some(model_version) = opts.model_version {
            params.model = Some(model_version);
        }

        (request, Some(params))
    } else {
        (request, Some(default_params))
    }
}

/// Get the default Deepgram model
fn get_default_model() -> String {
    std::env::var("DEEPGRAM_MODEL").unwrap_or_else(|_| "aura-2-asteria-en".to_string())
}

/// Convert AudioFormat to Deepgram parameters
fn audio_format_to_deepgram_params(
    format: AudioFormat,
) -> (String, Option<String>, u32, Option<u32>) {
    match format {
        AudioFormat::Mp3 => ("mp3".to_string(), None, 22050, Some(48000)),
        AudioFormat::Wav => ("linear16".to_string(), Some("wav".to_string()), 24000, None),
        AudioFormat::Pcm => (
            "linear16".to_string(),
            Some("none".to_string()),
            24000,
            None,
        ),
        AudioFormat::OggOpus => (
            "opus".to_string(),
            Some("ogg".to_string()),
            48000,
            Some(12000),
        ),
        AudioFormat::Aac => ("aac".to_string(), None, 22050, Some(48000)),
        AudioFormat::Flac => ("flac".to_string(), None, 48000, None),
        AudioFormat::Mulaw => ("mulaw".to_string(), Some("wav".to_string()), 8000, None),
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

    // Check character limits based on model
    let max_chars = if let Some(m) = model {
        if m.starts_with("aura-2-") {
            1000 // Aura-2 limit
        } else {
            2000 // Aura-1 limit
        }
    } else {
        1000 // Default to stricter limit
    };

    if character_count > max_chars {
        errors.push(format!(
            "Text exceeds maximum character limit of {} characters (current: {})",
            max_chars, character_count
        ));
    }

    if text.trim().is_empty() {
        errors.push("Text cannot be empty".to_string());
    }

    // Check for potentially problematic characters
    if text.contains('\0') {
        warnings.push("Text contains null characters which may cause issues".to_string());
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
