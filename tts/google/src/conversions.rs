use crate::client::{
    AudioConfig, AudioEncoding, SsmlVoiceGender, SynthesisInput, 
    SynthesizeSpeechRequest, Voice as GoogleVoice, VoiceSelectionParams,
    BatchSynthesisParams
};
use golem_tts::golem::tts::types::{
    AudioFormat, SynthesisResult, SynthesisMetadata, VoiceGender,
    VoiceQuality, VoiceSettings, TextInput,
};
use golem_tts::golem::tts::voices::{LanguageInfo, VoiceFilter, VoiceInfo};
use golem_tts::golem::tts::synthesis::{SynthesisOptions, ValidationResult};
use uuid;

/// Estimate audio duration in seconds based on audio data size
/// This is a rough estimation for different audio formats
pub fn estimate_audio_duration(audio_data: &[u8], sample_rate: u32, encoding: &AudioEncoding) -> f32 {
    if audio_data.is_empty() {
        return 0.0;
    }
    
    match encoding {
        AudioEncoding::Linear16 | AudioEncoding::Pcm => {
            // For LINEAR16: 16 bits per sample * 1 channel
            let bytes_per_second = (sample_rate * 2) as f32;
            audio_data.len() as f32 / bytes_per_second
        }
        AudioEncoding::Mp3 => {
            // For MP3, estimate based on typical bitrate (128 kbps)
            let estimated_bitrate_bps = 128000; // 128 kbps
            let bytes_per_second = estimated_bitrate_bps / 8;
            audio_data.len() as f32 / bytes_per_second as f32
        }
        AudioEncoding::OggOpus => {
            // For OGG Opus, estimate based on typical bitrate (64 kbps)
            let estimated_bitrate_bps = 64000; // 64 kbps
            let bytes_per_second = estimated_bitrate_bps / 8;
            audio_data.len() as f32 / bytes_per_second as f32
        }
        _ => {
            // Default estimation for other formats
            let estimated_bitrate_bps = 96000; // 96 kbps average
            let bytes_per_second = estimated_bitrate_bps / 8;
            audio_data.len() as f32 / bytes_per_second as f32
        }
    }
}

pub fn voice_filter_to_language_code(filter: Option<VoiceFilter>) -> Option<String> {
    filter.and_then(|f| f.language)
}

pub fn google_voice_to_voice_info(voice: GoogleVoice) -> VoiceInfo {
    let gender = ssml_gender_to_voice_gender(&voice.ssml_gender);
    let quality = infer_quality_from_voice(&voice);
    
    // Extract primary language from language codes
    let language = voice.language_codes.first()
        .map(|code| normalize_language_code(code))
        .unwrap_or_else(|| "en-US".to_string());
    
    let additional_languages = voice.language_codes
        .iter()
        .skip(1)
        .map(|code| normalize_language_code(code))
        .collect();
    
    let use_cases = infer_use_cases_from_voice_name(&voice.name);
    let voice_type = extract_voice_type_from_name(&voice.name);
    let description = generate_voice_description(&voice);
    
    VoiceInfo {
        id: voice.name.clone(),
        name: extract_display_name(&voice.name),
        language,
        additional_languages,
        gender,
        quality,
        description: Some(description),
        provider: "google".to_string(),
        sample_rate: voice.natural_sample_rate_hertz as u32,
        is_custom: voice_type.contains("Custom"),
        is_cloned: false, // Google doesn't have voice cloning like ElevenLabs
        preview_url: None, // Google doesn't provide preview URLs
        use_cases,
    }
}

/// Convert Google's SsmlVoiceGender to our VoiceGender
pub fn ssml_gender_to_voice_gender(gender: &SsmlVoiceGender) -> VoiceGender {
    match gender {
        SsmlVoiceGender::Male => VoiceGender::Male,
        SsmlVoiceGender::Female => VoiceGender::Female,
        SsmlVoiceGender::Neutral => VoiceGender::Neutral,
        SsmlVoiceGender::SsmlVoiceGenderUnspecified => VoiceGender::Neutral,
    }
}

/// Convert our VoiceGender to Google's SsmlVoiceGender
#[allow(dead_code)]
pub fn voice_gender_to_ssml_gender(gender: &VoiceGender) -> SsmlVoiceGender {
    match gender {
        VoiceGender::Male => SsmlVoiceGender::Male,
        VoiceGender::Female => SsmlVoiceGender::Female,
        VoiceGender::Neutral => SsmlVoiceGender::Neutral,
    }
}

/// Infer voice quality from Google voice characteristics
pub fn infer_quality_from_voice(voice: &GoogleVoice) -> VoiceQuality {
    let voice_name = voice.name.to_lowercase();
    
    if voice_name.contains("wavenet") {
        VoiceQuality::Premium
    } else if voice_name.contains("neural2") || voice_name.contains("journey") {
        VoiceQuality::Premium
    } else if voice_name.contains("polyglot") {
        VoiceQuality::Premium
    } else if voice_name.contains("studio") {
        VoiceQuality::Premium
    } else if voice_name.contains("standard") {
        VoiceQuality::Standard
    } else {
        VoiceQuality::Standard
    }
}

/// Extract voice type from Google voice name
fn extract_voice_type_from_name(name: &str) -> String {
    if name.contains("Wavenet") {
        "WaveNet".to_string()
    } else if name.contains("Neural2") {
        "Neural2".to_string()
    } else if name.contains("Journey") {
        "Journey".to_string()
    } else if name.contains("Polyglot") {
        "Polyglot".to_string()
    } else if name.contains("Studio") {
        "Studio".to_string()
    } else if name.contains("Standard") {
        "Standard".to_string()
    } else if name.contains("Custom") {
        "Custom".to_string()
    } else {
        "Standard".to_string()
    }
}

/// Extract display name from Google voice name (remove language prefix)
pub fn extract_display_name(name: &str) -> String {
    // Google voice names are like "en-US-Wavenet-A", extract the meaningful part
    let parts: Vec<&str> = name.split('-').collect();
    if parts.len() >= 3 {
        parts[2..].join("-")
    } else {
        name.to_string()
    }
}

/// Generate a descriptive text for the voice
pub fn generate_voice_description(voice: &GoogleVoice) -> String {
    let voice_type = extract_voice_type_from_name(&voice.name);
    let gender = match voice.ssml_gender {
        SsmlVoiceGender::Male => "male",
        SsmlVoiceGender::Female => "female",
        SsmlVoiceGender::Neutral => "neutral",
        SsmlVoiceGender::SsmlVoiceGenderUnspecified => "unspecified gender",
    };
    
    let languages = voice.language_codes.join(", ");
    let sample_rate = voice.natural_sample_rate_hertz;
    
    format!(
        "Google Cloud {} voice with {} gender, supporting languages: {}. Natural sample rate: {} Hz.",
        voice_type, gender, languages, sample_rate
    )
}

/// Infer use cases from voice name and characteristics
fn infer_use_cases_from_voice_name(name: &str) -> Vec<String> {
    let mut use_cases = Vec::new();
    let name_lower = name.to_lowercase();
    
    if name_lower.contains("wavenet") {
        use_cases.extend_from_slice(&[
            "high-quality speech synthesis".to_string(),
            "audiobooks".to_string(),
            "voice assistants".to_string()
        ]);
    } else if name_lower.contains("neural2") {
        use_cases.extend_from_slice(&[
            "conversational AI".to_string(),
            "customer service".to_string(),
            "interactive voice response".to_string()
        ]);
    } else if name_lower.contains("journey") {
        use_cases.extend_from_slice(&[
            "conversational AI".to_string(),
            "dynamic responses".to_string(),
            "context-aware synthesis".to_string()
        ]);
    } else if name_lower.contains("polyglot") {
        use_cases.extend_from_slice(&[
            "multilingual applications".to_string(),
            "global content".to_string(),
            "cross-language synthesis".to_string()
        ]);
    } else if name_lower.contains("studio") {
        use_cases.extend_from_slice(&[
            "professional audio production".to_string(),
            "media content".to_string(),
            "high-fidelity synthesis".to_string()
        ]);
    } else {
        use_cases.extend_from_slice(&[
            "general purpose".to_string(),
            "text-to-speech".to_string()
        ]);
    }
    
    use_cases.sort();
    use_cases.dedup();
    use_cases
}

/// Normalize language code to standard format
fn normalize_language_code(code: &str) -> String {
    // Google uses standard BCP-47 codes, so minimal normalization needed
    code.to_string()
}

pub fn synthesis_options_to_tts_request(
    input: &TextInput,
    voice_name: &str,
    language_code: &str,
    options: Option<SynthesisOptions>,
) -> (SynthesizeSpeechRequest, Option<BatchSynthesisParams>) {
    let default_request = SynthesizeSpeechRequest {
        input: SynthesisInput {
            text: None,
            ssml: None,
            multi_speaker_markup: None,
            custom_pronunciations: None,
        },
        voice: VoiceSelectionParams {
            language_code: language_code.to_string(),
            name: Some(voice_name.to_string()),
            ssml_gender: None,
            custom_voice: None,
            voice_clone: None,
        },
        audio_config: AudioConfig {
            audio_encoding: AudioEncoding::Mp3,
            speaking_rate: None,
            pitch: None,
            volume_gain_db: None,
            sample_rate_hertz: None,
            effects_profile_id: None,
        },
        advanced_voice_options: None,
    };

    let default_params = BatchSynthesisParams {
        max_chunk_size: Some(5000), // Google's limit is 5000 bytes for text input
        chunk_overlap: Some(100),
    };

    if let Some(opts) = options {
        let mut request = default_request;
        let params = default_params;

        // Set input based on text input type
        match input.text_type {
            golem_tts::golem::tts::types::TextType::Plain => {
                request.input.text = Some(input.content.clone());
            },
            golem_tts::golem::tts::types::TextType::Ssml => {
                request.input.ssml = Some(input.content.clone());
            },
        }

        // Map audio config
        if let Some(audio_config) = opts.audio_config {
            request.audio_config.audio_encoding = audio_format_to_encoding(audio_config.format);
            if let Some(sample_rate) = audio_config.sample_rate {
                request.audio_config.sample_rate_hertz = Some(sample_rate as i32);
            }
        }

        // Map voice settings
        if let Some(voice_settings) = opts.voice_settings {
            if let Some(speed) = voice_settings.speed {
                request.audio_config.speaking_rate = Some(speed as f64);
            }
            if let Some(pitch) = voice_settings.pitch {
                request.audio_config.pitch = Some(pitch as f64);
            }
            if let Some(volume) = voice_settings.volume {
                request.audio_config.volume_gain_db = Some(volume as f64);
            }
        }

        (request, Some(params))
    } else {
        let mut request = default_request;
        
        // Set input based on text input type
        match input.text_type {
            golem_tts::golem::tts::types::TextType::Plain => {
                request.input.text = Some(input.content.clone());
            },
            golem_tts::golem::tts::types::TextType::Ssml => {
                request.input.ssml = Some(input.content.clone());
            },
        }
        
        (request, Some(default_params))
    }
}

#[allow(dead_code)]
pub fn voice_settings_to_audio_config(settings: VoiceSettings) -> AudioConfig {
    AudioConfig {
        audio_encoding: AudioEncoding::Mp3, // Default
        speaking_rate: settings.speed.map(|s| s as f64),
        pitch: settings.pitch.map(|p| p as f64),
        volume_gain_db: settings.volume.map(|v| v as f64),
        sample_rate_hertz: None,
        effects_profile_id: None,
    }
}

pub fn audio_format_to_encoding(format: AudioFormat) -> AudioEncoding {
    match format {
        AudioFormat::Mp3 => AudioEncoding::Mp3,
        AudioFormat::Wav => AudioEncoding::Linear16,
        AudioFormat::OggOpus => AudioEncoding::OggOpus,
        AudioFormat::Pcm => AudioEncoding::Pcm,
        AudioFormat::Mulaw => AudioEncoding::Mulaw,
        AudioFormat::Alaw => AudioEncoding::Alaw,
        _ => AudioEncoding::Mp3, // Default fallback
    }
}

#[allow(dead_code)]
pub fn encoding_to_audio_format(encoding: AudioEncoding) -> AudioFormat {
    match encoding {
        AudioEncoding::Mp3 => AudioFormat::Mp3,
        AudioEncoding::Linear16 => AudioFormat::Wav,
        AudioEncoding::OggOpus => AudioFormat::OggOpus,
        AudioEncoding::Pcm => AudioFormat::Pcm,
        AudioEncoding::Mulaw => AudioFormat::Mulaw,
        AudioEncoding::Alaw => AudioFormat::Alaw,
        AudioEncoding::AudioEncodingUnspecified => AudioFormat::Mp3,
    }
}

pub fn audio_data_to_synthesis_result(
    audio_data: Vec<u8>,
    text: &str,
    encoding: &AudioEncoding,
    sample_rate: u32,
) -> SynthesisResult {
    let audio_size = audio_data.len() as u32;
    let duration = estimate_audio_duration(&audio_data, sample_rate, encoding);
    
    // Estimate word count for metadata
    let word_count = text.split_whitespace().count() as u32;
    
    SynthesisResult {
        audio_data,
        metadata: SynthesisMetadata {
            duration_seconds: duration,
            audio_size_bytes: audio_size,
            word_count,
            character_count: text.len() as u32,
            request_id: format!("google-{}", uuid::Uuid::new_v4()),
            provider_info: Some("Google Cloud TTS".to_string()),
        },
    }
}

pub fn create_validation_result(is_valid: bool, message: Option<String>) -> ValidationResult {
    ValidationResult {
        is_valid,
        character_count: 0,
        estimated_duration: None,
        warnings: vec![],
        errors: if let Some(msg) = message { vec![msg] } else { vec![] },
    }
}

pub fn google_voices_to_language_info(voices: Vec<GoogleVoice>) -> Vec<LanguageInfo> {
    use std::collections::HashMap;
    
    let mut language_map = HashMap::new();
    
    // Count voices per language
    for voice in voices {
        for lang_code in voice.language_codes {
            let entry = language_map.entry(lang_code.clone()).or_insert_with(|| {
                LanguageInfo {
                    code: lang_code.clone(),
                    name: get_language_name(&lang_code),
                    native_name: get_native_language_name(&lang_code),
                    voice_count: 0,
                }
            });
            entry.voice_count += 1;
        }
    }
    
    // If no voices found, provide default comprehensive list
    if language_map.is_empty() {
        return get_default_google_language_info();
    }
    
    let mut languages: Vec<LanguageInfo> = language_map.into_values().collect();
    languages.sort_by(|a, b| b.voice_count.cmp(&a.voice_count));
    languages
}

fn get_language_name(code: &str) -> String {
    match code {
        "en-US" => "English (United States)".to_string(),
        "en-GB" => "English (United Kingdom)".to_string(),
        "en-AU" => "English (Australia)".to_string(),
        "en-IN" => "English (India)".to_string(),
        "es-ES" => "Spanish (Spain)".to_string(),
        "es-US" => "Spanish (United States)".to_string(),
        "fr-FR" => "French (France)".to_string(),
        "fr-CA" => "French (Canada)".to_string(),
        "de-DE" => "German (Germany)".to_string(),
        "it-IT" => "Italian (Italy)".to_string(),
        "pt-BR" => "Portuguese (Brazil)".to_string(),
        "pt-PT" => "Portuguese (Portugal)".to_string(),
        "ja-JP" => "Japanese (Japan)".to_string(),
        "ko-KR" => "Korean (South Korea)".to_string(),
        "zh-CN" => "Chinese (Simplified)".to_string(),
        "zh-TW" => "Chinese (Traditional)".to_string(),
        "hi-IN" => "Hindi (India)".to_string(),
        "ar-XA" => "Arabic".to_string(),
        "ru-RU" => "Russian (Russia)".to_string(),
        "pl-PL" => "Polish (Poland)".to_string(),
        "tr-TR" => "Turkish (Turkey)".to_string(),
        "nl-NL" => "Dutch (Netherlands)".to_string(),
        "sv-SE" => "Swedish (Sweden)".to_string(),
        "da-DK" => "Danish (Denmark)".to_string(),
        "no-NO" => "Norwegian (Norway)".to_string(),
        "fi-FI" => "Finnish (Finland)".to_string(),
        _ => {
            // Extract language part and try to map it
            let lang_part = code.split('-').next().unwrap_or(code);
            match lang_part {
                "en" => "English".to_string(),
                "es" => "Spanish".to_string(),
                "fr" => "French".to_string(),
                "de" => "German".to_string(),
                "it" => "Italian".to_string(),
                "pt" => "Portuguese".to_string(),
                "ja" => "Japanese".to_string(),
                "ko" => "Korean".to_string(),
                "zh" => "Chinese".to_string(),
                "hi" => "Hindi".to_string(),
                "ar" => "Arabic".to_string(),
                "ru" => "Russian".to_string(),
                _ => code.to_string(),
            }
        }
    }
}

fn get_native_language_name(code: &str) -> String {
    match code {
        "en-US" | "en-GB" | "en-AU" | "en-IN" | "en" => "English".to_string(),
        "es-ES" | "es-US" | "es" => "Español".to_string(),
        "fr-FR" | "fr-CA" | "fr" => "Français".to_string(),
        "de-DE" | "de" => "Deutsch".to_string(),
        "it-IT" | "it" => "Italiano".to_string(),
        "pt-BR" | "pt-PT" | "pt" => "Português".to_string(),
        "ja-JP" | "ja" => "日本語".to_string(),
        "ko-KR" | "ko" => "한국어".to_string(),
        "zh-CN" | "zh-TW" | "zh" => "中文".to_string(),
        "hi-IN" | "hi" => "हिन्दी".to_string(),
        "ar-XA" | "ar" => "العربية".to_string(),
        "ru-RU" | "ru" => "Русский".to_string(),
        "pl-PL" | "pl" => "Polski".to_string(),
        "tr-TR" | "tr" => "Türkçe".to_string(),
        "nl-NL" | "nl" => "Nederlands".to_string(),
        "sv-SE" | "sv" => "Svenska".to_string(),
        "da-DK" | "da" => "Dansk".to_string(),
        "no-NO" | "no" => "Norsk".to_string(),
        "fi-FI" | "fi" => "Suomi".to_string(),
        _ => get_language_name(code),
    }
}

fn get_default_google_language_info() -> Vec<LanguageInfo> {
    vec![
        LanguageInfo {
            code: "en-US".to_string(),
            name: "English (United States)".to_string(),
            native_name: "English".to_string(),
            voice_count: 20,
        },
        LanguageInfo {
            code: "es-ES".to_string(),
            name: "Spanish (Spain)".to_string(),
            native_name: "Español".to_string(),
            voice_count: 10,
        },
        LanguageInfo {
            code: "fr-FR".to_string(),
            name: "French (France)".to_string(),
            native_name: "Français".to_string(),
            voice_count: 8,
        },
        LanguageInfo {
            code: "de-DE".to_string(),
            name: "German (Germany)".to_string(),
            native_name: "Deutsch".to_string(),
            voice_count: 8,
        },
        LanguageInfo {
            code: "ja-JP".to_string(),
            name: "Japanese (Japan)".to_string(),
            native_name: "日本語".to_string(),
            voice_count: 6,
        },
        LanguageInfo {
            code: "zh-CN".to_string(),
            name: "Chinese (Simplified)".to_string(),
            native_name: "中文".to_string(),
            voice_count: 6,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_gender_conversion() {
        assert_eq!(ssml_gender_to_voice_gender(&SsmlVoiceGender::Male), VoiceGender::Male);
        assert_eq!(ssml_gender_to_voice_gender(&SsmlVoiceGender::Female), VoiceGender::Female);
        assert_eq!(ssml_gender_to_voice_gender(&SsmlVoiceGender::Neutral), VoiceGender::Neutral);
    }

    #[test]
    fn test_audio_format_conversion() {
        assert_eq!(audio_format_to_encoding(AudioFormat::Mp3), AudioEncoding::Mp3);
        assert_eq!(audio_format_to_encoding(AudioFormat::Wav), AudioEncoding::Linear16);
        assert_eq!(encoding_to_audio_format(AudioEncoding::Mp3), AudioFormat::Mp3);
    }

    #[test]
    fn test_quality_inference() {
        let wavenet_voice = GoogleVoice {
            name: "en-US-Wavenet-A".to_string(),
            language_codes: vec!["en-US".to_string()],
            ssml_gender: SsmlVoiceGender::Female,
            natural_sample_rate_hertz: 24000,
        };
        assert_eq!(infer_quality_from_voice(&wavenet_voice), VoiceQuality::Premium);

        let standard_voice = GoogleVoice {
            name: "en-US-Standard-A".to_string(),
            language_codes: vec!["en-US".to_string()],
            ssml_gender: SsmlVoiceGender::Male,
            natural_sample_rate_hertz: 22050,
        };
        assert_eq!(infer_quality_from_voice(&standard_voice), VoiceQuality::Standard);
    }

    #[test]
    fn test_display_name_extraction() {
        assert_eq!(extract_display_name("en-US-Wavenet-A"), "Wavenet-A");
        assert_eq!(extract_display_name("ja-JP-Neural2-B"), "Neural2-B");
        assert_eq!(extract_display_name("SimpleVoice"), "SimpleVoice");
    }
}
