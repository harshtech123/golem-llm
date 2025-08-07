use crate::client::{
    CreateVoiceRequest, ListVoicesParams, Model, TextToSpeechParams, TextToSpeechRequest,
    Voice as ElevenLabsVoice, VoiceSettings as ElevenLabsVoiceSettings,
};
use golem_tts::golem::tts::advanced::{AgeCategory, AudioSample, VoiceDesignParams};
use golem_tts::golem::tts::synthesis::{SynthesisOptions, ValidationResult};
use golem_tts::golem::tts::types::{
    AudioFormat, SynthesisMetadata, SynthesisResult, VoiceGender, VoiceQuality, VoiceSettings,
};
use golem_tts::golem::tts::voices::{LanguageInfo, VoiceFilter, VoiceInfo};

/// Estimate audio duration in seconds based on audio data size
/// This is a rough estimation for MP3 data at given sample rate
pub fn estimate_audio_duration(audio_data: &[u8], _sample_rate: u32) -> f32 {
    if audio_data.is_empty() {
        return 0.0;
    }

    // For MP3 files, use average bitrate estimation
    // ElevenLabs typically uses ~64-128 kbps for streaming
    let estimated_bitrate_bps = 96000; // 96 kbps average
    let bytes_per_second = estimated_bitrate_bps / 8;

    (audio_data.len() as f32) / (bytes_per_second as f32)
}

pub fn voice_filter_to_list_params(filter: Option<VoiceFilter>) -> Option<ListVoicesParams> {
    filter.map(|f| ListVoicesParams {
        next_page_token: None,
        page_size: None, // Remove limit field as it doesn't exist in VoiceFilter
        search: f.search_query,
        sort: None,
        sort_direction: None,
        voice_type: match f.gender {
            Some(VoiceGender::Male) => Some("male".to_string()),
            Some(VoiceGender::Female) => Some("female".to_string()),
            Some(VoiceGender::Neutral) => Some("neutral".to_string()),
            None => None,
        },
        category: match f.quality {
            Some(VoiceQuality::Standard) => Some("standard".to_string()),
            Some(VoiceQuality::Premium) => Some("premium".to_string()),
            Some(VoiceQuality::Studio) => Some("studio".to_string()),
            Some(VoiceQuality::Neural) => Some("neural".to_string()),
            None => None,
        },
        include_total_count: Some(true),
    })
}

pub fn elevenlabs_voice_to_voice_info(voice: ElevenLabsVoice) -> VoiceInfo {
    let gender = infer_gender_from_name(&voice.name).unwrap_or(VoiceGender::Neutral);
    let quality = infer_quality_from_category(&voice.category).unwrap_or(VoiceQuality::Standard);

    // Enhanced language detection
    let language = detect_voice_language(&voice);
    let additional_languages = detect_additional_languages(&voice);
    let use_cases = infer_use_cases_from_voice(&voice);
    let is_custom = voice.category.to_lowercase() == "cloned";
    let is_cloned = voice.category.to_lowercase() == "cloned";
    let preview_url = voice.preview_url.clone();
    let description = voice.description.clone();

    VoiceInfo {
        id: voice.voice_id,
        name: voice.name,
        language,
        additional_languages,
        gender,
        quality,
        description,
        provider: "elevenlabs".to_string(),
        sample_rate: 22050, // Default ElevenLabs sample rate
        is_custom,
        is_cloned,
        preview_url,
        use_cases,
    }
}

/// Detect primary language for a voice using multiple heuristics
fn detect_voice_language(voice: &ElevenLabsVoice) -> String {
    // Priority 1: Check voice labels/metadata for language hints
    if let Some(labels) = &voice.labels {
        if let Some(language) = extract_language_from_labels(labels) {
            return language;
        }
    }

    // Priority 2: Infer from voice name patterns
    if let Some(language) = infer_language_from_name(&voice.name) {
        return language;
    }

    // Priority 3: Infer from description
    if let Some(description) = &voice.description {
        if let Some(language) = infer_language_from_description(description) {
            return language;
        }
    }

    // Priority 4: Category-based inference
    match voice.category.to_lowercase().as_str() {
        "multilingual" => "en".to_string(), // Default for multilingual
        _ => "en".to_string(),              // Default fallback
    }
}

/// Detect additional supported languages for multilingual voices
fn detect_additional_languages(voice: &ElevenLabsVoice) -> Vec<String> {
    let mut languages = Vec::new();

    // Check if voice is multilingual
    if voice.category.to_lowercase().contains("multilingual") {
        // Common ElevenLabs multilingual support
        languages.extend_from_slice(&[
            "es".to_string(),
            "fr".to_string(),
            "de".to_string(),
            "it".to_string(),
            "pt".to_string(),
            "pl".to_string(),
            "zh".to_string(),
            "ja".to_string(),
            "hi".to_string(),
        ]);
    }

    // Check voice name/description for language mentions
    if let Some(description) = &voice.description {
        languages.extend(extract_mentioned_languages(description));
    }

    // Remove duplicates and primary language
    let primary = detect_voice_language(voice);
    languages.retain(|lang| lang != &primary);
    languages.sort();
    languages.dedup();

    languages
}

/// Extract language from ElevenLabs voice labels/metadata
fn extract_language_from_labels(labels: &serde_json::Value) -> Option<String> {
    if let Some(obj) = labels.as_object() {
        // Check common language fields
        if let Some(lang) = obj.get("language").and_then(|v| v.as_str()) {
            return Some(normalize_language_code(lang));
        }
        if let Some(lang) = obj.get("accent").and_then(|v| v.as_str()) {
            return Some(accent_to_language_code(lang));
        }
        if let Some(lang) = obj.get("origin").and_then(|v| v.as_str()) {
            return Some(origin_to_language_code(lang));
        }
    }
    None
}

/// Infer language from voice name patterns
fn infer_language_from_name(name: &str) -> Option<String> {
    let name_lower = name.to_lowercase();

    // Direct language mentions
    if name_lower.contains("english")
        || name_lower.contains("american")
        || name_lower.contains("british")
    {
        return Some("en".to_string());
    }
    if name_lower.contains("spanish") || name_lower.contains("español") {
        return Some("es".to_string());
    }
    if name_lower.contains("french") || name_lower.contains("français") {
        return Some("fr".to_string());
    }
    if name_lower.contains("german") || name_lower.contains("deutsch") {
        return Some("de".to_string());
    }
    if name_lower.contains("italian") || name_lower.contains("italiano") {
        return Some("it".to_string());
    }
    if name_lower.contains("portuguese") || name_lower.contains("português") {
        return Some("pt".to_string());
    }
    if name_lower.contains("polish") || name_lower.contains("polski") {
        return Some("pl".to_string());
    }
    if name_lower.contains("chinese") || name_lower.contains("mandarin") {
        return Some("zh".to_string());
    }
    if name_lower.contains("japanese") || name_lower.contains("日本") {
        return Some("ja".to_string());
    }
    if name_lower.contains("hindi") || name_lower.contains("हिन्दी") {
        return Some("hi".to_string());
    }

    None
}

/// Infer language from voice description
fn infer_language_from_description(description: &str) -> Option<String> {
    let desc_lower = description.to_lowercase();

    // Language pattern matching in descriptions
    if desc_lower.contains("english")
        || desc_lower.contains("american accent")
        || desc_lower.contains("british accent")
    {
        return Some("en".to_string());
    }
    if desc_lower.contains("spanish")
        || desc_lower.contains("latino")
        || desc_lower.contains("hispanic")
    {
        return Some("es".to_string());
    }
    if desc_lower.contains("french") || desc_lower.contains("parisian") {
        return Some("fr".to_string());
    }
    if desc_lower.contains("german") || desc_lower.contains("bavarian") {
        return Some("de".to_string());
    }

    None
}

/// Extract mentioned languages from text
fn extract_mentioned_languages(text: &str) -> Vec<String> {
    let mut languages = Vec::new();
    let text_lower = text.to_lowercase();

    let language_patterns = [
        ("english", "en"),
        ("spanish", "es"),
        ("french", "fr"),
        ("german", "de"),
        ("italian", "it"),
        ("portuguese", "pt"),
        ("polish", "pl"),
        ("chinese", "zh"),
        ("japanese", "ja"),
        ("hindi", "hi"),
        ("arabic", "ar"),
        ("russian", "ru"),
    ];

    for (pattern, code) in language_patterns {
        if text_lower.contains(pattern) {
            languages.push(code.to_string());
        }
    }

    languages
}

/// Infer use cases from voice characteristics
fn infer_use_cases_from_voice(voice: &ElevenLabsVoice) -> Vec<String> {
    let mut use_cases = Vec::new();

    // Infer from category
    match voice.category.to_lowercase().as_str() {
        "narration" => use_cases.push("audiobooks".to_string()),
        "conversational" => {
            use_cases.extend_from_slice(&["chatbots".to_string(), "assistant".to_string()])
        }
        "news" => use_cases.push("news-reading".to_string()),
        "storytelling" => {
            use_cases.extend_from_slice(&["audiobooks".to_string(), "podcasts".to_string()])
        }
        _ => {}
    }

    // Infer from name/description
    if let Some(description) = &voice.description {
        let desc_lower = description.to_lowercase();

        if desc_lower.contains("narrator") || desc_lower.contains("storytelling") {
            use_cases.push("audiobooks".to_string());
        }
        if desc_lower.contains("news") || desc_lower.contains("anchor") {
            use_cases.push("news-reading".to_string());
        }
        if desc_lower.contains("assistant") || desc_lower.contains("helpful") {
            use_cases.push("assistant".to_string());
        }
        if desc_lower.contains("podcast") {
            use_cases.push("podcasts".to_string());
        }
        if desc_lower.contains("commercial") || desc_lower.contains("advertising") {
            use_cases.push("commercials".to_string());
        }
    }

    // Default use cases if none inferred
    if use_cases.is_empty() {
        use_cases.push("general".to_string());
    }

    use_cases.sort();
    use_cases.dedup();
    use_cases
}

// Helper functions for language normalization
fn normalize_language_code(code: &str) -> String {
    match code.to_lowercase().as_str() {
        "en-us" | "en-gb" | "english" => "en".to_string(),
        "es-es" | "es-mx" | "spanish" => "es".to_string(),
        "fr-fr" | "french" => "fr".to_string(),
        "de-de" | "german" => "de".to_string(),
        "it-it" | "italian" => "it".to_string(),
        "pt-pt" | "pt-br" | "portuguese" => "pt".to_string(),
        _ => code.to_lowercase(),
    }
}

fn accent_to_language_code(accent: &str) -> String {
    match accent.to_lowercase().as_str() {
        "american" | "british" | "australian" => "en".to_string(),
        "mexican" | "argentinian" | "colombian" => "es".to_string(),
        "parisian" | "canadian" => "fr".to_string(),
        "bavarian" | "austrian" => "de".to_string(),
        _ => "en".to_string(), // Default fallback
    }
}

fn origin_to_language_code(origin: &str) -> String {
    match origin.to_lowercase().as_str() {
        "usa" | "uk" | "canada" | "australia" => "en".to_string(),
        "spain" | "mexico" | "argentina" => "es".to_string(),
        "france" => "fr".to_string(),
        "germany" | "austria" => "de".to_string(),
        "italy" => "it".to_string(),
        "portugal" | "brazil" => "pt".to_string(),
        "poland" => "pl".to_string(),
        "china" => "zh".to_string(),
        "japan" => "ja".to_string(),
        "india" => "hi".to_string(),
        _ => "en".to_string(), // Default fallback
    }
}

pub fn synthesis_options_to_tts_request(
    options: Option<SynthesisOptions>,
    model_version: &str,
) -> (TextToSpeechRequest, Option<TextToSpeechParams>) {
    let default_request = TextToSpeechRequest {
        text: String::new(),
        model_id: Some(model_version.to_string()),
        language_code: None,
        voice_settings: None,
        pronunciation_dictionary_locators: None,
        seed: None,
        previous_text: None,
        next_text: None,
        previous_request_ids: None,
        next_request_ids: None,
        apply_text_normalization: Some("auto".to_string()),
        apply_language_text_normalization: Some(true),
        use_pvc_as_ivc: Some(false),
    };

    let default_params = TextToSpeechParams {
        enable_logging: Some(false),
        optimize_streaming_latency: None,
        output_format: Some("mp3_22050_32".to_string()),
    };

    if let Some(opts) = options {
        let mut request = default_request;
        let mut params = default_params;

        // Map voice settings
        if let Some(voice_settings) = opts.voice_settings {
            request.voice_settings = Some(voice_settings_to_elevenlabs(voice_settings));
        }

        // Map audio config
        if let Some(audio_config) = opts.audio_config {
            params.output_format = Some(audio_format_to_string(audio_config.format));
        }

        // Map seed
        if let Some(seed) = opts.seed {
            request.seed = Some(seed);
        }

        // Map model version
        if let Some(model_version) = opts.model_version {
            request.model_id = Some(model_version);
        }

        // Map synthesis context
        if let Some(context) = opts.context {
            request.previous_text = context.previous_text;
            request.next_text = context.next_text;
        }

        (request, Some(params))
    } else {
        (default_request, Some(default_params))
    }
}

pub fn voice_settings_to_elevenlabs(settings: VoiceSettings) -> ElevenLabsVoiceSettings {
    ElevenLabsVoiceSettings {
        stability: settings.stability,
        similarity_boost: settings.similarity,
        style: settings.style,
        use_speaker_boost: None, // Not directly available in VoiceSettings
        speed: settings.speed,
    }
}

pub fn audio_format_to_string(format: AudioFormat) -> String {
    match format {
        AudioFormat::Mp3 => "mp3_22050_32".to_string(),
        AudioFormat::Wav => "pcm_22050".to_string(),
        AudioFormat::Pcm => "pcm_22050".to_string(),
        AudioFormat::OggOpus => "mp3_22050_32".to_string(), // Fallback to MP3
        AudioFormat::Aac => "mp3_22050_32".to_string(),     // Fallback to MP3
        AudioFormat::Flac => "pcm_22050".to_string(), // ElevenLabs doesn't support FLAC directly
        AudioFormat::Mulaw => "pcm_22050".to_string(), // Fallback to PCM
        AudioFormat::Alaw => "pcm_22050".to_string(), // Fallback to PCM
    }
}

pub fn audio_data_to_synthesis_result(audio_data: Vec<u8>, text: &str) -> SynthesisResult {
    let audio_size = audio_data.len() as u32;

    // Estimate duration based on average speaking rate (150 words per minute)
    let word_count = text.split_whitespace().count();
    let estimated_duration = if word_count > 0 {
        (word_count as f32 / 150.0) * 60.0 // words per minute to seconds
    } else {
        0.0
    };

    SynthesisResult {
        audio_data,
        metadata: SynthesisMetadata {
            duration_seconds: estimated_duration,
            character_count: text.len() as u32,
            word_count: word_count as u32,
            audio_size_bytes: audio_size,
            request_id: format!("elevenlabs-{}", uuid::Uuid::new_v4()),
            provider_info: Some("ElevenLabs".to_string()),
        },
    }
}

pub fn create_voice_request_from_samples(
    name: String,
    description: Option<String>,
    samples: Vec<AudioSample>,
) -> CreateVoiceRequest {
    use crate::client::AudioFile;

    let files = samples
        .into_iter()
        .map(|sample| AudioFile { data: sample.data })
        .collect();

    CreateVoiceRequest {
        name,
        description,
        files,
        labels: None,
    }
}

pub fn voice_design_params_to_create_request(params: VoiceDesignParams) -> CreateVoiceRequest {
    let gender_str = match params.gender {
        VoiceGender::Male => "male",
        VoiceGender::Female => "female",
        VoiceGender::Neutral => "neutral",
    };
    let age_str = match params.age_category {
        AgeCategory::Child => "child",
        AgeCategory::YoungAdult => "young_adult",
        AgeCategory::MiddleAged => "middle_aged",
        AgeCategory::Elderly => "elderly",
    };

    CreateVoiceRequest {
        name: format!("Generated_{}_{}_{}", gender_str, age_str, params.accent),
        description: Some(format!(
            "Generated voice with traits: {:?}",
            params.personality_traits
        )),
        files: vec![], // Voice design would generate files programmatically
        labels: None,
    }
}

// Helper functions
pub fn infer_gender_from_name(name: &str) -> Option<VoiceGender> {
    let name_lower = name.to_lowercase();
    if name_lower.contains("female") || name_lower.contains("woman") {
        Some(VoiceGender::Female)
    } else if name_lower.contains("male") || name_lower.contains("man") {
        Some(VoiceGender::Male)
    } else {
        None // Default to None when we can't infer
    }
}

pub fn infer_quality_from_category(category: &str) -> Option<VoiceQuality> {
    match category.to_lowercase().as_str() {
        "premade" => Some(VoiceQuality::Standard),
        "cloned" => Some(VoiceQuality::Premium),
        "professional" => Some(VoiceQuality::Studio),
        _ => Some(VoiceQuality::Standard),
    }
}

pub fn create_validation_result(is_valid: bool, message: Option<String>) -> ValidationResult {
    ValidationResult {
        is_valid,
        character_count: 0, // Would need actual text to determine
        estimated_duration: None,
        warnings: message.map(|m| vec![m]).unwrap_or_default(),
        errors: vec![],
    }
}

pub fn models_to_language_info(models: Vec<Model>) -> Vec<LanguageInfo> {
    // Extract languages from ElevenLabs models for accurate language support
    let mut language_map = std::collections::HashMap::new();

    // Parse languages from models
    for model in models {
        for lang in model.languages {
            let entry = language_map
                .entry(lang.language_id.clone())
                .or_insert_with(|| LanguageInfo {
                    code: lang.language_id.clone(),
                    name: lang.name.clone(),
                    native_name: get_native_language_name(&lang.language_id),
                    voice_count: 0,
                });
            // Increment voice count based on model support
            entry.voice_count += 1;
        }
    }

    // If models don't provide languages, fall back to comprehensive list
    if language_map.is_empty() {
        return get_default_language_info();
    }

    let mut languages: Vec<LanguageInfo> = language_map.into_values().collect();
    languages.sort_by(|a, b| b.voice_count.cmp(&a.voice_count)); // Sort by voice count descending
    languages
}

fn get_native_language_name(language_code: &str) -> String {
    match language_code {
        "en" | "en-US" | "en-GB" => "English".to_string(),
        "es" | "es-ES" | "es-MX" => "Español".to_string(),
        "fr" | "fr-FR" => "Français".to_string(),
        "de" | "de-DE" => "Deutsch".to_string(),
        "it" | "it-IT" => "Italiano".to_string(),
        "pt" | "pt-PT" | "pt-BR" => "Português".to_string(),
        "pl" | "pl-PL" => "Polski".to_string(),
        "hi" | "hi-IN" => "हिन्दी".to_string(),
        "ar" | "ar-SA" => "العربية".to_string(),
        "zh" | "zh-CN" => "中文".to_string(),
        "ja" | "ja-JP" => "日本語".to_string(),
        "ko" | "ko-KR" => "한국어".to_string(),
        "ru" | "ru-RU" => "Русский".to_string(),
        "nl" | "nl-NL" => "Nederlands".to_string(),
        "sv" | "sv-SE" => "Svenska".to_string(),
        "no" | "nb-NO" => "Norsk".to_string(),
        "da" | "da-DK" => "Dansk".to_string(),
        "fi" | "fi-FI" => "Suomi".to_string(),
        "tr" | "tr-TR" => "Türkçe".to_string(),
        "uk" | "uk-UA" => "Українська".to_string(),
        "cs" | "cs-CZ" => "Čeština".to_string(),
        "hu" | "hu-HU" => "Magyar".to_string(),
        "ro" | "ro-RO" => "Română".to_string(),
        "sk" | "sk-SK" => "Slovenčina".to_string(),
        "bg" | "bg-BG" => "Български".to_string(),
        "hr" | "hr-HR" => "Hrvatski".to_string(),
        "et" | "et-EE" => "Eesti".to_string(),
        "lv" | "lv-LV" => "Latviešu".to_string(),
        "lt" | "lt-LT" => "Lietuvių".to_string(),
        "sl" | "sl-SI" => "Slovenščina".to_string(),
        _ => language_code.to_string(), // Fallback to code if unknown
    }
}

fn get_default_language_info() -> Vec<LanguageInfo> {
    // Comprehensive fallback list based on ElevenLabs typical support
    vec![
        LanguageInfo {
            code: "en".to_string(),
            name: "English".to_string(),
            native_name: "English".to_string(),
            voice_count: 50,
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
            voice_count: 10,
        },
        LanguageInfo {
            code: "it".to_string(),
            name: "Italian".to_string(),
            native_name: "Italiano".to_string(),
            voice_count: 8,
        },
        LanguageInfo {
            code: "pt".to_string(),
            name: "Portuguese".to_string(),
            native_name: "Português".to_string(),
            voice_count: 6,
        },
        LanguageInfo {
            code: "pl".to_string(),
            name: "Polish".to_string(),
            native_name: "Polski".to_string(),
            voice_count: 5,
        },
        LanguageInfo {
            code: "hi".to_string(),
            name: "Hindi".to_string(),
            native_name: "हिन्दी".to_string(),
            voice_count: 4,
        },
        LanguageInfo {
            code: "ar".to_string(),
            name: "Arabic".to_string(),
            native_name: "العربية".to_string(),
            voice_count: 3,
        },
        LanguageInfo {
            code: "zh".to_string(),
            name: "Chinese".to_string(),
            native_name: "中文".to_string(),
            voice_count: 2,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_filter_conversion() {
        let filter = VoiceFilter {
            gender: Some(VoiceGender::Female),
            quality: Some(VoiceQuality::Premium),
            language: Some("en".to_string()),
            supports_ssml: Some(true),
            provider: Some("ElevenLabs".to_string()),
            search_query: Some("narrator".to_string()),
        };

        let params = voice_filter_to_list_params(Some(filter)).unwrap();
        assert_eq!(params.voice_type, Some("female".to_string()));
        assert_eq!(params.category, Some("premium".to_string()));
        assert_eq!(params.search, Some("narrator".to_string()));
        assert_eq!(params.page_size, Some(10));
    }

    #[test]
    fn test_audio_format_conversion() {
        assert_eq!(audio_format_to_string(AudioFormat::Mp3), "mp3_22050_32");
        assert_eq!(audio_format_to_string(AudioFormat::Wav), "pcm_22050");
    }

    #[test]
    fn test_gender_inference() {
        assert_eq!(
            infer_gender_from_name("Sarah Female Voice"),
            Some(VoiceGender::Female)
        );
        assert_eq!(
            infer_gender_from_name("John Male Voice"),
            Some(VoiceGender::Male)
        );
        assert_eq!(infer_gender_from_name("Alex"), None);
    }

    #[test]
    fn test_quality_inference() {
        assert_eq!(
            infer_quality_from_category("premade"),
            Some(VoiceQuality::Standard)
        );
        assert_eq!(
            infer_quality_from_category("cloned"),
            Some(VoiceQuality::Premium)
        );
        assert_eq!(
            infer_quality_from_category("professional"),
            Some(VoiceQuality::Studio)
        );
    }
}
