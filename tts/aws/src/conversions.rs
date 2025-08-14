use crate::client::{
    DescribeVoicesParams, Engine, OutputFormat, SpeechMarkType, SynthesizeSpeechParams, TextType,
    Voice as AwsVoice,
};
use golem_tts::golem::tts::synthesis::{SynthesisOptions, ValidationResult};
use golem_tts::golem::tts::types::{
    AudioFormat, SynthesisMetadata, SynthesisResult, VoiceGender, VoiceQuality, TtsError, TextType as TtsTextType, VoiceSettings,
};
use golem_tts::golem::tts::voices::{LanguageInfo, VoiceFilter, VoiceInfo};
use log::trace;

pub fn validate_synthesis_input(
    input: &golem_tts::golem::tts::types::TextInput,
    options: Option<&SynthesisOptions>,
) -> Result<(), TtsError> {
    if input.content.trim().is_empty() {
        return Err(TtsError::InvalidText("Text content cannot be empty".to_string()));
    }

    if input.text_type == TtsTextType::Ssml {
        if let Err(msg) = validate_ssml_content(&input.content) {
            return Err(TtsError::InvalidSsml(msg));
        }
    }

    if let Some(ref language) = input.language {
        if !is_supported_language(language) {
            return Err(TtsError::UnsupportedLanguage(format!(
                "Language '{}' is not supported by AWS Polly", language
            )));
        }
    }

    if let Some(opts) = options {
        if let Some(ref voice_settings) = opts.voice_settings {
            validate_voice_settings(voice_settings)?;
        }
    }

    Ok(())
}

pub fn validate_ssml_content(content: &str) -> Result<(), String> {
    let mut tag_stack = Vec::new();
    let mut chars = content.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '<' {
            let mut tag = String::new();
            let mut is_closing = false;
            let mut is_self_closing = false;
            
            if chars.peek() == Some(&'/') {
                is_closing = true;
                chars.next(); 
            }

            let mut full_tag_content = String::new();
            while let Some(&ch) = chars.peek() {
                if ch == '>' {
                    break;
                }
                if ch == ' ' && tag.is_empty() {
                    tag = full_tag_content.clone();
                }
                full_tag_content.push(chars.next().unwrap());
            }

            if tag.is_empty() {
                tag = full_tag_content.clone();
            }

            if full_tag_content.ends_with('/') {
                is_self_closing = true;
                if tag.ends_with('/') {
                    tag = tag[..tag.len()-1].to_string();
                }
            }

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

pub fn is_supported_language(language: &str) -> bool {
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

pub fn validate_voice_settings(settings: &VoiceSettings) -> Result<(), TtsError> {
    if let Some(speed) = settings.speed {
        if speed < 0.25 || speed > 4.0 {
            return Err(TtsError::InvalidConfiguration(
                "Speed must be between 0.25 and 4.0".to_string()
            ));
        }
    }

    if let Some(pitch) = settings.pitch {
        if pitch < -10.0 || pitch > 10.0 {
            return Err(TtsError::InvalidConfiguration(
                "Pitch must be between -10.0 and +10.0".to_string()
            ));
        }
    }

    if let Some(volume) = settings.volume {
        if volume < -20.0 || volume > 20.0 {
            return Err(TtsError::InvalidConfiguration(
                "Volume must be between -20.0 and +20.0".to_string()
            ));
        }
    }

    if let Some(stability) = settings.stability {
        if stability < 0.0 || stability > 1.0 {
            return Err(TtsError::InvalidConfiguration(
                "Stability must be between 0.0 and 1.0".to_string()
            ));
        }
    }

    if let Some(similarity) = settings.similarity {
        if similarity < 0.0 || similarity > 1.0 {
            return Err(TtsError::InvalidConfiguration(
                "Similarity must be between 0.0 and 1.0".to_string()
            ));
        }
    }

    if let Some(style) = settings.style {
        if style < 0.0 || style > 1.0 {
            return Err(TtsError::InvalidConfiguration(
                "Style must be between 0.0 and 1.0".to_string()
            ));
        }
    }

    Ok(())
}

pub fn split_text_intelligently(text: &str, max_chunk_size: usize) -> Vec<String> {
    if text.len() <= max_chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    
    let mut sentences = Vec::new();
    let mut current_sentence = String::new();
    let mut chars = text.chars().peekable();
    
    while let Some(ch) = chars.next() {
        current_sentence.push(ch);
        
        if matches!(ch, '.' | '!' | '?') {
            if chars.peek().map(|c| c.is_whitespace()).unwrap_or(true) {
                sentences.push(current_sentence.trim().to_string());
                current_sentence.clear();
            }
        } else if ch == '\n' {
            let sentence = current_sentence.trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
                current_sentence.clear();
            }
        }
    }
    
    let remaining = current_sentence.trim();
    if !remaining.is_empty() {
        sentences.push(remaining.to_string());
    }

    for sentence in sentences {
        if sentence.trim().is_empty() {
            continue;
        }

        if current_chunk.len() + sentence.len() + 1 > max_chunk_size {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
            }

            if sentence.len() > max_chunk_size {
                let word_chunks = split_by_words(&sentence, max_chunk_size);
                chunks.extend(word_chunks);
            } else {
                current_chunk = sentence;
            }
        } else {
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(&sentence);
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    if chunks.is_empty() {
        chunks.push(text.to_string());
    }

    chunks
}

pub fn split_by_words(text: &str, max_chunk_size: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for word in words {
        if current_chunk.len() + word.len() + 1 > max_chunk_size {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
            }
            
            if word.len() > max_chunk_size {
                chunks.push(word[..max_chunk_size].to_string());
            } else {
                current_chunk = word.to_string();
            }
        } else {
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(word);
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

pub fn combine_audio_chunks(chunks: Vec<Vec<u8>>, format: &AudioFormat) -> Vec<u8> {
    if chunks.is_empty() {
        return Vec::new();
    }

    if chunks.len() == 1 {
        return chunks.into_iter().next().unwrap();
    }

    match format {
        AudioFormat::Pcm => {
            chunks.into_iter().flatten().collect()
        }
        AudioFormat::Mp3 => {           
            chunks.into_iter().flatten().collect()
        }
        AudioFormat::OggOpus => {
            chunks.into_iter().flatten().collect()
        }
        _ => {
            chunks.into_iter().flatten().collect()
        }
    }
}

pub fn estimate_audio_duration(audio_data: &[u8], sample_rate: u32, format: &AudioFormat) -> f32 {
    if audio_data.is_empty() {
        return 0.0;
    }

    match format {
        AudioFormat::Mp3 => {
            let estimated_bitrate_bps = 128000; 
            let bytes_per_second = estimated_bitrate_bps / 8;
            (audio_data.len() as f32) / (bytes_per_second as f32)
        }
        AudioFormat::Wav | AudioFormat::Pcm => {
            let channels = 1;
            let bytes_per_sample = 2; 
            let bytes_per_second = sample_rate * channels * bytes_per_sample;
            (audio_data.len() as f32) / (bytes_per_second as f32)
        }
        AudioFormat::OggOpus => {
            let estimated_bitrate_bps = 64000;
            let bytes_per_second = estimated_bitrate_bps / 8;
            (audio_data.len() as f32) / (bytes_per_second as f32)
        }
        _ => {
            let estimated_chars = audio_data.len() / 100; 
            (estimated_chars as f32 / 150.0) * 60.0
        }
    }
}

pub fn voice_filter_to_describe_params(
    filter: Option<VoiceFilter>,
) -> Option<DescribeVoicesParams> {
    filter.map(|f| DescribeVoicesParams {
        engine: None,
        language_code: f.language,
        include_additional_language_codes: Some(true),
        next_token: None,
    })
}

pub fn aws_voice_to_voice_info(voice: AwsVoice) -> VoiceInfo {
    trace!("Converting AWS voice: {} ({})", voice.name, voice.id);
    
    let gender = match voice.gender.to_lowercase().as_str() {
        "male" => VoiceGender::Male,
        "female" => VoiceGender::Female,
        _ => VoiceGender::Neutral,
    };

    let quality = if voice.supported_engines.contains(&"neural".to_string()) {
        VoiceQuality::Neural
    } else if voice.supported_engines.contains(&"generative".to_string()) {
        VoiceQuality::Studio
    } else {
        VoiceQuality::Standard
    };

    let additional_languages = voice.additional_language_codes.clone().unwrap_or_default();

    let use_cases = infer_use_cases_from_aws_voice(&voice);

    let sample_rate = 22050; 

    VoiceInfo {
        id: voice.id.clone(),
        name: voice.name.clone(),
        language: voice.language_code.clone(),
        additional_languages,
        gender,
        quality,
        description: Some(format!("{} voice from AWS Polly", voice.language_name)),
        provider: "AWS Polly".to_string(),
        sample_rate,
        is_custom: false, 
        is_cloned: false,
        preview_url: None,
        use_cases,
    }
}

fn infer_use_cases_from_aws_voice(voice: &AwsVoice) -> Vec<String> {
    let mut use_cases = Vec::new();

    use_cases.push("general".to_string());
    use_cases.push("content".to_string());

    if voice.supported_engines.contains(&"neural".to_string()) {
        use_cases.push("audiobooks".to_string());
        use_cases.push("news".to_string());
        use_cases.push("conversational".to_string());
    }

    if voice.supported_engines.contains(&"long-form".to_string()) {
        use_cases.push("long-form".to_string());
        use_cases.push("books".to_string());
    }

    if voice.supported_engines.contains(&"generative".to_string()) {
        use_cases.push("expressive".to_string());
        use_cases.push("creative".to_string());
    }

    match voice.language_code.as_str() {
        "en-US" | "en-GB" | "en-AU" => {
            use_cases.push("business".to_string());
            use_cases.push("education".to_string());
        }
        "es-ES" | "es-MX" | "es-US" => {
            use_cases.push("multilingual".to_string());
        }
        _ => {}
    }

    use_cases.sort();
    use_cases.dedup();
    use_cases
}

pub fn synthesis_options_to_polly_params(
    options: Option<SynthesisOptions>,
    voice_id: String,
    text: String,
) -> SynthesizeSpeechParams {
    let mut params = SynthesizeSpeechParams {
        engine: Some(Engine::Neural),
        language_code: None,
        lexicon_names: None,
        output_format: OutputFormat::Mp3,
        sample_rate: None,
        speech_mark_types: None,
        text,
        text_type: Some(TextType::Text),
        voice_id,
    };

    if let Some(opts) = options {
        if let Some(audio_config) = opts.audio_config {
            params.output_format = audio_format_to_polly_format(audio_config.format);

            if let Some(requested_rate) = audio_config.sample_rate {
                let validated_rate = match (requested_rate, &params.output_format) {
                    (rate, OutputFormat::Pcm) if rate <= 8000 => 8000,
                    (rate, OutputFormat::Pcm) if rate <= 16000 => 16000,
                    (_rate, OutputFormat::Pcm) => 16000,
                    (rate, _) if rate <= 8000 => 8000,
                    (rate, _) if rate <= 16000 => 16000,
                    (_, _) => 22050,
                };
                params.sample_rate = Some(validated_rate.to_string());
            } else {
                let default_rate = match params.output_format {
                    OutputFormat::Mp3 => "22050",
                    OutputFormat::Pcm => "16000",
                    OutputFormat::OggVorbis => "22050",
                    OutputFormat::Json => "22050",
                };
                params.sample_rate = Some(default_rate.to_string());
            }
        }

        if let Some(model_version) = opts.model_version {
            params.engine = Some(match model_version.as_str() {
                "standard" => Engine::Standard,
                "neural" => Engine::Neural,
                "long-form" => Engine::LongForm,
                "generative" => Engine::Generative,
                _ => Engine::Neural,
            });
        }

        if opts.enable_timing.unwrap_or(false) || opts.enable_word_timing.unwrap_or(false) {
            match params.output_format {
                OutputFormat::Json => {
                    let mut speech_marks = Vec::new();
                    if opts.enable_word_timing.unwrap_or(false) {
                        speech_marks.push(SpeechMarkType::Word);
                    }
                    speech_marks.push(SpeechMarkType::Sentence);
                    params.speech_mark_types = Some(speech_marks);
                }
                _ => {
                    trace!("Timing marks requested but not supported for format {:?}, ignoring", params.output_format);
                }
            }
        }
    }

    params
}

pub fn audio_format_to_polly_format(format: AudioFormat) -> OutputFormat {
    match format {
        AudioFormat::Mp3 => OutputFormat::Mp3,
        AudioFormat::Wav | AudioFormat::Pcm => OutputFormat::Pcm,
        AudioFormat::OggOpus => OutputFormat::OggVorbis,
        _ => OutputFormat::Mp3,
    }
}

pub fn polly_format_to_audio_format(format: &OutputFormat) -> AudioFormat {
    match format {
        OutputFormat::Mp3 => AudioFormat::Mp3,
        OutputFormat::Pcm => AudioFormat::Pcm,
        OutputFormat::OggVorbis => AudioFormat::OggOpus,
        OutputFormat::Json => AudioFormat::Mp3,
    }
}

pub fn audio_data_to_synthesis_result(
    audio_data: Vec<u8>,
    text: &str,
    format: &AudioFormat,
    sample_rate: u32,
) -> SynthesisResult {
    let audio_size = audio_data.len() as u32;
    let duration = estimate_audio_duration(&audio_data, sample_rate, format);

    let word_count = text.split_whitespace().count() as u32;
    let character_count = text.chars().count() as u32;

    SynthesisResult {
        audio_data,
        metadata: SynthesisMetadata {
            duration_seconds: duration,
            character_count,
            word_count,
            audio_size_bytes: audio_size,
            request_id: format!("polly-{}", chrono::Utc::now().timestamp()),
            provider_info: Some("AWS Polly".to_string()),
        },
    }
}

pub fn _create_validation_result(is_valid: bool, message: Option<String>) -> ValidationResult {
    ValidationResult {
        is_valid,
        character_count: 0,  
        estimated_duration: None,
        warnings: Vec::new(),
        errors: if is_valid {
            Vec::new()
        } else {
            vec![message.unwrap_or("Invalid input".to_string())]
        },
    }
}

pub fn validate_polly_input(text: &str, voice_id: &str) -> ValidationResult {
    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    let char_count = text.chars().count() as u32;
    if char_count == 0 {
        errors.push("Text cannot be empty".to_string());
    } else if char_count > 3000 {
        errors.push("Text exceeds 3000 character limit for real-time synthesis".to_string());
    } else if char_count > 1500 {
        warnings.push(
            "Text is quite long, consider using batch synthesis for better performance".to_string(),
        );
    }

    if voice_id.is_empty() {
        errors.push("Voice ID cannot be empty".to_string());
    }

    let is_ssml = text.trim_start().starts_with('<');
    if is_ssml && !text.contains("</speak>") {
        warnings.push("Text appears to be SSML but doesn't have proper closing tag".to_string());
    }

    let word_count = text.split_whitespace().count();
    let estimated_duration = if word_count > 0 {
        Some((word_count as f32 / 150.0) * 60.0)
    } else {
        None
    };

    ValidationResult {
        is_valid: errors.is_empty(),
        character_count: char_count,
        estimated_duration,
        warnings,
        errors,
    }
}

pub fn get_polly_language_info() -> Vec<LanguageInfo> {
    vec![
        LanguageInfo {
            code: "en-US".to_string(),
            name: "English (US)".to_string(),
            native_name: "English (United States)".to_string(),
            voice_count: 16,
        },
        LanguageInfo {
            code: "en-GB".to_string(),
            name: "English (UK)".to_string(),
            native_name: "English (United Kingdom)".to_string(),
            voice_count: 5,
        },
        LanguageInfo {
            code: "en-AU".to_string(),
            name: "English (Australia)".to_string(),
            native_name: "English (Australia)".to_string(),
            voice_count: 2,
        },
        LanguageInfo {
            code: "en-IN".to_string(),
            name: "English (India)".to_string(),
            native_name: "English (India)".to_string(),
            voice_count: 3,
        },
        LanguageInfo {
            code: "es-ES".to_string(),
            name: "Spanish (Spain)".to_string(),
            native_name: "Español (España)".to_string(),
            voice_count: 4,
        },
        LanguageInfo {
            code: "es-MX".to_string(),
            name: "Spanish (Mexico)".to_string(),
            native_name: "Español (México)".to_string(),
            voice_count: 2,
        },
        LanguageInfo {
            code: "es-US".to_string(),
            name: "Spanish (US)".to_string(),
            native_name: "Español (Estados Unidos)".to_string(),
            voice_count: 3,
        },
        LanguageInfo {
            code: "fr-FR".to_string(),
            name: "French (France)".to_string(),
            native_name: "Français (France)".to_string(),
            voice_count: 4,
        },
        LanguageInfo {
            code: "fr-CA".to_string(),
            name: "French (Canada)".to_string(),
            native_name: "Français (Canada)".to_string(),
            voice_count: 1,
        },
        LanguageInfo {
            code: "de-DE".to_string(),
            name: "German".to_string(),
            native_name: "Deutsch".to_string(),
            voice_count: 3,
        },
        LanguageInfo {
            code: "it-IT".to_string(),
            name: "Italian".to_string(),
            native_name: "Italiano".to_string(),
            voice_count: 2,
        },
        LanguageInfo {
            code: "pt-PT".to_string(),
            name: "Portuguese (Portugal)".to_string(),
            native_name: "Português (Portugal)".to_string(),
            voice_count: 2,
        },
        LanguageInfo {
            code: "pt-BR".to_string(),
            name: "Portuguese (Brazil)".to_string(),
            native_name: "Português (Brasil)".to_string(),
            voice_count: 3,
        },
        LanguageInfo {
            code: "ja-JP".to_string(),
            name: "Japanese".to_string(),
            native_name: "日本語".to_string(),
            voice_count: 3,
        },
        LanguageInfo {
            code: "ko-KR".to_string(),
            name: "Korean".to_string(),
            native_name: "한국어".to_string(),
            voice_count: 1,
        },
        LanguageInfo {
            code: "zh-CN".to_string(),
            name: "Chinese (Simplified)".to_string(),
            native_name: "中文（简体）".to_string(),
            voice_count: 1,
        },
        LanguageInfo {
            code: "cmn-CN".to_string(),
            name: "Chinese Mandarin".to_string(),
            native_name: "普通话".to_string(),
            voice_count: 1,
        },
        LanguageInfo {
            code: "ar".to_string(),
            name: "Arabic".to_string(),
            native_name: "العربية".to_string(),
            voice_count: 1,
        },
        LanguageInfo {
            code: "hi-IN".to_string(),
            name: "Hindi".to_string(),
            native_name: "हिन्दी".to_string(),
            voice_count: 2,
        },
        LanguageInfo {
            code: "ru-RU".to_string(),
            name: "Russian".to_string(),
            native_name: "Русский".to_string(),
            voice_count: 2,
        },
        LanguageInfo {
            code: "nl-NL".to_string(),
            name: "Dutch".to_string(),
            native_name: "Nederlands".to_string(),
            voice_count: 2,
        },
        LanguageInfo {
            code: "pl-PL".to_string(),
            name: "Polish".to_string(),
            native_name: "Polski".to_string(),
            voice_count: 2,
        },
        LanguageInfo {
            code: "sv-SE".to_string(),
            name: "Swedish".to_string(),
            native_name: "Svenska".to_string(),
            voice_count: 1,
        },
        LanguageInfo {
            code: "nb-NO".to_string(),
            name: "Norwegian".to_string(),
            native_name: "Norsk".to_string(),
            voice_count: 1,
        },
        LanguageInfo {
            code: "da-DK".to_string(),
            name: "Danish".to_string(),
            native_name: "Dansk".to_string(),
            voice_count: 2,
        },
        LanguageInfo {
            code: "tr-TR".to_string(),
            name: "Turkish".to_string(),
            native_name: "Türkçe".to_string(),
            voice_count: 1,
        },
        LanguageInfo {
            code: "ro-RO".to_string(),
            name: "Romanian".to_string(),
            native_name: "Română".to_string(),
            voice_count: 1,
        },
        LanguageInfo {
            code: "cy-GB".to_string(),
            name: "Welsh".to_string(),
            native_name: "Cymraeg".to_string(),
            voice_count: 1,
        },
        LanguageInfo {
            code: "is-IS".to_string(),
            name: "Icelandic".to_string(),
            native_name: "Íslenska".to_string(),
            voice_count: 2,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_duration_estimation() {
        let audio_data = vec![0u8; 1024];
        let duration = estimate_audio_duration(&audio_data, 22050, &AudioFormat::Mp3);
        assert!(duration > 0.0);
    }

    #[test]
    fn test_voice_filter_conversion() {
        let filter = VoiceFilter {
            language: Some("en-US".to_string()),
            gender: Some(VoiceGender::Female),
            quality: None,
            supports_ssml: None,
            provider: None,
            search_query: None,
        };

        let params = voice_filter_to_describe_params(Some(filter));
        assert!(params.is_some());
        assert_eq!(params.unwrap().language_code, Some("en-US".to_string()));
    }

    #[test]
    fn test_validation() {
        let result = validate_polly_input("Hello, world!", "Joanna");
        assert!(result.is_valid);
        assert_eq!(result.character_count, 13);

        let result = validate_polly_input("", "Joanna");
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_format_conversion() {
        assert_eq!(
            audio_format_to_polly_format(AudioFormat::Mp3),
            OutputFormat::Mp3
        );
        assert_eq!(
            audio_format_to_polly_format(AudioFormat::Wav),
            OutputFormat::Pcm
        );
        assert_eq!(
            polly_format_to_audio_format(&OutputFormat::Mp3),
            AudioFormat::Mp3
        );
    }
}
