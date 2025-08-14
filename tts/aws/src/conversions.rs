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

/// Validate synthesis input and options for proper error handling
pub fn validate_synthesis_input(
    input: &golem_tts::golem::tts::types::TextInput,
    options: Option<&SynthesisOptions>,
) -> Result<(), TtsError> {
    // Validate empty text
    if input.content.trim().is_empty() {
        return Err(TtsError::InvalidText("Text content cannot be empty".to_string()));
    }

    // Note: We no longer reject large text here - we'll chunk it in synthesis
    // AWS Polly limit is 3000 characters, but we'll handle this by chunking

    // Validate SSML content if specified
    if input.text_type == TtsTextType::Ssml {
        if let Err(msg) = validate_ssml_content(&input.content) {
            return Err(TtsError::InvalidSsml(msg));
        }
    }

    // Validate language code if specified
    if let Some(ref language) = input.language {
        if !is_supported_language(language) {
            return Err(TtsError::UnsupportedLanguage(format!(
                "Language '{}' is not supported by AWS Polly", language
            )));
        }
    }

    // Validate voice settings if specified
    if let Some(opts) = options {
        if let Some(ref voice_settings) = opts.voice_settings {
            validate_voice_settings(voice_settings)?;
        }
    }

    Ok(())
}

/// Validate SSML content for basic structure
pub fn validate_ssml_content(content: &str) -> Result<(), String> {
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

/// Validate voice settings for AWS Polly limits
pub fn validate_voice_settings(settings: &VoiceSettings) -> Result<(), TtsError> {
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

/// Intelligently split text into chunks suitable for AWS Polly
/// Following best practices for text chunking
pub fn split_text_intelligently(text: &str, max_chunk_size: usize) -> Vec<String> {
    if text.len() <= max_chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    
    // Split by sentence-ending punctuation, keeping the delimiter
    let mut sentences = Vec::new();
    let mut current_sentence = String::new();
    let mut chars = text.chars().peekable();
    
    while let Some(ch) = chars.next() {
        current_sentence.push(ch);
        
        // Check for sentence endings
        if matches!(ch, '.' | '!' | '?') {
            // Check if it's really the end of a sentence (not an abbreviation)
            if chars.peek().map(|c| c.is_whitespace()).unwrap_or(true) {
                sentences.push(current_sentence.trim().to_string());
                current_sentence.clear();
            }
        } else if ch == '\n' {
            // Treat newlines as sentence boundaries
            let sentence = current_sentence.trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
                current_sentence.clear();
            }
        }
    }
    
    // Add any remaining text as a sentence
    let remaining = current_sentence.trim();
    if !remaining.is_empty() {
        sentences.push(remaining.to_string());
    }

    // Now group sentences into chunks
    for sentence in sentences {
        if sentence.trim().is_empty() {
            continue;
        }

        // If adding this sentence would exceed the limit, start a new chunk
        if current_chunk.len() + sentence.len() + 1 > max_chunk_size {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
            }

            // If a single sentence is too long, split it by words
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

    // Add the last chunk if it's not empty
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    // Ensure we have at least one chunk
    if chunks.is_empty() {
        chunks.push(text.to_string());
    }

    chunks
}

/// Split text by words when a single sentence is too long
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
            
            // If a single word is too long, truncate it (rare case)
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

/// Combine multiple audio chunks into a single audio result
pub fn combine_audio_chunks(chunks: Vec<Vec<u8>>, format: &AudioFormat) -> Vec<u8> {
    if chunks.is_empty() {
        return Vec::new();
    }

    if chunks.len() == 1 {
        return chunks.into_iter().next().unwrap();
    }

    // For MP3, we need to concatenate without headers (simplified approach)
    // For PCM, we can directly concatenate
    // For OGG, we need special handling
    match format {
        AudioFormat::Pcm => {
            // Simple concatenation for PCM
            chunks.into_iter().flatten().collect()
        }
        AudioFormat::Mp3 => {
            // For MP3, we'll do simple concatenation
            // Note: This is a simplified approach. For production, you'd want proper MP3 frame handling
            chunks.into_iter().flatten().collect()
        }
        AudioFormat::OggOpus => {
            // For OGG, simple concatenation may not work perfectly, but it's a start
            chunks.into_iter().flatten().collect()
        }
        _ => {
            // Fallback: simple concatenation
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
            // For MP3, estimate based on average bitrate
            let estimated_bitrate_bps = 128000; // 128 kbps average for Polly
            let bytes_per_second = estimated_bitrate_bps / 8;
            (audio_data.len() as f32) / (bytes_per_second as f32)
        }
        AudioFormat::Wav | AudioFormat::Pcm => {
            // For uncompressed audio: bytes / (sample_rate * channels * bytes_per_sample)
            let channels = 1; // Polly outputs mono by default
            let bytes_per_sample = 2; // 16-bit samples
            let bytes_per_second = sample_rate * channels * bytes_per_sample;
            (audio_data.len() as f32) / (bytes_per_second as f32)
        }
        AudioFormat::OggOpus => {
            // For Opus, estimate based on average bitrate
            let estimated_bitrate_bps = 64000; // 64 kbps average
            let bytes_per_second = estimated_bitrate_bps / 8;
            (audio_data.len() as f32) / (bytes_per_second as f32)
        }
        _ => {
            // Fallback estimation based on text length
            // Assume average speaking rate of 150 words per minute
            let estimated_chars = audio_data.len() / 100; // rough estimate
            (estimated_chars as f32 / 150.0) * 60.0
        }
    }
}

pub fn voice_filter_to_describe_params(
    filter: Option<VoiceFilter>,
) -> Option<DescribeVoicesParams> {
    filter.map(|f| DescribeVoicesParams {
        engine: None, // Will use default neural engine
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

    // Determine quality based on supported engines
    let quality = if voice.supported_engines.contains(&"neural".to_string()) {
        VoiceQuality::Neural
    } else if voice.supported_engines.contains(&"generative".to_string()) {
        VoiceQuality::Studio
    } else {
        VoiceQuality::Standard
    };

    // Extract additional languages before moving voice
    let additional_languages = voice.additional_language_codes.clone().unwrap_or_default();

    // Infer use cases based on voice characteristics
    let use_cases = infer_use_cases_from_aws_voice(&voice);

    // Determine sample rate based on engine support
    // AWS Polly supports 8000, 16000, and 22050 Hz for all engines
    let sample_rate = 22050; // Use highest supported rate as default

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
        is_custom: false, // AWS Polly doesn't have custom voices in the same way
        is_cloned: false,
        preview_url: None, // AWS Polly doesn't provide preview URLs
        use_cases,
    }
}

fn infer_use_cases_from_aws_voice(voice: &AwsVoice) -> Vec<String> {
    let mut use_cases = Vec::new();

    // Base use cases for all voices
    use_cases.push("general".to_string());
    use_cases.push("content".to_string());

    // Neural engine voices have enhanced capabilities
    if voice.supported_engines.contains(&"neural".to_string()) {
        use_cases.push("audiobooks".to_string());
        use_cases.push("news".to_string());
        use_cases.push("conversational".to_string());
    }

    // Long-form engine for extended content
    if voice.supported_engines.contains(&"long-form".to_string()) {
        use_cases.push("long-form".to_string());
        use_cases.push("books".to_string());
    }

    // Generative engine for expressive speech
    if voice.supported_engines.contains(&"generative".to_string()) {
        use_cases.push("expressive".to_string());
        use_cases.push("creative".to_string());
    }

    // Language-specific use cases
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
        engine: Some(Engine::Neural), // Default to neural engine
        language_code: None,
        lexicon_names: None,
        output_format: OutputFormat::Mp3, // Default format
        sample_rate: None,
        speech_mark_types: None,
        text,
        text_type: Some(TextType::Text), // Default to plain text
        voice_id,
    };

    if let Some(opts) = options {
        // Set audio format
        if let Some(audio_config) = opts.audio_config {
            params.output_format = audio_format_to_polly_format(audio_config.format);

            // AWS Polly supports specific sample rates: 8000, 16000, 22050
            if let Some(requested_rate) = audio_config.sample_rate {
                let validated_rate = match (requested_rate, &params.output_format) {
                    // For PCM, use more conservative sample rates
                    (rate, OutputFormat::Pcm) if rate <= 8000 => 8000,
                    (rate, OutputFormat::Pcm) if rate <= 16000 => 16000,
                    (_rate, OutputFormat::Pcm) => 16000, // PCM works better at 16kHz
                    // For other formats, use the full range
                    (rate, _) if rate <= 8000 => 8000,
                    (rate, _) if rate <= 16000 => 16000,
                    (_, _) => 22050, // Default to highest supported rate
                };
                params.sample_rate = Some(validated_rate.to_string());
            } else {
                // Set default sample rate based on format and engine
                let default_rate = match params.output_format {
                    OutputFormat::Mp3 => "22050",
                    OutputFormat::Pcm => "16000", // PCM often works better at 16kHz
                    OutputFormat::OggVorbis => "22050",
                    OutputFormat::Json => "22050",
                };
                params.sample_rate = Some(default_rate.to_string());
            }
        }

        // Set engine based on model version
        if let Some(model_version) = opts.model_version {
            params.engine = Some(match model_version.as_str() {
                "standard" => Engine::Standard,
                "neural" => Engine::Neural,
                "long-form" => Engine::LongForm,
                "generative" => Engine::Generative,
                _ => Engine::Neural,
            });
        }

        // Enable timing marks if requested - but only for compatible formats
        // AWS Polly only supports speech marks with JSON format
        if opts.enable_timing.unwrap_or(false) || opts.enable_word_timing.unwrap_or(false) {
            match params.output_format {
                OutputFormat::Json => {
                    // Only enable speech marks for JSON format
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
        _ => OutputFormat::Mp3, // Default fallback
    }
}

pub fn polly_format_to_audio_format(format: &OutputFormat) -> AudioFormat {
    match format {
        OutputFormat::Mp3 => AudioFormat::Mp3,
        OutputFormat::Pcm => AudioFormat::Pcm,
        OutputFormat::OggVorbis => AudioFormat::OggOpus,
        OutputFormat::Json => AudioFormat::Mp3, // Fallback for speech marks
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

    // Count words and characters
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
        character_count: 0, // Will be filled by caller
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

    // Check text length limits
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

    // Validate voice ID format
    if voice_id.is_empty() {
        errors.push("Voice ID cannot be empty".to_string());
    }

    // Check for SSML
    let is_ssml = text.trim_start().starts_with('<');
    if is_ssml && !text.contains("</speak>") {
        warnings.push("Text appears to be SSML but doesn't have proper closing tag".to_string());
    }

    // Estimate duration (150 words per minute average)
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

/// Get language information for AWS Polly supported languages
pub fn get_polly_language_info() -> Vec<LanguageInfo> {
    vec![
        // English variants
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
        // Spanish variants
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
        // French variants
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
        // German
        LanguageInfo {
            code: "de-DE".to_string(),
            name: "German".to_string(),
            native_name: "Deutsch".to_string(),
            voice_count: 3,
        },
        // Italian
        LanguageInfo {
            code: "it-IT".to_string(),
            name: "Italian".to_string(),
            native_name: "Italiano".to_string(),
            voice_count: 2,
        },
        // Portuguese variants
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
        // Japanese
        LanguageInfo {
            code: "ja-JP".to_string(),
            name: "Japanese".to_string(),
            native_name: "日本語".to_string(),
            voice_count: 3,
        },
        // Korean
        LanguageInfo {
            code: "ko-KR".to_string(),
            name: "Korean".to_string(),
            native_name: "한국어".to_string(),
            voice_count: 1,
        },
        // Chinese variants
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
        // Arabic
        LanguageInfo {
            code: "ar".to_string(),
            name: "Arabic".to_string(),
            native_name: "العربية".to_string(),
            voice_count: 1,
        },
        // Hindi
        LanguageInfo {
            code: "hi-IN".to_string(),
            name: "Hindi".to_string(),
            native_name: "हिन्दी".to_string(),
            voice_count: 2,
        },
        // Russian
        LanguageInfo {
            code: "ru-RU".to_string(),
            name: "Russian".to_string(),
            native_name: "Русский".to_string(),
            voice_count: 2,
        },
        // Dutch
        LanguageInfo {
            code: "nl-NL".to_string(),
            name: "Dutch".to_string(),
            native_name: "Nederlands".to_string(),
            voice_count: 2,
        },
        // Polish
        LanguageInfo {
            code: "pl-PL".to_string(),
            name: "Polish".to_string(),
            native_name: "Polski".to_string(),
            voice_count: 2,
        },
        // Swedish
        LanguageInfo {
            code: "sv-SE".to_string(),
            name: "Swedish".to_string(),
            native_name: "Svenska".to_string(),
            voice_count: 1,
        },
        // Norwegian
        LanguageInfo {
            code: "nb-NO".to_string(),
            name: "Norwegian".to_string(),
            native_name: "Norsk".to_string(),
            voice_count: 1,
        },
        // Danish
        LanguageInfo {
            code: "da-DK".to_string(),
            name: "Danish".to_string(),
            native_name: "Dansk".to_string(),
            voice_count: 2,
        },
        // Turkish
        LanguageInfo {
            code: "tr-TR".to_string(),
            name: "Turkish".to_string(),
            native_name: "Türkçe".to_string(),
            voice_count: 1,
        },
        // Romanian
        LanguageInfo {
            code: "ro-RO".to_string(),
            name: "Romanian".to_string(),
            native_name: "Română".to_string(),
            voice_count: 1,
        },
        // Welsh
        LanguageInfo {
            code: "cy-GB".to_string(),
            name: "Welsh".to_string(),
            native_name: "Cymraeg".to_string(),
            voice_count: 1,
        },
        // Icelandic
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
