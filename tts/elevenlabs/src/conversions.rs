use crate::bindings::exports::golem::tts::{
    synthesis::GuestSynthesisOptions,
    types::{TextInput, TextType, TtsError, VoiceSettings},
    voices::{GuestVoiceFilter, GuestVoiceInfo},
};
use golem_tts::golem::tts::voices::VoiceQuality;
use golem_tts::golem::tts::types::VoiceGender;
use golem_tts::golem::tts::voices::{AudioFormat};
use crate::client::{
    ListVoicesParams, TextToSpeechRequest, Voice, VoiceSettings as ClientVoiceSettings,
};

pub fn convert_voice_filter_from_guest(filter: GuestVoiceFilter) -> ListVoicesParams {
    ListVoicesParams {
        next_page_token: None,
        page_size: Some(50), // Default page size
        search: filter.search_query,
        sort: None,
        sort_direction: None,
        voice_type: None,
        category: None,
        include_total_count: Some(true),
    }
}

pub fn convert_voice_info_to_guest(voice: Voice) -> GuestVoiceInfo {
    GuestVoiceInfo {
        id: voice.voice_id,
        name: voice.name,
        language: "en".to_string(), // Default to English for ElevenLabs
        additional_languages: voice.high_quality_base_model_ids.unwrap_or_default(),
        gender: VoiceGender::Neutral,
        quality: VoiceQuality::Neural,
        description: voice.description,
        provider: "elevenlabs".to_string(),
        sample_rate: 44100, // Default sample rate
        is_custom: voice.category != "premade",
        is_cloned: voice.category == "cloned",
        preview_url: voice.preview_url,
        use_cases: vec!["text-to-speech".to_string()],
    }
}

pub fn convert_synthesis_request(
    input: TextInput,
    options: Option<GuestSynthesisOptions>,
) -> Result<TextToSpeechRequest, TtsError> {
    // Validate input
    if input.content.is_empty() {
        return Err(TtsError::InvalidText("Text content cannot be empty".to_string()));
    }

    if input.content.len() > 5000 {
        return Err(TtsError::TextTooLong(input.content.len() as u32));
    }

    // Check if SSML is being used but not supported in this context
    if input.text_type == TextType::Ssml {
        // ElevenLabs supports SSML, but we'll just pass it as text for now
        // In a full implementation, you'd want to handle SSML properly
    }

    let voice_settings = options
        .as_ref()
        .and_then(|opts| opts.voice_settings.as_ref())
        .map(convert_voice_settings);

    Ok(TextToSpeechRequest {
        text: input.content,
        model_id: options
            .as_ref()
            .and_then(|opts| opts.model_version.clone())
            .or_else(|| Some("eleven_multilingual_v2".to_string())),
        language_code: input.language,
        voice_settings,
        pronunciation_dictionary_locators: None,
        seed: options.as_ref().and_then(|opts| opts.seed),
        previous_text: options
            .as_ref()
            .and_then(|opts| opts.context.as_ref())
            .and_then(|ctx| ctx.previous_text.clone()),
        next_text: options
            .as_ref()
            .and_then(|opts| opts.context.as_ref())
            .and_then(|ctx| ctx.next_text.clone()),
        previous_request_ids: None,
        next_request_ids: None,
        apply_text_normalization: Some("auto".to_string()),
        apply_language_text_normalization: Some(false),
        use_pvc_as_ivc: Some(false),
    })
}

fn convert_voice_settings(settings: &VoiceSettings) -> ClientVoiceSettings {
    ClientVoiceSettings {
        stability: settings.stability,
        similarity_boost: settings.similarity,
        style: settings.style,
        use_speaker_boost: None,
        speed: settings.speed,
    }
}

// Helper function to convert audio format strings
pub fn convert_audio_format_to_string(
    format: &AudioFormat,
) -> String {
    match format {
        AudioFormat::Mp3 => "mp3_44100_128".to_string(),
        AudioFormat::Wav => "wav".to_string(),
        AudioFormat::Pcm => "pcm_44100".to_string(),
        AudioFormat::OggOpus => "ogg_opus".to_string(),
        AudioFormat::Aac => "aac".to_string(),
        AudioFormat::Flac => "flac".to_string(),
        AudioFormat::Mulaw => "ulaw_8000".to_string(),
        AudioFormat::Alaw => "alaw_8000".to_string(),
    }
}
