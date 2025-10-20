mod client;
mod conversions;

use client::EmbeddingsApi;
use conversions::{create_request, process_embedding_response};
use golem_embed::{
    config::with_config_key,
    durability::{DurableEmbed, ExtendedGuest},
    golem::embed::embed::{
        Config, ContentPart, EmbeddingResponse, Error, ErrorCode, Guest, RerankResponse,
    },
    LOGGING_STATE,
};

struct OpenAIComponent;

impl OpenAIComponent {
    const ENV_VAR_NAME: &'static str = "OPENAI_API_KEY";

    fn embeddings(
        client: EmbeddingsApi,
        inputs: Vec<ContentPart>,
        config: Config,
    ) -> Result<EmbeddingResponse, Error> {
        let request = create_request(inputs, config);
        match request {
            Ok(request) => match client.generate_embeding(request) {
                Ok(response) => process_embedding_response(response),
                Err(err) => Err(err),
            },
            Err(err) => Err(err),
        }
    }
}

impl Guest for OpenAIComponent {
    fn generate(inputs: Vec<ContentPart>, config: Config) -> Result<EmbeddingResponse, Error> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        with_config_key(Self::ENV_VAR_NAME, Err, |openai_api_key| {
            let client = EmbeddingsApi::new(openai_api_key);
            Self::embeddings(client, inputs, config)
        })
    }

    fn rerank(
        _query: String,
        _documents: Vec<String>,
        _config: Config,
    ) -> Result<RerankResponse, Error> {
        Err(Error {
            code: ErrorCode::Unsupported,
            message: "OpenAI does not support rerank".to_string(),
            provider_error_json: None,
        })
    }
}

impl ExtendedGuest for OpenAIComponent {}

type DurableOpenAIComponent = DurableEmbed<OpenAIComponent>;

golem_embed::export_embed!(DurableOpenAIComponent with_types_in golem_embed);
