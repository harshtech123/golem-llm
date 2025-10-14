use std::marker::PhantomData;

use crate::exports::golem::stt::languages::Guest as WitLanguageGuest;
use crate::guest::SttTranscriptionGuest;

pub struct DurableStt<Impl> {
    phantom: PhantomData<Impl>,
}

pub trait ExtendedGuest: SttTranscriptionGuest + WitLanguageGuest + 'static {}

#[cfg(not(feature = "durability"))]
mod passthrough_impl {
    use bytes::Bytes;

    use crate::exports::golem::stt::languages::{
        Guest as WitLanguageGuest, LanguageInfo as WitLanguageInfo,
    };

    use crate::durability::{DurableStt, ExtendedGuest};
    use crate::exports::golem::stt::transcription::{
        Guest as WitTranscriptionGuest, MultiTranscriptionResult as WitMultiTranscriptionResult,
        TranscriptionRequest as WitTranscriptionRequest,
    };

    use crate::exports::golem::stt::types::{
        SttError as WitSttError, TranscriptionResult as WitTranscriptionResult,
    };

    use crate::guest::SttTranscriptionRequest;
    use crate::LOGGING_STATE;
    use golem_rust::{FromValueAndType, IntoValue};

    impl<Impl: ExtendedGuest> WitTranscriptionGuest for DurableStt<Impl> {
        fn transcribe(
            request: WitTranscriptionRequest,
        ) -> Result<WitTranscriptionResult, WitSttError> {
            LOGGING_STATE.with_borrow_mut(|state| state.init());

            let request = SttTranscriptionRequest {
                request_id: request.request_id,
                audio: Bytes::from(request.audio),
                config: request.config,
                options: request.options,
            };

            Impl::transcribe(request)
        }

        fn transcribe_many(
            requests: Vec<WitTranscriptionRequest>,
        ) -> Result<WitMultiTranscriptionResult, WitSttError> {
            LOGGING_STATE.with_borrow_mut(|state| state.init());

            let stt_requests: Vec<SttTranscriptionRequest> = requests
                .into_iter()
                .map(|req| SttTranscriptionRequest {
                    request_id: req.request_id,
                    audio: Bytes::from(req.audio),
                    config: req.config,
                    options: req.options,
                })
                .collect();

            Impl::transcribe_many(stt_requests)
        }
    }

    impl<Impl: ExtendedGuest> WitLanguageGuest for DurableStt<Impl> {
        fn list_languages() -> Result<Vec<WitLanguageInfo>, WitSttError> {
            Impl::list_languages()
        }
    }

    #[derive(Debug, Clone, PartialEq, IntoValue, FromValueAndType)]
    struct TranscribeInput {
        request: WitTranscriptionRequest,
    }

    #[derive(Debug, Clone, PartialEq, IntoValue, FromValueAndType)]
    struct TranscribeManyInput {
        requests: Vec<WitTranscriptionRequest>,
    }

    impl From<&WitSttError> for WitSttError {
        fn from(error: &WitSttError) -> Self {
            error.clone()
        }
    }
}

#[cfg(feature = "durability")]
mod durable_impl {
    use bytes::Bytes;
    use golem_rust::bindings::golem::durability::durability::DurableFunctionType;
    use golem_rust::durability::Durability;

    use crate::exports::golem::stt::languages::{
        Guest as WitLanguageGuest, LanguageInfo as WitLanguageInfo,
    };

    use crate::durability::{DurableStt, ExtendedGuest};
    use crate::exports::golem::stt::transcription::{
        Guest as WitTranscriptionGuest, MultiTranscriptionResult as WitMultiTranscriptionResult,
        TranscriptionRequest as WitTranscriptionRequest,
    };

    use crate::exports::golem::stt::types::{
        SttError as WitSttError, TranscriptionResult as WitTranscriptionResult,
    };

    use crate::guest::SttTranscriptionRequest;
    use crate::LOGGING_STATE;
    use golem_rust::{with_persistence_level, FromValueAndType, IntoValue, PersistenceLevel};

    impl<Impl: ExtendedGuest> WitTranscriptionGuest for DurableStt<Impl> {
        fn transcribe(
            request: WitTranscriptionRequest,
        ) -> Result<WitTranscriptionResult, WitSttError> {
            LOGGING_STATE.with_borrow_mut(|state| state.init());
            let durability = Durability::<WitTranscriptionResult, WitSttError>::new(
                "golem_stt",
                "transcribe",
                DurableFunctionType::WriteRemote,
            );

            let audio_bytes = Bytes::from(request.audio);
            let request_id = request.request_id;
            let config = request.config;
            let options = request.options;

            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    let request = SttTranscriptionRequest {
                        request_id: request_id.clone(),
                        audio: audio_bytes.clone(),
                        config,
                        options: options.clone(),
                    };

                    Impl::transcribe(request)
                });

                // Reconstruct original request for persistence
                let orig_request_copy = WitTranscriptionRequest {
                    request_id,
                    audio: audio_bytes.to_vec(),
                    config,
                    options,
                };

                durability.persist(
                    TranscribeInput {
                        request: orig_request_copy,
                    },
                    result,
                )
            } else {
                durability.replay()
            }
        }

        fn transcribe_many(
            requests: Vec<WitTranscriptionRequest>,
        ) -> Result<WitMultiTranscriptionResult, WitSttError> {
            LOGGING_STATE.with_borrow_mut(|state| state.init());
            let durability = Durability::<WitMultiTranscriptionResult, WitSttError>::new(
                "golem_stt",
                "transcribe_many",
                DurableFunctionType::WriteRemote,
            );

            let requests_with_bytes: Vec<_> = requests
                .into_iter()
                .map(|req| {
                    (
                        Bytes::from(req.audio),
                        req.request_id,
                        req.config,
                        req.options,
                    )
                })
                .collect();

            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    let stt_requests: Vec<SttTranscriptionRequest> = requests_with_bytes
                        .iter()
                        .map(
                            |(audio_bytes, request_id, config, options)| SttTranscriptionRequest {
                                request_id: request_id.clone(),
                                audio: audio_bytes.clone(),
                                config: *config,
                                options: options.clone(),
                            },
                        )
                        .collect();

                    Impl::transcribe_many(stt_requests)
                });

                // Reconstruct original requests for persistence
                let orig_requests_copy: Vec<WitTranscriptionRequest> = requests_with_bytes
                    .into_iter()
                    .map(
                        |(audio_bytes, request_id, config, options)| WitTranscriptionRequest {
                            request_id,
                            audio: audio_bytes.to_vec(),
                            config,
                            options,
                        },
                    )
                    .collect();

                durability.persist(
                    TranscribeManyInput {
                        requests: orig_requests_copy,
                    },
                    result,
                )
            } else {
                durability.replay()
            }
        }
    }

    impl<Impl: ExtendedGuest> WitLanguageGuest for DurableStt<Impl> {
        fn list_languages() -> Result<Vec<WitLanguageInfo>, WitSttError> {
            Impl::list_languages()
        }
    }

    #[derive(Debug, Clone, PartialEq, IntoValue, FromValueAndType)]
    struct TranscribeInput {
        request: WitTranscriptionRequest,
    }

    #[derive(Debug, Clone, PartialEq, IntoValue, FromValueAndType)]
    struct TranscribeManyInput {
        requests: Vec<WitTranscriptionRequest>,
    }

    impl From<&WitSttError> for WitSttError {
        fn from(error: &WitSttError) -> Self {
            error.clone()
        }
    }
}
