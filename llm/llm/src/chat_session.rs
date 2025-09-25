use crate::golem::llm::llm::{
    ChatEvent, ChatResponse, ChatStream, CompleteResponse, Config, Guest, GuestChatSession,
    GuestChatStream, Message, ResponseMetadata, StreamEvent, ToolResult,
};
use std::sync::{Arc, RwLock};

pub struct ChatSession<LlmImpl>
where
    LlmImpl: Guest + 'static,
{
    config: Config,
    events: Arc<RwLock<Vec<ChatEvent>>>,
    _phantom_llm_impl: std::marker::PhantomData<LlmImpl>,
}

impl<LlmImpl> GuestChatSession for ChatSession<LlmImpl>
where
    LlmImpl: Guest + 'static,
{
    fn new(config: Config) -> Self {
        Self {
            config,
            events: Arc::new(RwLock::new(Vec::new())),
            _phantom_llm_impl: std::marker::PhantomData,
        }
    }

    fn add_message(&self, message: Message) -> () {
        self.events
            .write()
            .unwrap()
            .push(ChatEvent::Message(message));
    }

    fn add_messages(&self, messages: Vec<Message>) -> () {
        let mut events = self.events.write().unwrap();
        events.extend(messages.into_iter().map(|m| ChatEvent::Message(m)));
    }

    fn add_tool_result(&self, tool_result: ToolResult) -> () {
        self.events
            .write()
            .unwrap()
            .push(ChatEvent::ToolResults(vec![tool_result]));
    }

    fn add_tool_results(&self, tool_results: Vec<ToolResult>) -> () {
        self.events
            .write()
            .unwrap()
            .push(ChatEvent::ToolResults(tool_results));
    }

    fn get_chat_events(&self) -> Vec<ChatEvent> {
        self.events.read().unwrap().clone()
    }

    fn set_chat_events(&self, events: Vec<ChatEvent>) -> () {
        let mut e = self.events.write().unwrap();
        e.clear();
        e.extend(events)
    }

    fn send(&self) -> ChatResponse {
        let result = LlmImpl::send(self.config.clone(), self.get_chat_events());

        match &result {
            ChatResponse::Message(complete_response) => {
                self.events
                    .write()
                    .unwrap()
                    .push(ChatEvent::Response(complete_response.clone()));
            }
            ChatResponse::ToolCalls(tool_calls) => {
                self.events
                    .write()
                    .unwrap()
                    .push(ChatEvent::ToolCalls(tool_calls.clone()));
            }
            ChatResponse::Error(_) => {
                // NOP
            }
        }

        result
    }

    fn stream(&self) -> ChatStream {
        ChatStream::new(ChatSessionStreamAdapter::new(
            self.events.clone(),
            LlmImpl::stream(self.config.clone(), self.get_chat_events())
                .into_inner::<LlmImpl::ChatStream>(),
        ))
    }
}

struct ChatSessionStreamAdapter<ChatStreamImpl>
where
    ChatStreamImpl: GuestChatStream + 'static,
{
    events: Arc<RwLock<Vec<ChatEvent>>>,
    complete_response: RwLock<Option<CompleteResponse>>,
    inner: ChatStreamImpl,
    _phantom_chat_stream_impl: std::marker::PhantomData<ChatStreamImpl>,
}

impl<ChatStreamImpl> ChatSessionStreamAdapter<ChatStreamImpl>
where
    ChatStreamImpl: GuestChatStream + 'static,
{
    pub fn new(events: Arc<RwLock<Vec<ChatEvent>>>, inner: ChatStreamImpl) -> Self {
        Self {
            events,
            complete_response: RwLock::new(Some(CompleteResponse {
                id: "".to_string(),
                content: vec![],
                tool_calls: vec![],
                metadata: ResponseMetadata {
                    finish_reason: None,
                    usage: None,
                    provider_id: None,
                    timestamp: None,
                    provider_metadata_json: None,
                },
            })),
            inner,
            _phantom_chat_stream_impl: std::marker::PhantomData,
        }
    }

    fn add_stream_events(&self, events: &[StreamEvent]) -> () {
        for event in events {
            match event {
                StreamEvent::Delta(delta) => {
                    let mut complete_response = self.complete_response.write().unwrap();
                    let complete_response = complete_response.as_mut().unwrap();

                    if let Some(content) = &delta.content {
                        complete_response.content.extend(content.iter().cloned());
                    }

                    if let Some(tool_calls) = &delta.tool_calls {
                        complete_response
                            .tool_calls
                            .extend(tool_calls.iter().cloned());
                    }
                }
                StreamEvent::Finish(metadata) => {
                    let mut complete_response =
                        self.complete_response.write().unwrap().take().unwrap();

                    complete_response.metadata = metadata.clone();
                    self.events
                        .write()
                        .unwrap()
                        .push(ChatEvent::Response(complete_response));
                }
                StreamEvent::Error(_) => {
                    // NOP
                }
            }
        }
    }
}

impl<ChatStreamImpl> GuestChatStream for ChatSessionStreamAdapter<ChatStreamImpl>
where
    ChatStreamImpl: GuestChatStream + 'static,
{
    fn get_next(&self) -> Option<Vec<StreamEvent>> {
        let result = self.inner.get_next();
        if let Some(events) = result.as_ref() {
            self.add_stream_events(events);
        }
        result
    }

    fn blocking_get_next(&self) -> Vec<StreamEvent> {
        let events = self.inner.blocking_get_next();
        self.add_stream_events(&events);
        events
    }
}
