use crate::golem::llm::llm::{
    ChatError, ChatEvent, ChatResponse, ChatStream, Config, Guest, GuestChatSession,
    GuestChatStream, Message, ResponseMetadata, StreamEvent, ToolResult,
};
use std::cell::RefCell;
use std::rc::Rc;

pub struct ChatSession<LlmImpl>
where
    LlmImpl: Guest + 'static,
{
    config: Config,
    events: Rc<RefCell<Vec<ChatEvent>>>,
    _phantom_llm_impl: std::marker::PhantomData<LlmImpl>,
}

impl<LlmImpl> GuestChatSession for ChatSession<LlmImpl>
where
    LlmImpl: Guest + 'static,
{
    fn new(config: Config) -> Self {
        Self {
            config,
            events: Rc::new(RefCell::new(Vec::new())),
            _phantom_llm_impl: std::marker::PhantomData,
        }
    }

    fn add_message(&self, message: Message) -> () {
        self.events.borrow_mut().push(ChatEvent::Message(message));
    }

    fn add_messages(&self, messages: Vec<Message>) -> () {
        let mut events = self.events.borrow_mut();
        events.extend(messages.into_iter().map(|m| ChatEvent::Message(m)));
    }

    fn add_tool_result(&self, tool_result: ToolResult) -> () {
        self.events
            .borrow_mut()
            .push(ChatEvent::ToolResults(vec![tool_result]));
    }

    fn add_tool_results(&self, tool_results: Vec<ToolResult>) -> () {
        self.events
            .borrow_mut()
            .push(ChatEvent::ToolResults(tool_results));
    }

    fn get_chat_events(&self) -> Vec<ChatEvent> {
        self.events.borrow().clone()
    }

    fn set_chat_events(&self, events: Vec<ChatEvent>) -> () {
        let mut e = self.events.borrow_mut();
        e.clear();
        e.extend(events)
    }

    fn send(&self) -> Result<ChatResponse, ChatError> {
        let result = LlmImpl::send(self.config.clone(), self.get_chat_events());

        match &result {
            Ok(response) => {
                self.events
                    .borrow_mut()
                    .push(ChatEvent::Response(response.clone()));
            }
            Err(_) => {
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
    events: Rc<RefCell<Vec<ChatEvent>>>,
    chat_response: RefCell<Option<ChatResponse>>,
    inner: ChatStreamImpl,
    _phantom_chat_stream_impl: std::marker::PhantomData<ChatStreamImpl>,
}

impl<ChatStreamImpl> ChatSessionStreamAdapter<ChatStreamImpl>
where
    ChatStreamImpl: GuestChatStream + 'static,
{
    pub fn new(events: Rc<RefCell<Vec<ChatEvent>>>, inner: ChatStreamImpl) -> Self {
        Self {
            events,
            chat_response: RefCell::new(Some(ChatResponse {
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
                    let mut complete_response = self.chat_response.borrow_mut();
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
                    let mut complete_response = self.chat_response.borrow_mut().take().unwrap();

                    complete_response.metadata = metadata.clone();
                    self.events
                        .borrow_mut()
                        .push(ChatEvent::Response(complete_response));
                }
            }
        }
    }
}

impl<ChatStreamImpl> GuestChatStream for ChatSessionStreamAdapter<ChatStreamImpl>
where
    ChatStreamImpl: GuestChatStream + 'static,
{
    fn poll_next(&self) -> Result<Option<Vec<StreamEvent>>, ChatError> {
        let result = self.inner.poll_next();
        if let Ok(Some(events)) = &result {
            self.add_stream_events(events);
        }
        result
    }

    fn get_next(&self) -> Result<Vec<StreamEvent>, ChatError> {
        let result = self.inner.get_next();
        if let Ok(events) = &result {
            self.add_stream_events(events);
        }
        result
    }
}
