use crate::golem::llm::llm::{
    ChatEvent, ChatResponse, ChatStream, Config, GuestChatSession, Message, ToolResult,
};
use std::sync::RwLock;

pub struct ChatSession {
    config: Config,
    events: RwLock<Vec<ChatEvent>>,
}

impl GuestChatSession for ChatSession {
    fn new(config: Config) -> Self {
        Self {
            config,
            events: RwLock::new(Vec::new()),
        }
    }

    fn add_message(&self, message: Message) -> () {
        todo!()
    }

    fn add_tool_result(&self, tool_result: ToolResult) -> () {
        todo!()
    }

    fn get_chat_events(&self) -> Vec<ChatEvent> {
        todo!()
    }

    fn set_chat_events(&self, events: Vec<ChatEvent>) -> () {
        todo!()
    }

    fn send(&self) -> ChatResponse {
        todo!()
    }

    fn stream(&self) -> ChatStream {
        todo!()
    }
}
