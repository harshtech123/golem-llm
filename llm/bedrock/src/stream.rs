use crate::{
    async_utils,
    conversions::{converse_stream_output_to_stream_event, custom_error, merge_metadata},
};
use aws_sdk_bedrockruntime::{
    self as bedrock, primitives::event_stream::EventReceiver,
    types::error::ConverseStreamOutputError,
};
use golem_llm::golem::llm::llm;
use std::cell::{RefCell, RefMut};

type BedrockEventSource =
    EventReceiver<bedrock::types::ConverseStreamOutput, ConverseStreamOutputError>;

pub struct BedrockChatStream {
    stream: RefCell<Option<BedrockEventSource>>,
    failure: Option<llm::Error>,
    finished: RefCell<bool>,
}

impl BedrockChatStream {
    pub fn new(stream: BedrockEventSource) -> BedrockChatStream {
        BedrockChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
        }
    }

    pub fn failed(error: llm::Error) -> BedrockChatStream {
        BedrockChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(true),
        }
    }

    fn stream_mut(&self) -> RefMut<'_, Option<BedrockEventSource>> {
        self.stream.borrow_mut()
    }

    fn failure(&self) -> &Option<llm::Error> {
        &self.failure
    }

    fn is_finished(&self) -> bool {
        *self.finished.borrow()
    }

    fn set_finished(&self) {
        *self.finished.borrow_mut() = true;
    }
    fn get_single_event(&self) -> Result<Option<llm::StreamEvent>, llm::Error> {
        if let Some(stream) = self.stream_mut().as_mut() {
            let runtime = async_utils::get_async_runtime();

            runtime.block_on(async move {
                let token = stream.recv().await;
                log::trace!("Bedrock stream event: {token:?}");

                match token {
                    Ok(Some(output)) => {
                        log::trace!("Processing bedrock stream event: {output:?}");
                        Ok(converse_stream_output_to_stream_event(output))
                    }
                    Ok(None) => {
                        log::trace!("running set_finished on stream due to None event received");
                        self.set_finished();
                        Ok(None)
                    }
                    Err(error) => {
                        log::trace!("running set_finished on stream due to error: {error:?}");
                        self.set_finished();
                        Err(custom_error(
                            llm::ErrorCode::InternalError,
                            format!("An error occurred while reading event stream: {error}"),
                        ))
                    }
                }
            })
        } else if let Some(error) = self.failure() {
            self.set_finished();
            Err(error.clone())
        } else {
            Ok(None)
        }
    }
}

impl llm::GuestChatStream for BedrockChatStream {
    fn poll_next(&self) -> Result<Option<Vec<llm::StreamEvent>>, llm::Error> {
        if self.is_finished() {
            return Ok(Some(vec![]));
        }

        let event = self.get_single_event()?;
        match event {
            Some(event) => {
                if let llm::StreamEvent::Finish(metadata) = event.clone() {
                    if let Some(llm::StreamEvent::Finish(final_metadata)) =
                        self.get_single_event()?
                    {
                        return Ok(Some(vec![llm::StreamEvent::Finish(merge_metadata(
                            metadata,
                            final_metadata,
                        ))]));
                    }
                }
                Ok(Some(vec![event]))
            }
            None => Ok(None),
        }
    }

    fn get_next(&self) -> Result<Vec<llm::StreamEvent>, llm::Error> {
        loop {
            if let Some(events) = self.poll_next()? {
                return Ok(events);
            }
        }
    }
}
