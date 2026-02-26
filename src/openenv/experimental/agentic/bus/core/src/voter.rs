// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Voter component for agentbus
//!
//! The Voter polls an AgentBus for intentions and voter policies, evaluates
//! each intention for safety using an LLM, and proposes votes back to the bus.
//!
//! This is the Rust equivalent of the Python ExternalLLMVoter.
//!
//! Uses OpenAI-compatible REST API for LLM inference, which works with:
//! - OpenAI API
//! - Local LLMs (ollama, vLLM, etc.)
//! - Any OpenAI-compatible endpoint

use std::rc::Rc;
use std::sync::Arc;

use agent_bus_proto_rust::agent_bus::intention::Intention as IntentionEnum;
use agent_bus_proto_rust::agent_bus::*;
use agentbus_api::AgentBus;
use agentbus_api::environment::Environment;
use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

/// Default system prompt for code safety evaluation
const DEFAULT_SAFETY_PROMPT: &str = r#"You are a code safety evaluator. Analyze the following code and determine if it is safe to execute.

Consider the following:
- Does the code attempt to access sensitive files or directories?
- Does the code attempt to execute system commands that could be harmful?
- Does the code attempt to access network resources in unexpected ways?
- Does the code contain any obvious security vulnerabilities?

Respond with XML in this exact format:
<safe>true</safe> or <safe>false</safe>
<reason>Brief explanation of your decision</reason>
<concerns>Any specific concerns, or "none" if safe</concerns>

Code to analyze:
"#;

/// Error type for Voter operations
#[derive(Error, Debug)]
pub enum VoterError {
    #[error("AgentBus call failed: {0}")]
    FailedAgentBusCall(#[from] anyhow::Error),

    #[error("Unknown payload type encountered at log position {0}")]
    UnknownPayloadType(i64),

    #[error("LLM call failed: {0}")]
    LlmCallFailed(String),
}

/// Default API endpoint (OpenAI-compatible)
pub const DEFAULT_API_ENDPOINT: &str = "https://api.openai.com/v1";

/// Default model name
pub const DEFAULT_MODEL: &str = "claude-sonnet-4-5";

/// Environment variable for API key
pub const API_KEY_ENV_VAR: &str = "LLM_API_KEY";

/// Configuration for the Voter's LLM client
#[derive(Clone)]
pub struct VoterConfig {
    /// Model name (e.g., "gpt-4", "llama-3", "claude-3-opus")
    pub model: String,
    /// OpenAI-compatible API endpoint (e.g., "https://api.openai.com/v1")
    pub api_endpoint: String,
    /// API key for authentication (reads from LLM_API_KEY env var if not set)
    pub api_key: Option<String>,
}

impl Default for VoterConfig {
    fn default() -> Self {
        Self {
            model: std::env::var("LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            api_endpoint: DEFAULT_API_ENDPOINT.to_string(),
            api_key: std::env::var(API_KEY_ENV_VAR).ok(),
        }
    }
}

/// OpenAI Chat Completions API request format
#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// OpenAI Chat Completions API response format
#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

/// Trait for LLM clients - enables testing with mocks
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat_completion(&self, model: &str, prompt: &str) -> Result<String, String>;
}

/// OpenAI-compatible HTTP client for LLM inference
pub struct OpenAIClient {
    http_client: reqwest::Client,
    api_endpoint: String,
    api_key: Option<String>,
}

impl OpenAIClient {
    pub fn new(api_endpoint: String, api_key: Option<String>) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            api_endpoint,
            api_key,
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for OpenAIClient {
    async fn chat_completion(&self, model: &str, prompt: &str) -> Result<String, String> {
        let url = format!("{}/chat/completions", self.api_endpoint);

        let request_body = ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
        };

        let mut request = self.http_client.post(&url).json(&request_body);

        if let Some(ref key) = self.api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(format!("API error {}: {}", status, body));
        }

        let completion: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        completion
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| "No response from LLM".to_string())
    }
}

/// Voter component that polls an AgentBus and proposes votes
///
/// Mirrors the Python ExternalLLMVoter implementation.
pub struct Voter<T: AgentBus, E: Environment> {
    agent_bus: T,
    agent_bus_id: String,
    next_log_position: i64,
    prompt_override: Option<String>,
    environment: Rc<E>,
    llm_client: Option<Arc<dyn LlmClient>>,
    config: VoterConfig,
}

impl<T: AgentBus, E: Environment> Voter<T, E> {
    /// Create a new Voter for the given AgentBus without LLM (placeholder mode)
    pub fn new(agent_bus: T, agent_bus_id: String, environment: Rc<E>) -> Self {
        Self {
            agent_bus,
            agent_bus_id,
            next_log_position: 0,
            prompt_override: None,
            environment,
            llm_client: None,
            config: VoterConfig::default(),
        }
    }

    /// Create a new Voter with LLM client for safety evaluation
    pub fn with_llm_client(
        agent_bus: T,
        agent_bus_id: String,
        environment: Rc<E>,
        llm_client: Arc<dyn LlmClient>,
        config: VoterConfig,
    ) -> Self {
        Self {
            agent_bus,
            agent_bus_id,
            next_log_position: 0,
            prompt_override: None,
            environment,
            llm_client: Some(llm_client),
            config,
        }
    }

    /// Vote on the safety of code using LLM
    async fn vote(&self, code: &str, intention_id: i64) -> Vote {
        let (is_safe, reason) = match &self.llm_client {
            Some(client) => self.evaluate_with_llm(client.as_ref(), code).await,
            None => {
                // Placeholder mode - approve everything
                tracing::debug!(
                    intention_id = intention_id,
                    "No LLM client configured, approving by default"
                );
                (true, "No LLM configured - placeholder approval".to_string())
            }
        };

        Vote {
            abstract_vote: Some(VoteType {
                vote_type: Some(vote_type::VoteType::BooleanVote(is_safe)),
            }),
            intention_id,
            info: Some(VoteInfo {
                vote_info: Some(vote_info::VoteInfo::ExternalLlmVoteInfo(
                    ExternalLlmVoteInfo {
                        reason,
                        model: self.config.model.clone(),
                    },
                )),
            }),
        }
    }

    /// Call the LLM to evaluate code safety
    async fn evaluate_with_llm(&self, client: &dyn LlmClient, code: &str) -> (bool, String) {
        let prompt_text = match &self.prompt_override {
            Some(override_prompt) => {
                format!(
                    "{}\n\nOVERRIDE: {}\n\n{}",
                    DEFAULT_SAFETY_PROMPT, override_prompt, code
                )
            }
            None => format!("{}{}", DEFAULT_SAFETY_PROMPT, code),
        };

        match client
            .chat_completion(&self.config.model, &prompt_text)
            .await
        {
            Ok(text) => {
                tracing::trace!(output = %text, "LLM evaluation output");
                parse_safety_response(&text)
            }
            Err(e) => {
                tracing::error!(error = %e, "LLM call failed, blocking by default");
                (false, format!("LLM call failed: {}", e))
            }
        }
    }

    /// Poll the AgentBus and process new entries
    /// Returns the number of entries processed
    pub async fn poll_and_vote(&mut self) -> Result<usize, VoterError> {
        let payload_types = vec![
            SelectivePollType::Intention as i32,
            SelectivePollType::VoterPolicy as i32,
        ];

        let poll_request = PollRequest {
            agent_bus_id: self.agent_bus_id.clone(),
            start_log_position: self.next_log_position,
            max_entries: 1,
            filter: Some(PayloadTypeFilter { payload_types }),
        };

        let poll_response =
            self.agent_bus.poll(poll_request).await.map_err(|e| {
                VoterError::FailedAgentBusCall(anyhow::anyhow!("Poll failed: {:?}", e))
            })?;

        let entries_count = poll_response.entries.len();

        for entry in poll_response.entries {
            self.process_entry(entry).await?;
        }

        Ok(entries_count)
    }

    /// Run the voter in a loop, continuously polling and voting
    pub async fn run(&mut self, poll_interval: std::time::Duration) -> Result<(), VoterError> {
        tracing::info!(
            agent_bus_id = %self.agent_bus_id,
            model = %self.config.model,
            has_llm_client = self.llm_client.is_some(),
            api_endpoint = %self.config.api_endpoint,
            "Starting Voter service"
        );

        loop {
            let entries_processed = match self.poll_and_vote().await {
                Ok(count) => count,
                Err(VoterError::FailedAgentBusCall(e)) => {
                    tracing::error!("AgentBus call failed: {}", e);
                    let sleep_future = self.environment.sleep(poll_interval * 5);
                    sleep_future.await;
                    continue;
                }
                Err(VoterError::UnknownPayloadType(position)) => {
                    tracing::warn!("Unknown payload type at log position {}", position);
                    self.next_log_position = position + 1;
                    continue;
                }
                Err(VoterError::LlmCallFailed(e)) => {
                    tracing::error!("LLM call failed: {}", e);
                    let sleep_future = self.environment.sleep(poll_interval * 5);
                    sleep_future.await;
                    continue;
                }
            };

            if entries_processed == 0 {
                let sleep_future = self.environment.sleep(poll_interval);
                sleep_future.await;
            }
        }
    }

    /// Process a single BusEntry
    async fn process_entry(&mut self, entry: BusEntry) -> Result<(), VoterError> {
        let log_position = entry.header.as_ref().map(|h| h.log_position).unwrap_or(0);

        if let Some(payload) = entry.payload {
            match payload.payload {
                Some(payload::Payload::Intention(intention)) => {
                    self.process_intention(log_position, intention).await?;
                }
                Some(payload::Payload::VoterPolicy(voter_policy)) => {
                    self.process_voter_policy(voter_policy);
                }
                Some(payload::Payload::Vote(_))
                | Some(payload::Payload::DeciderPolicy(_))
                | Some(payload::Payload::Commit(_))
                | Some(payload::Payload::Abort(_))
                | Some(payload::Payload::Control(_))
                | Some(payload::Payload::InferenceInput(_))
                | Some(payload::Payload::InferenceOutput(_))
                | Some(payload::Payload::ActionOutput(_))
                | Some(payload::Payload::AgentInput(_))
                | Some(payload::Payload::AgentOutput(_)) => {
                    return Err(VoterError::UnknownPayloadType(log_position));
                }
                None => {
                    return Err(VoterError::UnknownPayloadType(log_position));
                }
            }
        }

        self.next_log_position = log_position + 1;
        Ok(())
    }

    /// Process an Intention entry
    async fn process_intention(
        &mut self,
        log_position: i64,
        intention: Intention,
    ) -> Result<(), VoterError> {
        let code = match &intention.intention {
            Some(IntentionEnum::StringIntention(s)) => s.as_str(),
            None => "",
        };

        tracing::debug!(
            log_position = log_position,
            code_len = code.len(),
            "Processing intention"
        );

        let vote_result = self.vote(code, log_position).await;

        let propose_request = ProposeRequest {
            agent_bus_id: self.agent_bus_id.clone(),
            payload: Some(Payload {
                payload: Some(payload::Payload::Vote(vote_result.clone())),
            }),
        };

        self.agent_bus.propose(propose_request).await.map_err(|e| {
            VoterError::FailedAgentBusCall(anyhow::anyhow!("Propose vote failed: {:?}", e))
        })?;

        let is_approved = vote_result
            .abstract_vote
            .as_ref()
            .and_then(|v| v.vote_type.as_ref())
            .map(|vt| matches!(vt, vote_type::VoteType::BooleanVote(true)))
            .unwrap_or(false);

        let reason = vote_result
            .info
            .as_ref()
            .and_then(|i| i.vote_info.as_ref())
            .map(|vi| match vi {
                vote_info::VoteInfo::ExternalLlmVoteInfo(info) => info.reason.as_str(),
            })
            .unwrap_or("No reason");

        if is_approved {
            tracing::info!(log_position = log_position, reason = %reason, "APPROVED");
        } else {
            tracing::warn!(log_position = log_position, reason = %reason, "BLOCKED");
        }

        Ok(())
    }

    /// Process a VoterPolicy entry
    fn process_voter_policy(&mut self, voter_policy: VoterPolicy) {
        if !voter_policy.prompt_override.is_empty() {
            self.prompt_override = Some(voter_policy.prompt_override.clone());
            tracing::info!(
                prompt_override_len = voter_policy.prompt_override.len(),
                "Applied voter policy update"
            );
        }
    }
}

/// Parse the LLM response to extract safety decision
fn parse_safety_response(response: &str) -> (bool, String) {
    // Extract <safe> tag
    let is_safe = if let Some(start) = response.find("<safe>") {
        if let Some(end) = response[start..].find("</safe>") {
            let safe_str = &response[start + 6..start + end];
            safe_str.trim().eq_ignore_ascii_case("true")
        } else {
            false
        }
    } else {
        false
    };

    // Extract <reason> tag
    let reason = if let Some(start) = response.find("<reason>") {
        if let Some(end) = response[start..].find("</reason>") {
            response[start + 8..start + end].trim().to_string()
        } else {
            "Could not parse reason".to_string()
        }
    } else {
        "No reason provided".to_string()
    };

    // Extract <concerns> tag
    let concerns = if let Some(start) = response.find("<concerns>") {
        if let Some(end) = response[start..].find("</concerns>") {
            response[start + 10..start + end].trim().to_string()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let full_reason = if concerns.is_empty() || concerns.eq_ignore_ascii_case("none") {
        reason
    } else {
        format!("{}. Concerns: {}", reason, concerns)
    };

    tracing::debug!(is_safe = is_safe, reason = %full_reason, "Parsed safety response");

    (is_safe, full_reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_safety_response_safe() {
        let response = r#"
<safe>true</safe>
<reason>The code only prints a greeting</reason>
<concerns>none</concerns>
"#;
        let (is_safe, reason) = parse_safety_response(response);
        assert!(is_safe);
        assert!(reason.contains("greeting"));
    }

    #[test]
    fn test_parse_safety_response_unsafe() {
        let response = r#"
<safe>false</safe>
<reason>The code attempts to delete system files</reason>
<concerns>Uses rm -rf on root directory</concerns>
"#;
        let (is_safe, reason) = parse_safety_response(response);
        assert!(!is_safe);
        assert!(reason.contains("delete system files"));
        assert!(reason.contains("rm -rf"));
    }

    #[test]
    fn test_parse_safety_response_malformed() {
        let response = "Some random text without XML tags";
        let (is_safe, reason) = parse_safety_response(response);
        assert!(!is_safe); // Default to unsafe
        assert!(reason.contains("No reason"));
    }

    #[test]
    fn test_voter_error_display() {
        let err = VoterError::UnknownPayloadType(42);
        assert!(err.to_string().contains("42"));

        let err = VoterError::LlmCallFailed("connection timeout".to_string());
        assert!(err.to_string().contains("connection timeout"));
    }

    #[test]
    fn test_voter_config_constants() {
        assert_eq!(DEFAULT_MODEL, "claude-sonnet-4-5");
        assert_eq!(DEFAULT_API_ENDPOINT, "https://api.openai.com/v1");
        assert_eq!(API_KEY_ENV_VAR, "LLM_API_KEY");
    }
}
