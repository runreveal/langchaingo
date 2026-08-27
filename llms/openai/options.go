package openai

import (
	"encoding/json"

	"github.com/tmc/langchaingo/llms"
)

// WithMaxCompletionTokens sets the max_completion_tokens field for token generation.
// This is the recommended way to limit tokens with OpenAI models.
//
// Usage:
//
//	llm.GenerateContent(ctx, messages,
//	    openai.WithMaxCompletionTokens(100),
//	)
//
// Note: While llms.WithMaxTokens() still works for backward compatibility,
// WithMaxCompletionTokens is preferred for clarity when using OpenAI.
func WithMaxCompletionTokens(maxTokens int) llms.CallOption {
	return func(opts *llms.CallOptions) {
		opts.MaxTokens = maxTokens
	}
}

// WithLegacyMaxTokensField forces the use of the max_tokens field instead of max_completion_tokens.
// This is useful when connecting to older OpenAI-compatible inference servers that only
// support the max_tokens field and don't recognize max_completion_tokens.
//
// Usage:
//
//	llm.GenerateContent(ctx, messages,
//	    llms.WithMaxTokens(100),
//	    openai.WithLegacyMaxTokensField(), // Forces use of max_tokens field
//	)
func WithLegacyMaxTokensField() llms.CallOption {
	return func(opts *llms.CallOptions) {
		if opts.Metadata == nil {
			opts.Metadata = make(map[string]interface{})
		}
		opts.Metadata["openai:use_legacy_max_tokens"] = true
	}
}

// Metadata keys used to carry Responses-API settings through llms.CallOptions.
// They are stripped before the request is sent.
const (
	metaAPIFlavor        = "openai:api_flavor"
	metaReasoningEffort  = "openai:reasoning_effort"
	metaReasoningSummary = "openai:reasoning_summary"
	metaReasoningItems   = "openai:reasoning_items"
)

// apiFlavor selects which OpenAI endpoint serves a generation.
type apiFlavor string

const (
	// flavorAuto picks the endpoint from the model, which is the default.
	flavorAuto apiFlavor = ""
	// flavorResponses forces POST /v1/responses.
	flavorResponses apiFlavor = "responses"
	// flavorChatCompletions forces POST /v1/chat/completions.
	flavorChatCompletions apiFlavor = "chat_completions"
)

func setMeta(opts *llms.CallOptions, key string, value any) {
	if opts.Metadata == nil {
		opts.Metadata = make(map[string]interface{})
	}
	opts.Metadata[key] = value
}

// WithResponsesAPI routes the call to /v1/responses.
//
// Use it to reach Responses-only behaviour on a model this package would not
// route there by itself, such as an Azure deployment whose name does not carry a
// model version.
func WithResponsesAPI() llms.CallOption {
	return func(opts *llms.CallOptions) { setMeta(opts, metaAPIFlavor, flavorResponses) }
}

// WithChatCompletionsAPI pins the call to /v1/chat/completions, overriding the
// automatic routing. Use it against OpenAI-compatible servers that do not
// implement /v1/responses.
func WithChatCompletionsAPI() llms.CallOption {
	return func(opts *llms.CallOptions) { setMeta(opts, metaAPIFlavor, flavorChatCompletions) }
}

// WithReasoningEffort sets how much the model reasons before answering. Valid
// values depend on the model; gpt-5.6 accepts "none", "low", "medium", "high",
// "xhigh" and "max".
//
// It applies to the Responses API. On /chat/completions the value is sent as
// reasoning_effort only for models that accept it.
func WithReasoningEffort(effort string) llms.CallOption {
	return func(opts *llms.CallOptions) { setMeta(opts, metaReasoningEffort, effort) }
}

// WithReasoningSummary asks for a natural-language summary of the model's
// reasoning, e.g. "auto" or "detailed". The summary is returned in
// ContentChoice.ReasoningContent, and streamed to any
// llms.WithStreamingReasoningFunc callback. Responses API only.
func WithReasoningSummary(summary string) llms.CallOption {
	return func(opts *llms.CallOptions) { setMeta(opts, metaReasoningSummary, summary) }
}

// WithPriorReasoningItems replays the reasoning Items of an earlier response, so
// the model keeps its train of thought across the turns of a tool-calling loop
// instead of reasoning from scratch each time.
//
// Pass the value that the previous response left in
// GenerationInfo[GenInfoReasoningItems]. Responses API only.
func WithPriorReasoningItems(items []json.RawMessage) llms.CallOption {
	return func(opts *llms.CallOptions) { setMeta(opts, metaReasoningItems, items) }
}

func flavorOption(opts *llms.CallOptions) apiFlavor {
	f, _ := opts.Metadata[metaAPIFlavor].(apiFlavor)
	return f
}

func reasoningEffortOption(opts *llms.CallOptions) string {
	s, _ := opts.Metadata[metaReasoningEffort].(string)
	return s
}

func reasoningSummaryOption(opts *llms.CallOptions) string {
	s, _ := opts.Metadata[metaReasoningSummary].(string)
	return s
}

func priorReasoningItems(opts *llms.CallOptions) []json.RawMessage {
	items, _ := opts.Metadata[metaReasoningItems].([]json.RawMessage)
	return items
}
