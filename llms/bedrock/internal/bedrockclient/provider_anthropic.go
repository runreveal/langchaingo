package bedrockclient

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/tmc/langchaingo/llms"
)

// Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
// Also: https://docs.anthropic.com/claude/reference/messages_post

// anthropicBinGenerationInputSource is the source of the content.
type anthropicBinGenerationInputSource struct {
	// The type of the source. Required
	// One of: "base64"
	Type string `json:"type"`
	// The MIME type of the source. Required
	// One of: []"image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp"]
	MediaType string `json:"media_type"`
	// The data of the source. Required
	// For example if type is "base64" then data is a base64 encoded string
	Data string `json:"data"`
}

// anthropicTextGenerationInputContent is a single message in the input.
type anthropicTextGenerationInputContent struct {
	// The type of the content. Required.
	// One of: "text", "image", "tool_result", "tool_use"
	Type string `json:"type"`
	// The source of the content. Required if type is "image"
	Source *anthropicBinGenerationInputSource `json:"source,omitempty"`
	// The text content. Required if type is "text"
	Text string `json:"text,omitempty"`
	// Tool result fields
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
	// Tool use fields (for tool calls from AI)
	ID    string      `json:"id,omitempty"`
	Name  string      `json:"name,omitempty"`
	Input interface{} `json:"input,omitempty"`
	// CacheControl marks this block as the end of a cacheable prompt prefix.
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

// anthropicCacheControl is a prompt-cache breakpoint. TTL is omitted for the
// default 5-minute cache: Bedrock rejects the field on models that only offer it.
type anthropicCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// anthropicSystemPrompt is the system field. It is sent as a plain string unless
// it carries a cache breakpoint, which only the block-array form can express.
type anthropicSystemPrompt struct {
	Text         string
	CacheControl *anthropicCacheControl
}

func (s anthropicSystemPrompt) MarshalJSON() ([]byte, error) {
	if s.CacheControl == nil {
		return json.Marshal(s.Text)
	}
	return json.Marshal([]anthropicTextGenerationInputContent{{
		Type:         AnthropicMessageTypeText,
		Text:         s.Text,
		CacheControl: s.CacheControl,
	}})
}

// anthropicContextManagement configures server-side context compaction.
type anthropicContextManagement struct {
	Edits []anthropicContextManagementEdit `json:"edits"`
}

type anthropicContextManagementEdit struct {
	Type                 string                             `json:"type"`
	Trigger              *anthropicContextManagementTrigger `json:"trigger,omitempty"`
	Instructions         string                             `json:"instructions,omitempty"`
	PauseAfterCompaction bool                               `json:"pause_after_compaction,omitempty"`
}

type anthropicContextManagementTrigger struct {
	Type  string `json:"type"`
	Value int    `json:"value"`
}

type anthropicTextGenerationInputMessage struct {
	// The role of the message. Required
	// One of: ["user", "assistant"]
	// For system prompt, use the system field in the input
	Role string `json:"role"`
	// The content of the message. Required
	Content []anthropicTextGenerationInputContent `json:"content"`
}

// anthropicTextGenerationInput is the input to the model.
type anthropicTextGenerationInput struct {
	// The version of the model to use. Required
	AnthropicVersion string `json:"anthropic_version"`
	// The maximum number of tokens to generate per result. Required
	MaxTokens int `json:"max_tokens"`
	// Beta features to enable, e.g. "compact-2026-01-12". Optional
	AnthropicBeta []string `json:"anthropic_beta,omitempty"`
	// The system prompt to use. Optional
	System *anthropicSystemPrompt `json:"system,omitempty"`
	// The messages to use. Required
	Messages []*anthropicTextGenerationInputMessage `json:"messages"`
	// The amount of randomness injected into the response. Optional, default = 1
	Temperature float64 `json:"temperature,omitempty"`
	// The probability mass from which tokens are sampled. Optional, default = 1
	TopP float64 `json:"top_p,omitempty"`
	// Only sample from the top K options for each subsequent token.
	// Use top_k to remove long tail low probability responses.
	// Optional, default = 250
	TopK int `json:"top_k,omitempty"`
	// Sequences that will cause the model to stop generating tokens. Optional
	StopSequences []string `json:"stop_sequences,omitempty"`
	// Tools available to the model. Optional
	Tools []BedrockTool `json:"tools,omitempty"`
	// Tool choice configuration. Optional
	ToolChoice *BedrockToolChoice `json:"tool_choice,omitempty"`
	// Server-side context management (compaction). Optional
	ContextManagement *anthropicContextManagement `json:"context_management,omitempty"`
}

// anthropicTextGenerationOutput is the generated output.
type anthropicTextGenerationOutput struct {
	// Type of the content.
	// For messages, it is "message"
	Type string `json:"type"`
	// Conversational role of the generated message.
	// This will always be "assistant".
	Role string `json:"role"`
	// This is an array of content blocks, each of which has a type that determines its shape.
	// Can be "text" or "tool_use"
	Content []anthropicContentBlock `json:"content"`
	// The reason for the completion of the generation.
	// One of: ["end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal"]
	StopReason string `json:"stop_reason"`
	// Which custom stop sequence was matched, if any.
	StopSequence string `json:"stop_sequence"`
	// Present when a safety guardrail blocks the request (stop_reason "refusal").
	StopDetails *anthropicStopDetails `json:"stop_details,omitempty"`
	Usage       anthropicUsage        `json:"usage"`
}

// anthropicUsage is the token accounting for a response. InputTokens excludes
// the cache partitions, so the full prompt is the sum of all three input fields.
type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	// Iterations is present only when compaction ran during the request; the
	// top-level counts then exclude the compaction pass.
	Iterations []anthropicUsageIteration `json:"iterations,omitempty"`
}

type anthropicUsageIteration struct {
	Type         string `json:"type"`
	InputTokens  int    `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
}

// generationInfo returns the usage fields in the keys both the legacy Bedrock
// callers (snake_case) and the direct Anthropic provider (PascalCase) use.
func (u anthropicUsage) generationInfo() map[string]any {
	info := map[string]any{
		"input_tokens":             u.InputTokens,
		"output_tokens":            u.OutputTokens,
		"CacheCreationInputTokens": u.CacheCreationInputTokens,
		"CacheReadInputTokens":     u.CacheReadInputTokens,
	}
	for _, it := range u.Iterations {
		if it.Type == AnthropicMessageTypeCompaction {
			info["CompactionInputTokens"] = it.InputTokens
			info["CompactionOutputTokens"] = it.OutputTokens
		}
	}
	return info
}

// anthropicStopDetails carries the structured reason a generation stopped. Newer
// Claude models (Opus 5, Fable 5) populate it when a request is blocked by a
// safety guardrail: Type is "refusal", Category names the policy area, and
// Explanation is human-readable text suitable for showing to the user.
type anthropicStopDetails struct {
	Type        string `json:"type"`
	Category    string `json:"category"`
	Explanation string `json:"explanation"`
}

// anthropicContentBlock represents a content block in Anthropic response
type anthropicContentBlock struct {
	Type string `json:"type"` // "text", "tool_use" or "compaction"
	Text string `json:"text,omitempty"`
	// Content is the summary of a "compaction" block.
	Content string `json:"content,omitempty"`
	// Tool use fields
	ID    string      `json:"id,omitempty"`
	Name  string      `json:"name,omitempty"`
	Input interface{} `json:"input,omitempty"`
}

// Finish reason for the completion of the generation.
const (
	AnthropicCompletionReasonEndTurn      = "end_turn"
	AnthropicCompletionReasonMaxTokens    = "max_tokens"
	AnthropicCompletionReasonStopSequence = "stop_sequence"
	AnthropicCompletionReasonRefusal      = "refusal"
	AnthropicCompletionReasonCompaction   = "compaction"
)

// defaultRefusalMessage is used when a refusal arrives without an explanation.
const defaultRefusalMessage = "This request was blocked by the model's safety guardrails."

// anthropicRefusalChoice converts a blocked (empty-content) response into a usable
// ContentChoice carrying the refusal explanation, or returns nil when the response
// is empty for some reason other than a refusal.
func anthropicRefusalChoice(stopReason string, details *anthropicStopDetails, inTok, outTok int) *llms.ContentChoice {
	isRefusal := stopReason == AnthropicCompletionReasonRefusal ||
		(details != nil && details.Type == AnthropicCompletionReasonRefusal)
	if !isRefusal {
		return nil
	}
	message := defaultRefusalMessage
	genInfo := map[string]any{"input_tokens": inTok, "output_tokens": outTok}
	if details != nil {
		if details.Explanation != "" {
			message = details.Explanation
		}
		genInfo["refusal_category"] = details.Category
	}
	return &llms.ContentChoice{
		Content:        message,
		StopReason:     AnthropicCompletionReasonRefusal,
		GenerationInfo: genInfo,
	}
}

// The latest version of the model.
const (
	AnthropicLatestVersion = "bedrock-2023-05-31"
)

// Role attribute for the anthropic message.
const (
	AnthropicSystem        = "system"
	AnthropicRoleUser      = "user"
	AnthropicRoleAssistant = "assistant"
)

// Type attribute for the anthropic message.
const (
	AnthropicMessageTypeText       = "text"
	AnthropicMessageTypeImage      = "image"
	AnthropicMessageTypeToolUse    = "tool_use"
	AnthropicMessageTypeToolResult = "tool_result"
	AnthropicMessageTypeCompaction = "compaction"
)

// Beta flag and strategy for threshold-triggered server-side compaction.
// Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-compaction.html
const (
	AnthropicBetaCompaction       = "compact-2026-01-12"
	anthropicCompactionStrategy   = "compact_20260112"
	anthropicMinCompactionTrigger = 50_000
)

// Call-option metadata keys read by the Anthropic provider.
const (
	MetadataKeyCompaction    = "bedrock:compaction"
	MetadataKeyAnthropicBeta = "bedrock:anthropic_beta"
)

// CompactionConfig configures server-side context compaction for Claude models
// on Bedrock (compact_20260112, Claude 4.6+).
type CompactionConfig struct {
	// TriggerTokens is the input-token count that triggers compaction. Zero sends
	// no trigger, so Bedrock applies its own default (150k at the time of writing);
	// values below the 50k minimum are raised to it.
	TriggerTokens int
	// Instructions replaces the default summarization prompt entirely.
	Instructions string
	// PauseAfterCompaction returns with stop_reason "compaction" after the summary
	// is generated, so the caller can rebuild its history before continuing.
	PauseAfterCompaction bool
}

func createAnthropicCompletion(ctx context.Context,
	client *bedrockruntime.Client,
	modelID string,
	messages []Message,
	options llms.CallOptions,
) (*llms.ContentResponse, error) {
	input, err := buildAnthropicInput(messages, options)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}

	if options.StreamingFunc != nil {
		modelInput := &bedrockruntime.InvokeModelWithResponseStreamInput{
			ModelId:     aws.String(modelID),
			Accept:      aws.String("*/*"),
			ContentType: aws.String("application/json"),
			Body:        body,
		}
		return parseStreamingCompletionResponse(ctx, client, modelInput, options)
	}

	modelInput := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		Accept:      aws.String("*/*"),
		ContentType: aws.String("application/json"),
		Body:        body,
	}
	resp, err := client.InvokeModel(ctx, modelInput)
	if err != nil {
		return nil, err
	}

	return parseAnthropicResponse(resp.Body)
}

// buildAnthropicInput assembles the InvokeModel request body from the messages
// and call options.
func buildAnthropicInput(messages []Message, options llms.CallOptions) (anthropicTextGenerationInput, error) {
	inputContents, systemPrompt, err := processInputMessagesAnthropic(messages)
	if err != nil {
		return anthropicTextGenerationInput{}, err
	}

	input := anthropicTextGenerationInput{
		AnthropicVersion: AnthropicLatestVersion,
		MaxTokens:        getMaxTokens(options.MaxTokens, 2048),
		System:           systemPrompt,
		Messages:         inputContents,
		Temperature:      options.Temperature,
		TopP:             options.TopP,
		TopK:             options.TopK,
		StopSequences:    options.StopWords,
	}

	// Add tools if provided
	if len(options.Tools) > 0 {
		bedrockTools, err := convertToolsToBedrockTools(options.Tools)
		if err != nil {
			return anthropicTextGenerationInput{}, fmt.Errorf("failed to convert tools: %w", err)
		}
		input.Tools = bedrockTools

		// Add tool choice if provided
		if options.ToolChoice != nil {
			toolChoice, err := convertToolChoiceToBedrockToolChoice(options.ToolChoice)
			if err != nil {
				return anthropicTextGenerationInput{}, fmt.Errorf("failed to convert tool choice: %w", err)
			}
			input.ToolChoice = toolChoice
		}
	}

	if betas, ok := options.Metadata[MetadataKeyAnthropicBeta].([]string); ok {
		input.AnthropicBeta = appendUnique(input.AnthropicBeta, betas...)
	}
	if cfg, ok := options.Metadata[MetadataKeyCompaction].(*CompactionConfig); ok && cfg != nil {
		input.ContextManagement = compactionContextManagement(cfg)
		input.AnthropicBeta = appendUnique(input.AnthropicBeta, AnthropicBetaCompaction)
	}

	return input, nil
}

// compactionContextManagement converts a CompactionConfig into the request's
// context_management block.
func compactionContextManagement(cfg *CompactionConfig) *anthropicContextManagement {
	edit := anthropicContextManagementEdit{
		Type:                 anthropicCompactionStrategy,
		Instructions:         cfg.Instructions,
		PauseAfterCompaction: cfg.PauseAfterCompaction,
	}
	if cfg.TriggerTokens > 0 {
		edit.Trigger = &anthropicContextManagementTrigger{
			Type:  "input_tokens",
			Value: max(cfg.TriggerTokens, anthropicMinCompactionTrigger),
		}
	}
	return &anthropicContextManagement{Edits: []anthropicContextManagementEdit{edit}}
}

func appendUnique(dst []string, values ...string) []string {
	for _, v := range values {
		if !slices.Contains(dst, v) {
			dst = append(dst, v)
		}
	}
	return dst
}

// parseAnthropicResponse converts an InvokeModel response body into a single
// ContentChoice.
func parseAnthropicResponse(body []byte) (*llms.ContentResponse, error) {
	var output anthropicTextGenerationOutput
	if err := json.Unmarshal(body, &output); err != nil {
		return nil, err
	}

	// A safety guardrail blocks the request with an empty content array and
	// stop_reason "refusal"; surface the explanation as a normal choice instead of
	// an opaque "no results" error (mirrors the direct Anthropic client).
	if len(output.Content) == 0 {
		choice := anthropicRefusalChoice(
			output.StopReason, output.StopDetails, output.Usage.InputTokens, output.Usage.OutputTokens,
		)
		if choice != nil {
			return &llms.ContentResponse{Choices: []*llms.ContentChoice{choice}}, nil
		}
		return nil, errors.New("no results")
	}
	// Any other stop_reason (max_tokens, refusal-with-content, pause_turn, ...) is
	// passed through on the choice below rather than turned into an error — erroring
	// discarded a partial answer on a max_tokens truncation and made every new stop
	// reason a hard failure. The caller decides what to do with StopReason.

	// Process content blocks and build a single ContentChoice
	choice := &llms.ContentChoice{
		StopReason:     output.StopReason,
		GenerationInfo: output.Usage.generationInfo(),
	}

	var textContent string
	var toolCalls []llms.ToolCall

	for _, block := range output.Content {
		switch block.Type {
		case AnthropicMessageTypeText:
			textContent += block.Text
		case AnthropicMessageTypeToolUse:
			toolCall, err := convertBedrockToolCallToLLMToolCall(BedrockToolCall{
				Type:  block.Type,
				ID:    block.ID,
				Name:  block.Name,
				Input: block.Input,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to convert tool call: %w", err)
			}
			toolCalls = append(toolCalls, toolCall)
		case AnthropicMessageTypeCompaction:
			// Same key as the direct Anthropic provider, so callers handle both alike.
			choice.GenerationInfo["CompactionSummary"] = block.Content
		}
	}

	choice.Content = textContent
	choice.ToolCalls = toolCalls

	// Set legacy FuncCall field for backward compatibility
	if len(toolCalls) > 0 {
		choice.FuncCall = toolCalls[0].FunctionCall
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{choice},
	}, nil
}

type streamingCompletionResponseChunk struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		Type         string `json:"type"`
		Text         string `json:"text"`
		StopReason   string `json:"stop_reason"`
		StopSequence any    `json:"stop_sequence"`
	} `json:"delta"`
	AmazonBedrockInvocationMetrics struct {
		InputTokenCount   int `json:"inputTokenCount"`
		OutputTokenCount  int `json:"outputTokenCount"`
		InvocationLatency int `json:"invocationLatency"`
		FirstByteLatency  int `json:"firstByteLatency"`
	} `json:"amazon-bedrock-invocationMetrics"`
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
	Message struct {
		ID           string         `json:"id"`
		Type         string         `json:"type"`
		Role         string         `json:"role"`
		Content      []any          `json:"content"`
		Model        string         `json:"model"`
		StopReason   any            `json:"stop_reason"`
		StopSequence any            `json:"stop_sequence"`
		Usage        anthropicUsage `json:"usage"`
	} `json:"message"`
}

func parseStreamingCompletionResponse(ctx context.Context, client *bedrockruntime.Client, modelInput *bedrockruntime.InvokeModelWithResponseStreamInput, options llms.CallOptions) (*llms.ContentResponse, error) {
	output, err := client.InvokeModelWithResponseStream(ctx, modelInput)
	if err != nil {
		return nil, err
	}
	stream := output.GetStream()
	if stream == nil {
		return nil, errors.New("no stream")
	}
	defer stream.Close()

	contentchoices := []*llms.ContentChoice{{GenerationInfo: map[string]interface{}{}}}
	for e := range stream.Events() {
		if err = stream.Err(); err != nil {
			return nil, err
		}

		if v, ok := e.(*types.ResponseStreamMemberChunk); ok {
			var resp streamingCompletionResponseChunk
			err := json.NewDecoder(bytes.NewReader(v.Value.Bytes)).Decode(&resp)
			if err != nil {
				return nil, err
			}

			switch resp.Type {
			case "message_start":
				contentchoices[0].GenerationInfo["input_tokens"] = resp.Message.Usage.InputTokens
				contentchoices[0].GenerationInfo["CacheCreationInputTokens"] = resp.Message.Usage.CacheCreationInputTokens
				contentchoices[0].GenerationInfo["CacheReadInputTokens"] = resp.Message.Usage.CacheReadInputTokens
			case "content_block_delta":
				if err = options.StreamingFunc(ctx, []byte(resp.Delta.Text)); err != nil {
					return nil, err
				}
				contentchoices[0].Content += resp.Delta.Text
			case "message_delta":
				contentchoices[0].StopReason = resp.Delta.StopReason
				contentchoices[0].GenerationInfo["output_tokens"] = resp.Usage.OutputTokens
			}
		}
	}
	if err = stream.Err(); err != nil {
		return nil, err
	}

	// A streamed refusal arrives as empty content with stop_reason "refusal";
	// substitute the generic message so the choice is not empty. (The streaming
	// delta does not carry stop_details, so no explanation is available here.)
	if choice := contentchoices[0]; choice.Content == "" && len(choice.ToolCalls) == 0 &&
		choice.StopReason == AnthropicCompletionReasonRefusal {
		choice.Content = defaultRefusalMessage
	}

	return &llms.ContentResponse{
		Choices: contentchoices,
	}, nil
}

// process the input messages to anthropic supported input
// returns the input content and system prompt (nil when there is none).
func processInputMessagesAnthropic(
	messages []Message,
) ([]*anthropicTextGenerationInputMessage, *anthropicSystemPrompt, error) {
	chunkedMessages := make([][]Message, 0, len(messages))
	currentChunk := make([]Message, 0, len(messages))
	var lastRole llms.ChatMessageType
	for _, message := range messages {
		if message.Role != lastRole {
			if len(currentChunk) > 0 {
				chunkedMessages = append(chunkedMessages, currentChunk)
			}
			currentChunk = make([]Message, 0, len(messages))
		}
		currentChunk = append(currentChunk, message)
		lastRole = message.Role
	}
	if len(currentChunk) > 0 {
		chunkedMessages = append(chunkedMessages, currentChunk)
	}

	inputContents := make([]*anthropicTextGenerationInputMessage, 0, len(messages))
	var systemPrompt *anthropicSystemPrompt
	for _, chunk := range chunkedMessages {
		role, err := getAnthropicRole(chunk[0].Role)
		if err != nil {
			return nil, nil, err
		}
		if role == AnthropicSystem {
			if systemPrompt != nil {
				return nil, nil, errors.New("multiple system prompts")
			}
			systemPrompt = &anthropicSystemPrompt{}
			for _, message := range chunk {
				c := getAnthropicInputContent(message)
				if c.Type != AnthropicMessageTypeText {
					return nil, nil, errors.New("system prompt must be text")
				}
				systemPrompt.Text += c.Text
				// The system prompt is sent as one block, so a breakpoint on any
				// part of it caches the whole of it.
				if c.CacheControl != nil {
					systemPrompt.CacheControl = c.CacheControl
				}
			}
			continue
		}
		content := make([]anthropicTextGenerationInputContent, 0, len(chunk))
		for _, message := range chunk {
			content = append(content, getAnthropicInputContent(message))
		}
		inputContents = append(inputContents, &anthropicTextGenerationInputMessage{
			Role:    role,
			Content: content,
		})
	}
	return inputContents, systemPrompt, nil
}

// process the role of the message to anthropic supported role.
func getAnthropicRole(role llms.ChatMessageType) (string, error) {
	switch role {
	case llms.ChatMessageTypeSystem:
		return AnthropicSystem, nil

	case llms.ChatMessageTypeAI:
		return AnthropicRoleAssistant, nil

	case llms.ChatMessageTypeGeneric:
		fallthrough
	case llms.ChatMessageTypeHuman:
		return AnthropicRoleUser, nil
	case llms.ChatMessageTypeFunction, llms.ChatMessageTypeTool:
		return AnthropicRoleUser, nil // Tool results are sent as user messages
	default:
		return "", errors.New("role not supported")
	}
}

func getAnthropicInputContent(message Message) anthropicTextGenerationInputContent {
	var c anthropicTextGenerationInputContent
	switch message.Type {
	case AnthropicMessageTypeText:
		c = anthropicTextGenerationInputContent{
			Type: message.Type,
			Text: message.Content,
		}
	case AnthropicMessageTypeImage:
		c = anthropicTextGenerationInputContent{
			Type: message.Type,
			Source: &anthropicBinGenerationInputSource{
				Type:      "base64",
				MediaType: message.MimeType,
				Data:      base64.StdEncoding.EncodeToString([]byte(message.Content)),
			},
		}
	case "tool_result":
		// Handle tool results from tool response messages
		c = anthropicTextGenerationInputContent{
			Type:      "tool_result",
			ToolUseID: message.ToolUseID,
			Content:   message.Content,
		}
	case "tool_call":
		// Handle tool calls from AI messages - convert to tool_use format for Anthropic
		var input interface{}
		if message.ToolArgs != "" {
			// Try to parse the arguments as JSON
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(message.ToolArgs), &args); err == nil {
				input = args
			} else {
				// If parsing fails, wrap in a simple structure
				input = map[string]interface{}{"arguments": message.ToolArgs}
			}
		} else {
			input = map[string]interface{}{}
		}

		c = anthropicTextGenerationInputContent{
			Type:  "tool_use",
			ID:    message.ToolCallID,
			Name:  message.ToolName,
			Input: input,
		}
	}
	c.CacheControl = anthropicCacheControlFrom(message.CacheControl)
	return c
}

// anthropicCacheControlFrom converts a generic cache breakpoint. A duration of an
// hour or more selects the 1-hour TTL; anything shorter uses the default.
func anthropicCacheControlFrom(cc *llms.CacheControl) *anthropicCacheControl {
	if cc == nil {
		return nil
	}
	out := &anthropicCacheControl{Type: cc.Type}
	if out.Type == "" {
		out.Type = "ephemeral"
	}
	if cc.Duration >= time.Hour {
		out.TTL = "1h"
	}
	return out
}
