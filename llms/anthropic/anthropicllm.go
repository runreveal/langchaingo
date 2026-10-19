package anthropic

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/httputil"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic/internal/anthropicclient"
	"github.com/tmc/langchaingo/llms/modelcatalog"
)

var (
	ErrEmptyResponse = errors.New("no response")
	ErrMissingToken  = errors.New(
		"missing the Anthropic API key, set it in the ANTHROPIC_API_KEY environment variable",
	)
	ErrUnexpectedResponseLength = errors.New("unexpected length of response")
	ErrInvalidContentType       = errors.New("invalid content type")
	ErrUnsupportedMessageType   = errors.New("unsupported message type")
	ErrUnsupportedContentType   = errors.New("unsupported content type")
)

const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleSystem    = "system"
)

type LLM struct {
	CallbacksHandler callbacks.Handler
	client           *anthropicclient.Client
	model            string // Track current model for reasoning detection
}

var (
	_ llms.Model          = (*LLM)(nil)
	_ llms.ReasoningModel = (*LLM)(nil)
)

// New returns a new Anthropic LLM.
func New(opts ...Option) (*LLM, error) {
	c, err := newClient(opts...)
	if err != nil {
		return nil, fmt.Errorf("anthropic: failed to create client: %w", err)
	}
	return &LLM{
		client: c,
		model:  c.Model, // Store the model for reasoning detection
	}, nil
}

func newClient(opts ...Option) (*anthropicclient.Client, error) {
	options := &options{
		token:      os.Getenv(tokenEnvVarName),
		baseURL:    anthropicclient.DefaultBaseURL,
		httpClient: httputil.DefaultClient,
	}

	for _, opt := range opts {
		opt(options)
	}

	if len(options.token) == 0 {
		return nil, ErrMissingToken
	}

	return anthropicclient.New(options.token, options.model, options.baseURL,
		anthropicclient.WithHTTPClient(options.httpClient),
		anthropicclient.WithLegacyTextCompletionsAPI(options.useLegacyTextCompletionsAPI),
		anthropicclient.WithAnthropicBetaHeader(options.anthropicBetaHeader),
	)
}

// Call requests a completion for the given prompt.
func (o *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, o, prompt, options...)
}

// GenerateContent implements the Model interface.
func (o *LLM) GenerateContent(
	ctx context.Context,
	messages []llms.MessageContent,
	options ...llms.CallOption,
) (*llms.ContentResponse, error) {
	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	opts := &llms.CallOptions{}
	for _, opt := range options {
		opt(opts)
	}

	if o.client.UseLegacyTextCompletionsAPI {
		return generateCompletionsContent(ctx, o, messages, opts)
	}
	return generateMessagesContent(ctx, o, messages, opts)
}

func generateCompletionsContent(
	ctx context.Context,
	o *LLM,
	messages []llms.MessageContent,
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	if len(messages) == 0 || len(messages[0].Parts) == 0 {
		return nil, ErrEmptyResponse
	}

	msg0 := messages[0]
	part := msg0.Parts[0]
	partText, ok := part.(llms.TextContent)
	if !ok {
		return nil, fmt.Errorf("anthropic: unexpected message type: %T", part)
	}
	prompt := fmt.Sprintf("\n\nHuman: %s\n\nAssistant:", partText.Text)
	result, err := o.client.CreateCompletion(ctx, &anthropicclient.CompletionRequest{
		Model:         opts.Model,
		Prompt:        prompt,
		MaxTokens:     opts.MaxTokens,
		StopWords:     opts.StopWords,
		Temperature:   opts.Temperature,
		TopP:          opts.TopP,
		StreamingFunc: opts.StreamingFunc,
	})
	if err != nil {
		if o.CallbacksHandler != nil {
			o.CallbacksHandler.HandleLLMError(ctx, err)
		}
		return nil, fmt.Errorf("anthropic: failed to create completion: %w", err)
	}

	resp := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				Content: result.Text,
			},
		},
	}
	return resp, nil
}

func generateMessagesContent(
	ctx context.Context,
	o *LLM,
	messages []llms.MessageContent,
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	chatMessages, systemPrompt, err := processMessages(messages)
	if err != nil {
		return nil, fmt.Errorf("anthropic: failed to process messages: %w", err)
	}

	tools := toolsToTools(opts.Tools)

	betaHeaders, thinking, outputConfig := extractThinkingOptions(o, opts)

	// Claude 4.7 and later reject non-default sampling parameters with a 400.
	temperature, topP := opts.Temperature, opts.TopP
	if model := cmp.Or(opts.Model, o.model); !acceptsSamplingParams(model) {
		temperature, topP = 0, 0
	}

	// Extract compaction configuration from metadata
	var contextMgmt *anthropicclient.ContextManagement
	if opts.Metadata != nil {
		if cfg, ok := opts.Metadata["anthropic:compaction"].(*CompactionConfig); ok && cfg != nil {
			triggerTokens := cfg.TriggerTokens
			if triggerTokens == 0 {
				triggerTokens = 100000
			}
			if triggerTokens < 50000 {
				triggerTokens = 50000 // Anthropic API minimum
			}
			contextMgmt = &anthropicclient.ContextManagement{
				Edits: []anthropicclient.ContextManagementEdit{
					{
						Type: "compact_20260112",
						Trigger: &anthropicclient.ContextManagementTrigger{
							Type:  "input_tokens",
							Value: triggerTokens,
						},
						Instructions:         cfg.Instructions,
						PauseAfterCompaction: cfg.PauseAfterCompaction,
					},
				},
			}
			// Add the compaction beta header
			betaHeaders = append(betaHeaders, "compact-2026-01-12")
		}
	}

	result, err := o.client.CreateMessage(ctx, &anthropicclient.MessageRequest{
		Model:                  opts.Model,
		Messages:               chatMessages,
		System:                 systemPrompt,
		MaxTokens:              opts.MaxTokens,
		StopWords:              opts.StopWords,
		Temperature:            temperature,
		TopP:                   topP,
		Tools:                  tools,
		Thinking:               thinking,
		OutputConfig:           outputConfig,
		ContextManagement:      contextMgmt,
		BetaHeaders:            betaHeaders,
		StreamingFunc:          opts.StreamingFunc,
		StreamingReasoningFunc: opts.StreamingReasoningFunc,
	})
	if err != nil {
		if o.CallbacksHandler != nil {
			o.CallbacksHandler.HandleLLMError(ctx, err)
		}
		// Classify before returning so callers can branch on llms.ErrorCode
		// (retry a rate limit, fail fast on a bad key) instead of matching text.
		return nil, MapError(fmt.Errorf("anthropic: failed to create message: %w", err))
	}
	return processAnthropicResponse(result)
}

// StopReasonRefusal is the stop_reason Anthropic returns when a safety guardrail
// blocks a request. It matches the value surfaced on ContentChoice.StopReason.
const StopReasonRefusal = "refusal"

// defaultRefusalMessage is used when a refusal arrives without an explanation.
const defaultRefusalMessage = "This request was blocked by the model's safety guardrails."

// refusalChoice builds a ContentChoice from a blocked (empty-content) response.
// It returns nil when the response is empty for some reason other than a refusal,
// so the caller can fall back to ErrEmptyResponse.
func refusalChoice(result *anthropicclient.MessageResponsePayload) *llms.ContentChoice {
	isRefusal := result.StopReason == StopReasonRefusal ||
		(result.StopDetails != nil && result.StopDetails.Type == StopReasonRefusal)
	if !isRefusal {
		return nil
	}

	message := defaultRefusalMessage
	genInfo := map[string]any{
		"InputTokens":  result.Usage.InputTokens,
		"OutputTokens": result.Usage.OutputTokens,
	}
	if result.StopDetails != nil {
		if result.StopDetails.Explanation != "" {
			message = result.StopDetails.Explanation
		}
		genInfo["RefusalCategory"] = result.StopDetails.Category
	}

	return &llms.ContentChoice{
		Content:        message,
		StopReason:     StopReasonRefusal,
		GenerationInfo: genInfo,
	}
}

// processAnthropicResponse converts Anthropic API response to standard ContentResponse
func processAnthropicResponse(result *anthropicclient.MessageResponsePayload) (*llms.ContentResponse, error) {
	if result == nil {
		return nil, ErrEmptyResponse
	}

	// A safety guardrail (Opus 5 / Fable 5 and newer) blocks the request with an
	// HTTP 200 whose content array is empty and stop_reason is "refusal", carrying
	// the human-readable reason in stop_details. Surface it as a normal choice with
	// the explanation as content: the caller sees why it was blocked, the StopReason
	// lets it decide whether to fall back to another model, and — critically — the
	// assistant turn is non-empty, so it does not later fail handleAIMessage when
	// replayed as history ("no valid content in AI message").
	if len(result.Content) == 0 {
		if choice := refusalChoice(result); choice != nil {
			return &llms.ContentResponse{Choices: []*llms.ContentChoice{choice}}, nil
		}
		return nil, ErrEmptyResponse
	}

	// Anthropic content blocks are ordered parts of one assistant message, not
	// alternative completions. Represent the message as one choice so callers do
	// not lose text when thinking or tool-use blocks precede it.
	choice := &llms.ContentChoice{
		StopReason: result.StopReason,
		GenerationInfo: map[string]any{
			"InputTokens":              result.Usage.InputTokens,
			"OutputTokens":             result.Usage.OutputTokens,
			"CacheCreationInputTokens": result.Usage.CacheCreationInputTokens,
			"CacheReadInputTokens":     result.Usage.CacheReadInputTokens,
		},
	}
	var text, thinking, output strings.Builder
	var thinkingSignature string

	for _, content := range result.Content {
		switch content.GetType() {
		case "text":
			if textContent, ok := content.(*anthropicclient.TextContent); ok {
				// Extract thinking content from the response text
				thinkingContent, outputContent := extractThinkingFromText(textContent.Text)
				text.WriteString(textContent.Text)
				thinking.WriteString(thinkingContent)
				output.WriteString(outputContent)
			} else {
				return nil, fmt.Errorf("anthropic: %w for text message", ErrInvalidContentType)
			}
		case "tool_use":
			if toolUseContent, ok := content.(*anthropicclient.ToolUseContent); ok {
				argumentsJSON, err := json.Marshal(toolUseContent.Input)
				if err != nil {
					return nil, fmt.Errorf("anthropic: failed to marshal tool use arguments: %w", err)
				}
				choice.ToolCalls = append(choice.ToolCalls, llms.ToolCall{
					ID: toolUseContent.ID,
					FunctionCall: &llms.FunctionCall{
						Name:      toolUseContent.Name,
						Arguments: string(argumentsJSON),
					},
				})
			} else {
				return nil, fmt.Errorf("anthropic: %w for tool use message %T", ErrInvalidContentType, content)
			}
		case "thinking":
			if thinkingContent, ok := content.(*anthropicclient.ThinkingContent); ok {
				thinking.WriteString(thinkingContent.Thinking)
				if thinkingContent.Signature != "" {
					thinkingSignature = thinkingContent.Signature
				}
			} else {
				return nil, fmt.Errorf("anthropic: %w for thinking message %T", ErrInvalidContentType, content)
			}
		case "compaction":
			if compactionContent, ok := content.(*anthropicclient.CompactionContent); ok {
				text.WriteString(compactionContent.Content)
				choice.GenerationInfo["CompactionSummary"] = compactionContent.Content
			} else {
				return nil, fmt.Errorf("anthropic: %w for compaction message %T", ErrInvalidContentType, content)
			}
		default:
			return nil, fmt.Errorf("anthropic: %w: %v", ErrUnsupportedContentType, content.GetType())
		}
	}
	choice.Content = text.String()
	choice.GenerationInfo["ThinkingContent"] = thinking.String()
	choice.GenerationInfo["OutputContent"] = output.String()
	if thinkingSignature != "" {
		choice.GenerationInfo["ThinkingSignature"] = thinkingSignature
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{choice},
	}, nil
}

func toolsToTools(tools []llms.Tool) []anthropicclient.Tool {
	toolReq := make([]anthropicclient.Tool, len(tools))
	for i, tool := range tools {
		toolReq[i] = anthropicclient.Tool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: tool.Function.Parameters,
		}
	}
	return toolReq
}

// processMessages converts the message slice into Anthropic chat messages plus a
// system prompt. The system value is nil when there is no system content; otherwise
// it carries the text and, when a cache breakpoint was attached to the system
// message, the cache_control block (emitted as the array form on the wire).
func processMessages(messages []llms.MessageContent) ([]anthropicclient.ChatMessage, *anthropicclient.SystemPrompt, error) {
	chatMessages := make([]anthropicclient.ChatMessage, 0, len(messages))
	var systemText strings.Builder
	var systemBlocks []anthropicclient.TextContent
	systemCached := false
	for _, msg := range messages {
		switch msg.Role {
		case llms.ChatMessageTypeSystem:
			content, cacheControl, err := handleSystemMessage(msg)
			if err != nil {
				return nil, nil, fmt.Errorf("anthropic: failed to handle system message: %w", err)
			}
			systemText.WriteString(content)
			block := anthropicclient.TextContent{Type: "text", Text: content}
			if cacheControl != nil {
				block.CacheControl = cacheControl
				systemCached = true
			}
			systemBlocks = append(systemBlocks, block)
		case llms.ChatMessageTypeHuman:
			chatMessage, err := handleHumanMessage(msg)
			if err != nil {
				return nil, nil, fmt.Errorf("anthropic: failed to handle human message: %w", err)
			}
			chatMessages = append(chatMessages, chatMessage)
		case llms.ChatMessageTypeAI:
			chatMessage, err := handleAIMessage(msg)
			if err != nil {
				return nil, nil, fmt.Errorf("anthropic: failed to handle AI message: %w", err)
			}
			chatMessages = append(chatMessages, chatMessage)
		case llms.ChatMessageTypeTool:
			chatMessage, err := handleToolMessage(msg)
			if err != nil {
				return nil, nil, fmt.Errorf("anthropic: failed to handle tool message: %w", err)
			}
			chatMessages = append(chatMessages, chatMessage)
		case llms.ChatMessageTypeGeneric, llms.ChatMessageTypeFunction:
			return nil, nil, fmt.Errorf("anthropic: %w: %v", ErrUnsupportedMessageType, msg.Role)
		default:
			return nil, nil, fmt.Errorf("anthropic: %w: %v", ErrUnsupportedMessageType, msg.Role)
		}
	}

	// Default to the plain-string form; only carry the block array (with
	// cache_control) when a cache breakpoint is present, so the static prefix is
	// served from cache on subsequent turns. nil omits the system field entirely.
	var system *anthropicclient.SystemPrompt
	switch {
	case systemCached:
		system = &anthropicclient.SystemPrompt{Blocks: systemBlocks}
	case systemText.Len() > 0:
		system = &anthropicclient.SystemPrompt{Text: systemText.String()}
	}
	return chatMessages, system, nil
}

func handleSystemMessage(msg llms.MessageContent) (string, *anthropicclient.CacheControl, error) {
	// Handle both direct TextContent and CachedContent wrapper
	part := msg.Parts[0]

	// If it's cached content, unwrap it and carry the cache control through so the
	// system prompt can be emitted as a cache breakpoint.
	var cacheControl *anthropicclient.CacheControl
	if cached, ok := part.(llms.CachedContent); ok {
		if cached.CacheControl != nil {
			cacheControl = &anthropicclient.CacheControl{Type: cached.CacheControl.Type}
		}
		part = cached.ContentPart
	}

	// Extract text from the part
	if textContent, ok := part.(llms.TextContent); ok {
		return textContent.Text, cacheControl, nil
	}

	return "", nil, fmt.Errorf("anthropic: %w for system message", ErrInvalidContentType)
}

func handleHumanMessage(msg llms.MessageContent) (anthropicclient.ChatMessage, error) {
	var contents []anthropicclient.Content

	for _, part := range msg.Parts {
		switch p := part.(type) {
		case llms.CachedContent:
			// Handle cached content with cache control
			var cacheControl *anthropicclient.CacheControl
			if p.CacheControl != nil {
				cacheControl = &anthropicclient.CacheControl{
					Type: p.CacheControl.Type,
				}
			}

			// Process the wrapped content
			switch wrapped := p.ContentPart.(type) {
			case llms.TextContent:
				contents = append(contents, &anthropicclient.TextContent{
					Type:         "text",
					Text:         wrapped.Text,
					CacheControl: cacheControl,
				})
			case llms.BinaryContent:
				contents = append(contents, &anthropicclient.ImageContent{
					Type: "image",
					Source: anthropicclient.ImageSource{
						Type:      "base64",
						MediaType: wrapped.MIMEType,
						Data:      base64.StdEncoding.EncodeToString(wrapped.Data),
					},
					CacheControl: cacheControl,
				})
			default:
				return anthropicclient.ChatMessage{}, fmt.Errorf("anthropic: unsupported cached content part type: %T", wrapped)
			}
		case llms.TextContent:
			contents = append(contents, &anthropicclient.TextContent{
				Type: "text",
				Text: p.Text,
			})
		case llms.BinaryContent:
			contents = append(contents, &anthropicclient.ImageContent{
				Type: "image",
				Source: anthropicclient.ImageSource{
					Type:      "base64",
					MediaType: p.MIMEType,
					Data:      base64.StdEncoding.EncodeToString(p.Data),
				},
			})
		default:
			return anthropicclient.ChatMessage{}, fmt.Errorf("anthropic: unsupported human message part type: %T", part)
		}
	}

	if len(contents) == 0 {
		return anthropicclient.ChatMessage{}, fmt.Errorf("anthropic: no valid content in human message")
	}

	return anthropicclient.ChatMessage{
		Role:    RoleUser,
		Content: contents,
	}, nil
}

func handleAIMessage(msg llms.MessageContent) (anthropicclient.ChatMessage, error) {
	var contents []anthropicclient.Content

	for _, part := range msg.Parts {
		switch p := part.(type) {
		case llms.ToolCall:
			var inputStruct map[string]interface{}
			if err := json.Unmarshal([]byte(p.FunctionCall.Arguments), &inputStruct); err != nil {
				return anthropicclient.ChatMessage{}, fmt.Errorf("anthropic: failed to unmarshal tool call arguments: %w", err)
			}
			contents = append(contents, anthropicclient.ToolUseContent{
				Type:  "tool_use",
				ID:    p.ID,
				Name:  p.FunctionCall.Name,
				Input: inputStruct,
			})
		case llms.TextContent:
			// Skip empty text blocks: Claude can return a tool_use alongside an
			// empty preamble, and the API rejects empty text content blocks.
			if p.Text != "" {
				contents = append(contents, &anthropicclient.TextContent{
					Type: "text",
					Text: p.Text,
				})
			}
		default:
			return anthropicclient.ChatMessage{}, fmt.Errorf("anthropic: %w for AI message part type %T", ErrInvalidContentType, part)
		}
	}

	if len(contents) == 0 {
		// An assistant turn with no usable content reaches here when a prior
		// generation was blocked by a guardrail (empty content, stop_reason
		// "refusal") and persisted to history. The API rejects empty text blocks,
		// so substitute a placeholder rather than failing the whole request — one
		// blocked turn must not make the entire conversation unreplayable.
		contents = append(contents, &anthropicclient.TextContent{
			Type: "text",
			Text: defaultRefusalMessage,
		})
	}

	return anthropicclient.ChatMessage{
		Role:    RoleAssistant,
		Content: contents,
	}, nil
}

type ToolResult struct {
	Type      string `json:"type"`
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
}

func handleToolMessage(msg llms.MessageContent) (anthropicclient.ChatMessage, error) {
	if toolCallResponse, ok := msg.Parts[0].(llms.ToolCallResponse); ok {
		toolContent := anthropicclient.ToolResultContent{
			Type:      "tool_result",
			ToolUseID: toolCallResponse.ToolCallID,
			Content:   toolCallResponse.Content,
		}

		return anthropicclient.ChatMessage{
			Role:    RoleUser,
			Content: []anthropicclient.Content{toolContent},
		}, nil
	}
	return anthropicclient.ChatMessage{}, fmt.Errorf("anthropic: %w for tool message", ErrInvalidContentType)
}

// SupportsReasoning implements the ReasoningModel interface.
// Returns true if the current model supports extended thinking capabilities.
func (o *LLM) SupportsReasoning() bool {
	return supportsReasoningForModel(o.model)
}

// supportsReasoningForModel checks if a specific model supports reasoning.
// This is a separate function to avoid race conditions when checking capabilities.
func supportsReasoningForModel(model string) bool {
	if model == "" {
		return false
	}
	if spec, ok := modelcatalog.Lookup(model); ok {
		return spec.Reasoning.Style != modelcatalog.ReasoningNone
	}

	// Ids the model catalog does not know (new releases, custom names) keep the
	// name-based guess.
	modelLower := strings.ToLower(model)
	return strings.Contains(modelLower, "claude-3-7") ||
		strings.Contains(modelLower, "claude-3.7") ||
		strings.Contains(modelLower, "claude-4") ||
		strings.Contains(modelLower, "claude-opus-4") ||
		strings.Contains(modelLower, "claude-sonnet-4") ||
		strings.Contains(modelLower, "claude-5") ||
		strings.Contains(modelLower, "claude-opus-5") ||
		strings.Contains(modelLower, "claude-sonnet-5") ||
		strings.Contains(modelLower, "claude-haiku-5") ||
		strings.Contains(modelLower, "claude-fable-5") ||
		strings.Contains(modelLower, "claude-mythos-5")
}

// reasoningStyleForModel reports how model takes a thinking request. Unknown
// reasoning models default to budget_tokens, the long-standing behaviour.
func reasoningStyleForModel(model string) modelcatalog.ReasoningStyle {
	if spec, ok := modelcatalog.Lookup(model); ok {
		return spec.Reasoning.Style
	}
	if supportsReasoningForModel(model) {
		return modelcatalog.ReasoningBudgetTokens
	}
	return modelcatalog.ReasoningNone
}

// acceptsSamplingParams reports whether model accepts a non-default
// temperature, top_p or top_k. Claude 4.7 and later reject them with a 400.
func acceptsSamplingParams(model string) bool {
	if spec, ok := modelcatalog.Lookup(model); ok {
		return spec.Temperature
	}
	return true
}

// extractThinkingOptions extracts the beta headers, thinking configuration and
// output configuration from call options. Budget-style models get
// thinking.budget_tokens; adaptive models (Claude 4.6 and later) get
// thinking.type=adaptive with the mode mapped to output_config.effort, because
// 4.7 and later reject budget_tokens.
func extractThinkingOptions(
	o *LLM,
	opts *llms.CallOptions,
) ([]string, *anthropicclient.ThinkingConfig, *anthropicclient.OutputConfig) {
	// Extract beta headers for prompt caching support
	var betaHeaders []string
	if opts.Metadata != nil {
		if headers, ok := opts.Metadata["anthropic:beta_headers"].([]string); ok {
			betaHeaders = headers
		}
	}

	config, ok := opts.Metadata["thinking_config"].(*llms.ThinkingConfig)
	if !ok || config == nil {
		return betaHeaders, nil, nil
	}
	currentModel := opts.Model
	if currentModel == "" {
		currentModel = o.model
	}

	switch reasoningStyleForModel(currentModel) {
	case modelcatalog.ReasoningAdaptive:
		thinking, output := adaptiveThinking(currentModel, config)
		// Adaptive thinking interleaves on its own; the beta header is not needed.
		return betaHeaders, thinking, output
	case modelcatalog.ReasoningBudgetTokens:
		var budgetTokens int
		if config.BudgetTokens > 0 {
			budgetTokens = config.BudgetTokens
		} else if config.Mode != llms.ThinkingModeNone {
			// Calculate budget based on mode
			budgetTokens = llms.CalculateThinkingBudget(config.Mode, opts.MaxTokens)
		}

		// Add interleaved thinking header if requested (Claude 4+)
		if config.InterleaveThinking {
			betaHeaders = append(betaHeaders, "interleaved-thinking-2025-05-14")
		}
		if budgetTokens <= 0 {
			return betaHeaders, nil, nil
		}
		// Ensure budget is within valid range for Claude 3.7+
		budgetTokens = min(max(budgetTokens, 1024), 128000)
		return betaHeaders, &anthropicclient.ThinkingConfig{
			Type:         "enabled",
			BudgetTokens: budgetTokens,
		}, nil
	default:
		return betaHeaders, nil, nil
	}
}

// adaptiveThinking builds the thinking and output configuration for an
// adaptive-thinking model. ThinkingModeNone sends nothing: several of these
// models reject thinking.type=disabled, and on the rest omitting thinking is
// how it stays off. An explicit token budget has no adaptive equivalent, so it
// only turns thinking on.
func adaptiveThinking(
	model string,
	config *llms.ThinkingConfig,
) (*anthropicclient.ThinkingConfig, *anthropicclient.OutputConfig) {
	var effort string
	switch config.Mode {
	case llms.ThinkingModeLow, llms.ThinkingModeMedium, llms.ThinkingModeHigh:
		effort = string(config.Mode)
	case llms.ThinkingModeAuto:
	default:
		if config.BudgetTokens <= 0 {
			return nil, nil
		}
	}

	thinking := &anthropicclient.ThinkingConfig{Type: "adaptive"}
	if spec, ok := modelcatalog.Lookup(model); ok && effort != "" && spec.Reasoning.SupportsEffort(effort) {
		return thinking, &anthropicclient.OutputConfig{Effort: effort}
	}
	return thinking, nil
}

// extractThinkingFromText extracts thinking content from Anthropic responses
// Anthropic models often embed thinking in <thinking> tags
func extractThinkingFromText(fullText string) (thinkingContent, outputContent string) {
	// Look for <thinking> tags in the text
	if strings.Contains(fullText, "<thinking>") {
		start := strings.Index(fullText, "<thinking>")
		end := strings.Index(fullText, "</thinking>")
		if start >= 0 && end > start {
			// Extract thinking content between tags
			thinkingContent = fullText[start+10 : end] // +10 for "<thinking>"

			// Extract output content (everything before and after thinking tags)
			beforeThinking := strings.TrimSpace(fullText[:start])
			afterThinking := ""
			if end+12 < len(fullText) { // +12 for "</thinking>"
				afterThinking = strings.TrimSpace(fullText[end+12:])
			}

			// Combine non-thinking content
			if beforeThinking != "" && afterThinking != "" {
				outputContent = beforeThinking + "\n\n" + afterThinking
			} else if beforeThinking != "" {
				outputContent = beforeThinking
			} else {
				outputContent = afterThinking
			}

			return strings.TrimSpace(thinkingContent), strings.TrimSpace(outputContent)
		}
	}

	// If no thinking tags found, treat entire text as output
	return "", fullText
}
