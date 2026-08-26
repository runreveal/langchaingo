package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai/internal/openaiclient"
)

// GenerationInfo keys set by the Responses path, in addition to the token
// counts the chat-completions path already reports.
const (
	// GenInfoResponseID is the id of the response, usable as
	// previous_response_id by callers that opt into server-side state.
	GenInfoResponseID = "ResponseID"
	// GenInfoReasoningItems holds the response's reasoning Items as raw JSON.
	// Passing them to WithPriorReasoningItems on the next call preserves the
	// model's reasoning across a tool-calling loop.
	GenInfoReasoningItems = "ReasoningItems"
	// GenInfoReasoningSummary holds the human-readable reasoning summary, when
	// one was requested via WithReasoningSummary.
	GenInfoReasoningSummary = "ReasoningSummary"
)

// generateWithResponses runs a generation against /v1/responses.
//
// The Responses API models a conversation as typed Items rather than a flat
// message list: a tool call and its result are separate Items linked by call_id,
// and the model's reasoning is an Item of its own. Reasoning Items must be
// replayed byte-for-byte to survive validation, so they travel through call
// options as raw JSON rather than being reconstructed from llms types.
func (o *LLM) generateWithResponses(
	ctx context.Context,
	messages []llms.MessageContent,
	opts *llms.CallOptions,
	model string,
) (*llms.ContentResponse, error) {
	instructions, input, err := responsesInputFromMessages(messages, priorReasoningItems(opts))
	if err != nil {
		return nil, err
	}

	req := &openaiclient.ResponsesRequest{
		Model:           model,
		Instructions:    instructions,
		Input:           input,
		ToolChoice:      opts.ToolChoice,
		MaxOutputTokens: opts.MaxTokens,
		TopP:            opts.TopP,
		Metadata:        apiMetadata(opts.Metadata),
		// Keep the exchange stateless: nothing is retained server-side, and
		// reasoning comes back encrypted so it can still be replayed.
		Store:                  false,
		Include:                []string{"reasoning.encrypted_content"},
		StreamingFunc:          opts.StreamingFunc,
		StreamingReasoningFunc: opts.StreamingReasoningFunc,
	}

	if effort := reasoningEffortOption(opts); effort != "" {
		req.Reasoning = &openaiclient.ResponsesReasoning{Effort: effort}
	}
	if summary := reasoningSummaryOption(opts); summary != "" {
		if req.Reasoning == nil {
			req.Reasoning = &openaiclient.ResponsesReasoning{}
		}
		req.Reasoning.Summary = summary
	}

	// Reasoning models reject a non-default temperature, exactly as on the
	// chat-completions path. Note that an unset effort does not mean no
	// reasoning: the server applies its own default, so temperature has to stay
	// off unless reasoning was explicitly turned off.
	if !openaiclient.IsReasoningModel(model) || reasoningExplicitlyOff(req.Reasoning) {
		temp := opts.Temperature
		req.Temperature = &temp
	}

	if opts.JSONMode {
		req.Text = &openaiclient.ResponsesText{Format: ResponseFormatJSON}
	}
	if o.client.ResponseFormat != nil {
		req.Text = &openaiclient.ResponsesText{Format: o.client.ResponseFormat}
	}

	for _, fn := range opts.Functions {
		req.Tools = append(req.Tools, openaiclient.ResponsesTool{
			Type:        "function",
			Name:        fn.Name,
			Description: fn.Description,
			Parameters:  fn.Parameters,
			Strict:      fn.Strict,
		})
	}
	for _, tool := range opts.Tools {
		t, err := responsesToolFromTool(tool)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, t)
	}

	resp, err := o.client.CreateResponse(ctx, req)
	if err != nil {
		return nil, MapError(err)
	}
	return contentResponseFromResponses(resp)
}

// responsesInputFromMessages converts langchaingo's message history into
// Responses input Items, returning the system guidance separately because the
// API takes it as the top-level instructions field.
//
// reasoning holds raw reasoning Items from the immediately preceding response.
// They are emitted before the assistant turn they belong to, which is where the
// API expects them.
func responsesInputFromMessages(
	messages []llms.MessageContent,
	reasoning []json.RawMessage,
) (string, []json.RawMessage, error) {
	var instructions strings.Builder
	input := make([]json.RawMessage, 0, len(messages))

	// Reasoning Items belong immediately before the final assistant turn. Find
	// it so they can be spliced in at the right point.
	lastAI := -1
	for i, mc := range messages {
		if mc.Role == llms.ChatMessageTypeAI {
			lastAI = i
		}
	}

	for i, mc := range messages {
		if i == lastAI && len(reasoning) > 0 {
			input = append(input, reasoning...)
		}

		switch mc.Role {
		case llms.ChatMessageTypeSystem:
			for _, part := range mc.Parts {
				text, ok := part.(llms.TextContent)
				if !ok {
					return "", nil, fmt.Errorf("openai: unsupported part %T in a system message", part)
				}
				if instructions.Len() > 0 {
					instructions.WriteString("\n\n")
				}
				instructions.WriteString(text.Text)
			}
		case llms.ChatMessageTypeHuman, llms.ChatMessageTypeGeneric:
			items, err := responsesMessageItems("user", mc.Parts)
			if err != nil {
				return "", nil, err
			}
			input = append(input, items...)
		case llms.ChatMessageTypeAI:
			items, err := responsesMessageItems("assistant", mc.Parts)
			if err != nil {
				return "", nil, err
			}
			input = append(input, items...)
		case llms.ChatMessageTypeTool, llms.ChatMessageTypeFunction:
			for _, part := range mc.Parts {
				tr, ok := part.(llms.ToolCallResponse)
				if !ok {
					return "", nil, fmt.Errorf("openai: expected ToolCallResponse for role %v, got %T", mc.Role, part)
				}
				item, err := openaiclient.NewResponsesFunctionCallOutputItem(tr.ToolCallID, tr.Content)
				if err != nil {
					return "", nil, err
				}
				input = append(input, item)
			}
		default:
			return "", nil, fmt.Errorf("openai: role %v not supported", mc.Role)
		}
	}

	return instructions.String(), input, nil
}

// responsesMessageItems converts the parts of one message into Items. Tool calls
// become their own function_call Items rather than riding along on the message,
// which is the core structural difference from /chat/completions.
func responsesMessageItems(role string, parts []llms.ContentPart) ([]json.RawMessage, error) {
	var (
		items []json.RawMessage
		text  strings.Builder
	)
	for _, part := range parts {
		switch p := part.(type) {
		case llms.TextContent:
			text.WriteString(p.Text)
		case llms.ToolCall:
			if p.FunctionCall == nil {
				return nil, fmt.Errorf("openai: tool call %q has no function call", p.ID)
			}
			item, err := openaiclient.NewResponsesFunctionCallItem(p.ID, p.FunctionCall.Name, p.FunctionCall.Arguments)
			if err != nil {
				return nil, err
			}
			items = append(items, item)
		default:
			return nil, fmt.Errorf("openai: unsupported content part %T for the responses API", part)
		}
	}
	if text.Len() > 0 {
		item, err := openaiclient.NewResponsesMessageItem(role, text.String())
		if err != nil {
			return nil, err
		}
		// The message precedes any tool calls it triggered.
		items = append([]json.RawMessage{item}, items...)
	}
	return items, nil
}

// responsesToolFromTool converts an llms tool definition into the Responses
// shape, which tags the function inline instead of nesting it.
func responsesToolFromTool(t llms.Tool) (openaiclient.ResponsesTool, error) {
	if t.Type != "function" || t.Function == nil {
		return openaiclient.ResponsesTool{}, fmt.Errorf("openai: unsupported tool type %q", t.Type)
	}
	return openaiclient.ResponsesTool{
		Type:        "function",
		Name:        t.Function.Name,
		Description: t.Function.Description,
		Parameters:  t.Function.Parameters,
		Strict:      t.Function.Strict,
	}, nil
}

// contentResponseFromResponses flattens a response's Items into the single
// ContentChoice that langchaingo callers expect.
func contentResponseFromResponses(resp *openaiclient.ResponsesResponse) (*llms.ContentResponse, error) {
	if len(resp.Output) == 0 {
		return nil, openaiclient.ErrNoResponsesOutput
	}

	choice := &llms.ContentChoice{GenerationInfo: map[string]any{}}

	var (
		answer     strings.Builder
		commentary strings.Builder
		summary    strings.Builder
		reasoning  []json.RawMessage
	)

	for _, item := range resp.Output {
		switch item.Type {
		case openaiclient.ResponsesItemMessage:
			// gpt-5.6 narrates tool use with "commentary" messages. Only the
			// final answer is the content; commentary would otherwise be
			// interleaved into it as if the model had said it twice.
			if item.Phase != "" && item.Phase != openaiclient.ResponsesPhaseFinalAnswer {
				commentary.WriteString(item.Text())
				continue
			}
			answer.WriteString(item.Text())
		case openaiclient.ResponsesItemReasoning:
			reasoning = append(reasoning, item.Raw)
			summary.WriteString(item.SummaryText())
		case openaiclient.ResponsesItemFunctionCall:
			choice.ToolCalls = append(choice.ToolCalls, llms.ToolCall{
				ID:   item.CallID,
				Type: "function",
				FunctionCall: &llms.FunctionCall{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
		}
	}

	choice.Content = answer.String()
	// A turn that only made tool calls has no final answer; fall back to the
	// commentary so the caller still sees what the model said.
	if choice.Content == "" {
		choice.Content = commentary.String()
	}
	if len(choice.ToolCalls) > 0 {
		choice.FuncCall = choice.ToolCalls[0].FunctionCall
	}
	if s := summary.String(); s != "" {
		choice.ReasoningContent = s
		choice.GenerationInfo[GenInfoReasoningSummary] = s
	}
	if len(reasoning) > 0 {
		choice.GenerationInfo[GenInfoReasoningItems] = reasoning
	}
	choice.StopReason = resp.StopReason()
	choice.GenerationInfo[GenInfoResponseID] = resp.ID
	if u := resp.Usage; u != nil {
		choice.GenerationInfo["PromptTokens"] = u.InputTokens
		choice.GenerationInfo["CompletionTokens"] = u.OutputTokens
		choice.GenerationInfo["TotalTokens"] = u.TotalTokens
		choice.GenerationInfo["ReasoningTokens"] = u.OutputTokensDetails.ReasoningTokens
		choice.GenerationInfo["CachedTokens"] = u.InputTokensDetails.CachedTokens
	}

	return &llms.ContentResponse{Choices: []*llms.ContentChoice{choice}}, nil
}

// reasoningExplicitlyOff reports whether the caller asked for no reasoning at
// all. Anything else - including leaving the effort unset - leaves the model
// reasoning at the server's default.
func reasoningExplicitlyOff(r *openaiclient.ResponsesReasoning) bool {
	return r != nil && r.Effort == openaiclient.ReasoningEffortNone
}

// apiMetadata drops this package's internal metadata keys so they are not sent
// to the API.
func apiMetadata(metadata map[string]any) map[string]any {
	if len(metadata) == 0 {
		return nil
	}
	out := make(map[string]any, len(metadata))
	for k, v := range metadata {
		if k == "thinking_config" || strings.HasPrefix(k, "openai:") {
			continue
		}
		out[k] = v
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
