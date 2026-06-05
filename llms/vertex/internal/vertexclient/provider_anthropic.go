package vertexclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/tmc/langchaingo/llms"
)

// Ref: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude
// Also: https://docs.anthropic.com/claude/reference/messages_post
//
// Vertex wire format is the native Anthropic Messages API with three deltas:
//   - "model" is in the URL path, not the body
//   - body includes {"anthropic_version": "vertex-2023-10-16"}
//   - auth is "Authorization: Bearer <GCP token>" (handled by the HTTP client)

// anthropicBinGenerationInputSource is the source of an image content block.
type anthropicBinGenerationInputSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// anthropicTextGenerationInputContent is a single content block in a message.
type anthropicTextGenerationInputContent struct {
	// "text", "image", "tool_use", or "tool_result".
	Type      string                             `json:"type"`
	Source    *anthropicBinGenerationInputSource `json:"source,omitempty"`
	Text      string                             `json:"text,omitempty"`
	ToolUseID string                             `json:"tool_use_id,omitempty"`
	Content   string                             `json:"content,omitempty"`
	ID        string                             `json:"id,omitempty"`
	Name      string                             `json:"name,omitempty"`
	Input     interface{}                        `json:"input,omitempty"`
}

type anthropicTextGenerationInputMessage struct {
	Role    string                                `json:"role"`
	Content []anthropicTextGenerationInputContent `json:"content"`
}

// anthropicTextGenerationInput is the request body for Vertex Anthropic models.
// Note: no Model field — the model is in the URL.
type anthropicTextGenerationInput struct {
	AnthropicVersion string                                 `json:"anthropic_version"`
	MaxTokens        int                                    `json:"max_tokens"`
	System           string                                 `json:"system,omitempty"`
	Messages         []*anthropicTextGenerationInputMessage `json:"messages"`
	Temperature      float64                                `json:"temperature,omitempty"`
	TopP             float64                                `json:"top_p,omitempty"`
	TopK             int                                    `json:"top_k,omitempty"`
	StopSequences    []string                               `json:"stop_sequences,omitempty"`
	Tools            []anthropicTool                        `json:"tools,omitempty"`
	ToolChoice       *anthropicToolChoice                   `json:"tool_choice,omitempty"`
	Stream           bool                                   `json:"stream,omitempty"`
}

type anthropicTool struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	InputSchema interface{} `json:"input_schema"`
}

type anthropicToolChoice struct {
	Type string `json:"type,omitempty"` // "auto", "any", "tool"
	Name string `json:"name,omitempty"` // required when Type is "tool"
}

// anthropicTextGenerationOutput is the non-streaming response shape.
type anthropicTextGenerationOutput struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []anthropicContentBlock `json:"content"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence string                  `json:"stop_sequence"`
	Usage        struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type anthropicContentBlock struct {
	Type  string      `json:"type"` // "text" or "tool_use"
	Text  string      `json:"text,omitempty"`
	ID    string      `json:"id,omitempty"`
	Name  string      `json:"name,omitempty"`
	Input interface{} `json:"input,omitempty"`
}

const (
	anthropicVersionVertex = "vertex-2023-10-16"

	anthropicStopReasonEndTurn      = "end_turn"
	anthropicStopReasonStopSequence = "stop_sequence"
	anthropicStopReasonToolUse      = "tool_use"

	anthropicRoleSystem    = "system"
	anthropicRoleUser      = "user"
	anthropicRoleAssistant = "assistant"

	anthropicMessageTypeText       = "text"
	anthropicMessageTypeImage      = "image"
	anthropicMessageTypeToolUse    = "tool_use"
	anthropicMessageTypeToolResult = "tool_result"
)

func createAnthropicCompletion(ctx context.Context,
	client *Client,
	modelID string,
	messages []Message,
	options llms.CallOptions,
) (*llms.ContentResponse, error) {
	inputContents, systemPrompt, err := processInputMessagesAnthropic(messages)
	if err != nil {
		return nil, err
	}

	input := anthropicTextGenerationInput{
		AnthropicVersion: anthropicVersionVertex,
		MaxTokens:        getMaxTokens(options.MaxTokens, 2048),
		System:           systemPrompt,
		Messages:         inputContents,
		Temperature:      options.Temperature,
		TopP:             options.TopP,
		TopK:             options.TopK,
		StopSequences:    options.StopWords,
	}

	if len(options.Tools) > 0 {
		tools, err := convertTools(options.Tools)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tools: %w", err)
		}
		input.Tools = tools

		if options.ToolChoice != nil {
			choice, err := convertToolChoice(options.ToolChoice)
			if err != nil {
				return nil, fmt.Errorf("failed to convert tool choice: %w", err)
			}
			input.ToolChoice = choice
		}
	}

	streaming := options.StreamingFunc != nil
	input.Stream = streaming

	body, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}

	endpoint := anthropicEndpoint(client, modelID, streaming)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("vertex anthropic: status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	if streaming {
		return parseAnthropicStream(ctx, resp.Body, options)
	}

	var output anthropicTextGenerationOutput
	if err := json.NewDecoder(resp.Body).Decode(&output); err != nil {
		return nil, err
	}

	if len(output.Content) == 0 {
		return nil, errors.New("no results")
	}
	switch output.StopReason {
	case anthropicStopReasonEndTurn, anthropicStopReasonStopSequence, anthropicStopReasonToolUse:
		// ok
	default:
		return nil, fmt.Errorf("completed due to %s. Maybe try increasing max tokens", output.StopReason)
	}

	choice := &llms.ContentChoice{
		StopReason: output.StopReason,
		GenerationInfo: map[string]interface{}{
			"input_tokens":  output.Usage.InputTokens,
			"output_tokens": output.Usage.OutputTokens,
		},
	}

	var textContent string
	var toolCalls []llms.ToolCall
	for _, block := range output.Content {
		switch block.Type {
		case anthropicMessageTypeText:
			textContent += block.Text
		case anthropicMessageTypeToolUse:
			tc, err := toolCallFromBlock(block)
			if err != nil {
				return nil, fmt.Errorf("failed to convert tool call: %w", err)
			}
			toolCalls = append(toolCalls, tc)
		}
	}
	choice.Content = textContent
	choice.ToolCalls = toolCalls
	if len(toolCalls) > 0 {
		choice.FuncCall = toolCalls[0].FunctionCall
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{choice},
	}, nil
}

// anthropicEndpoint builds the :rawPredict or :streamRawPredict URL.
func anthropicEndpoint(c *Client, modelID string, streaming bool) string {
	verb := "rawPredict"
	if streaming {
		verb = "streamRawPredict"
	}
	// Percent-encode the model so that "@version" suffixes survive.
	escaped := url.PathEscape(modelID)
	return fmt.Sprintf("https://%s/v1/projects/%s/locations/%s/publishers/anthropic/models/%s:%s",
		c.endpointHost(), c.project, c.location, escaped, verb)
}

// processInputMessagesAnthropic collapses consecutive same-role messages,
// extracts the system prompt, and converts neutral Messages to the Anthropic
// input content shape.
func processInputMessagesAnthropic(messages []Message) ([]*anthropicTextGenerationInputMessage, string, error) {
	chunked := make([][]Message, 0, len(messages))
	current := make([]Message, 0, len(messages))
	var lastRole llms.ChatMessageType
	for _, m := range messages {
		if m.Role != lastRole {
			if len(current) > 0 {
				chunked = append(chunked, current)
			}
			current = make([]Message, 0, len(messages))
		}
		current = append(current, m)
		lastRole = m.Role
	}
	if len(current) > 0 {
		chunked = append(chunked, current)
	}

	inputs := make([]*anthropicTextGenerationInputMessage, 0, len(chunked))
	var systemPrompt string
	for _, chunk := range chunked {
		role, err := getAnthropicRole(chunk[0].Role)
		if err != nil {
			return nil, "", err
		}
		if role == anthropicRoleSystem {
			if systemPrompt != "" {
				return nil, "", errors.New("multiple system prompts")
			}
			for _, m := range chunk {
				c := getAnthropicInputContent(m)
				if c.Type != anthropicMessageTypeText {
					return nil, "", errors.New("system prompt must be text")
				}
				systemPrompt += c.Text
			}
			continue
		}
		content := make([]anthropicTextGenerationInputContent, 0, len(chunk))
		for _, m := range chunk {
			content = append(content, getAnthropicInputContent(m))
		}
		inputs = append(inputs, &anthropicTextGenerationInputMessage{
			Role:    role,
			Content: content,
		})
	}
	return inputs, systemPrompt, nil
}

func getAnthropicRole(role llms.ChatMessageType) (string, error) {
	switch role {
	case llms.ChatMessageTypeSystem:
		return anthropicRoleSystem, nil
	case llms.ChatMessageTypeAI:
		return anthropicRoleAssistant, nil
	case llms.ChatMessageTypeGeneric, llms.ChatMessageTypeHuman:
		return anthropicRoleUser, nil
	case llms.ChatMessageTypeFunction, llms.ChatMessageTypeTool:
		// Tool results are sent as user messages in Anthropic's Messages API.
		return anthropicRoleUser, nil
	default:
		return "", fmt.Errorf("role not supported: %s", role)
	}
}

func getAnthropicInputContent(message Message) anthropicTextGenerationInputContent {
	switch message.Type {
	case anthropicMessageTypeText:
		return anthropicTextGenerationInputContent{
			Type: anthropicMessageTypeText,
			Text: message.Content,
		}
	case anthropicMessageTypeImage:
		return anthropicTextGenerationInputContent{
			Type: anthropicMessageTypeImage,
			Source: &anthropicBinGenerationInputSource{
				Type:      "base64",
				MediaType: message.MimeType,
				Data:      base64.StdEncoding.EncodeToString([]byte(message.Content)),
			},
		}
	case "tool_result":
		return anthropicTextGenerationInputContent{
			Type:      anthropicMessageTypeToolResult,
			ToolUseID: message.ToolUseID,
			Content:   message.Content,
		}
	case "tool_call":
		var input interface{}
		if message.ToolArgs != "" {
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(message.ToolArgs), &args); err == nil {
				input = args
			} else {
				input = map[string]interface{}{"arguments": message.ToolArgs}
			}
		} else {
			input = map[string]interface{}{}
		}
		return anthropicTextGenerationInputContent{
			Type:  anthropicMessageTypeToolUse,
			ID:    message.ToolCallID,
			Name:  message.ToolName,
			Input: input,
		}
	}
	return anthropicTextGenerationInputContent{}
}

func convertTools(tools []llms.Tool) ([]anthropicTool, error) {
	out := make([]anthropicTool, len(tools))
	for i, t := range tools {
		if t.Type != "function" {
			return nil, fmt.Errorf("only function tools are supported, got: %s", t.Type)
		}
		out[i] = anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		}
	}
	return out, nil
}

func convertToolChoice(choice interface{}) (*anthropicToolChoice, error) {
	if choice == nil {
		return nil, nil
	}
	switch v := choice.(type) {
	case string:
		switch v {
		case "auto":
			return &anthropicToolChoice{Type: "auto"}, nil
		case "none":
			return nil, nil
		case "required":
			return &anthropicToolChoice{Type: "any"}, nil
		default:
			return nil, fmt.Errorf("unsupported tool choice string: %s", v)
		}
	case map[string]interface{}:
		if typeVal, ok := v["type"].(string); ok && typeVal == "function" {
			if fn, ok := v["function"].(map[string]interface{}); ok {
				if name, ok := fn["name"].(string); ok {
					return &anthropicToolChoice{Type: "tool", Name: name}, nil
				}
			}
		}
		return nil, errors.New("unsupported tool choice structure")
	default:
		return nil, fmt.Errorf("unsupported tool choice type: %T", choice)
	}
}

func toolCallFromBlock(block anthropicContentBlock) (llms.ToolCall, error) {
	args, err := json.Marshal(block.Input)
	if err != nil {
		return llms.ToolCall{}, fmt.Errorf("failed to marshal tool input: %w", err)
	}
	return llms.ToolCall{
		ID:   block.ID,
		Type: "function",
		FunctionCall: &llms.FunctionCall{
			Name:      block.Name,
			Arguments: string(args),
		},
	}, nil
}

// streamEvent is a single SSE event decoded from :streamRawPredict.
type streamEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		Type         string `json:"type"`
		Text         string `json:"text"`
		PartialJSON  string `json:"partial_json"`
		StopReason   string `json:"stop_reason"`
		StopSequence any    `json:"stop_sequence"`
	} `json:"delta"`
	ContentBlock struct {
		Type  string      `json:"type"`
		ID    string      `json:"id"`
		Name  string      `json:"name"`
		Input interface{} `json:"input"`
		Text  string      `json:"text"`
	} `json:"content_block"`
	Message struct {
		ID    string `json:"id"`
		Type  string `json:"type"`
		Role  string `json:"role"`
		Model string `json:"model"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	} `json:"message"`
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// parseAnthropicStream reads SSE events from Vertex's :streamRawPredict and
// aggregates them into a final ContentResponse, invoking options.StreamingFunc
// for each text delta.
func parseAnthropicStream(ctx context.Context, body io.Reader, options llms.CallOptions) (*llms.ContentResponse, error) {
	choice := &llms.ContentChoice{GenerationInfo: map[string]interface{}{}}

	// Index → in-progress tool_use block (id, name, accumulated JSON args).
	type toolBuilder struct {
		id       string
		name     string
		rawInput string
	}
	toolBuilders := map[int]*toolBuilder{}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var dataBuf bytes.Buffer
	flush := func() error {
		if dataBuf.Len() == 0 {
			return nil
		}
		defer dataBuf.Reset()

		var ev streamEvent
		if err := json.Unmarshal(dataBuf.Bytes(), &ev); err != nil {
			// Non-JSON ping or comment — ignore.
			return nil
		}

		switch ev.Type {
		case "message_start":
			choice.GenerationInfo["input_tokens"] = ev.Message.Usage.InputTokens
		case "content_block_start":
			if ev.ContentBlock.Type == anthropicMessageTypeToolUse {
				toolBuilders[ev.Index] = &toolBuilder{
					id:   ev.ContentBlock.ID,
					name: ev.ContentBlock.Name,
				}
			}
		case "content_block_delta":
			switch ev.Delta.Type {
			case "text_delta":
				if ev.Delta.Text != "" {
					if err := options.StreamingFunc(ctx, []byte(ev.Delta.Text)); err != nil {
						return err
					}
					choice.Content += ev.Delta.Text
				}
			case "input_json_delta":
				if b, ok := toolBuilders[ev.Index]; ok {
					b.rawInput += ev.Delta.PartialJSON
				}
			}
		case "content_block_stop":
			if b, ok := toolBuilders[ev.Index]; ok {
				// Normalize empty input as {} for JSON argument consumers.
				args := strings.TrimSpace(b.rawInput)
				if args == "" {
					args = "{}"
				}
				choice.ToolCalls = append(choice.ToolCalls, llms.ToolCall{
					ID:   b.id,
					Type: "function",
					FunctionCall: &llms.FunctionCall{
						Name:      b.name,
						Arguments: args,
					},
				})
				delete(toolBuilders, ev.Index)
			}
		case "message_delta":
			if ev.Delta.StopReason != "" {
				choice.StopReason = ev.Delta.StopReason
			}
			if ev.Usage.OutputTokens > 0 {
				choice.GenerationInfo["output_tokens"] = ev.Usage.OutputTokens
			}
		}
		return nil
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := flush(); err != nil {
				return nil, err
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataBuf.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
		// Ignore "event:", "id:", and comment lines.
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	// Flush any pending event that didn't end with a blank line.
	if err := flush(); err != nil {
		return nil, err
	}

	if len(choice.ToolCalls) > 0 {
		choice.FuncCall = choice.ToolCalls[0].FunctionCall
	}
	return &llms.ContentResponse{Choices: []*llms.ContentChoice{choice}}, nil
}
