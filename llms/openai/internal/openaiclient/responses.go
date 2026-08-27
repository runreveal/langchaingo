package openaiclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmc/langchaingo/llms"
)

// responsesEndpoint is the path of OpenAI's Responses API, the successor to
// /chat/completions. Reasoning models from gpt-5.6 on only support function
// tools together with reasoning here.
const responsesEndpoint = "/responses"

// responsesStreamBufferBytes bounds a single SSE line. The terminal
// response.completed event carries the whole response object, including every
// encrypted reasoning item, so it is far larger than a chat completion chunk and
// easily exceeds bufio.Scanner's 64KiB default.
const responsesStreamBufferBytes = 8 << 20

// ResponsesRequest is a request to POST /v1/responses.
type ResponsesRequest struct {
	Model string `json:"model"`
	// Instructions carries system-level guidance. The Responses API takes it as
	// a top-level field rather than a message with role "system".
	Instructions string `json:"instructions,omitempty"`
	// Input is the conversation so far, as Items. Items returned by a previous
	// response must be replayed verbatim; see ResponsesOutputItem.Raw.
	Input []json.RawMessage `json:"input"`

	Tools      []ResponsesTool `json:"tools,omitempty"`
	ToolChoice any             `json:"tool_choice,omitempty"`

	Reasoning       *ResponsesReasoning `json:"reasoning,omitempty"`
	MaxOutputTokens int                 `json:"max_output_tokens,omitempty"`
	Temperature     *float64            `json:"temperature,omitempty"`
	TopP            float64             `json:"top_p,omitempty"`
	Text            *ResponsesText      `json:"text,omitempty"`
	Metadata        map[string]any      `json:"metadata,omitempty"`

	// Include asks for extra fields on output items, e.g.
	// "reasoning.encrypted_content", which is required to replay reasoning when
	// Store is false.
	Include []string `json:"include,omitempty"`
	// Store controls server-side retention of the response. It is serialised
	// unconditionally because the API defaults it to true and we want the
	// stateless default to be explicit.
	Store  bool `json:"store"`
	Stream bool `json:"stream,omitempty"`

	StreamingFunc          func(ctx context.Context, chunk []byte) error                 `json:"-"`
	StreamingReasoningFunc func(ctx context.Context, reasoningChunk, chunk []byte) error `json:"-"`
}

// ResponsesReasoning configures reasoning for a Responses request.
type ResponsesReasoning struct {
	// Effort is one of "none", "low", "medium", "high", "xhigh", "max".
	// Availability varies by model.
	Effort string `json:"effort,omitempty"`
	// Summary requests a natural-language summary of the reasoning, e.g. "auto".
	Summary string `json:"summary,omitempty"`
}

// ResponsesText configures the output text format. It is where the Responses
// API keeps what /chat/completions calls response_format.
type ResponsesText struct {
	Format *ResponseFormat `json:"format,omitempty"`
}

// ResponsesTool is a tool definition. Unlike /chat/completions, which nests the
// definition under a "function" key, Responses tags it inline.
type ResponsesTool struct {
	Type        string `json:"type"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
	// Strict is always serialised: Responses defaults it to true while
	// /chat/completions defaults it to false, and we want one behaviour.
	Strict bool `json:"strict"`
}

// ResponsesResponse is the response body of POST /v1/responses, and also the
// payload of the terminal response.completed stream event.
type ResponsesResponse struct {
	ID                string                `json:"id"`
	Object            string                `json:"object"`
	Status            string                `json:"status"`
	Model             string                `json:"model"`
	Output            []ResponsesOutputItem `json:"output"`
	Usage             *ResponsesUsage       `json:"usage"`
	Error             *ResponsesError       `json:"error"`
	IncompleteDetails *ResponsesIncomplete  `json:"incomplete_details"`
}

// ResponsesError is a terminal error reported inside a response body.
type ResponsesError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ResponsesIncomplete explains a status of "incomplete", e.g. hitting
// max_output_tokens.
type ResponsesIncomplete struct {
	Reason string `json:"reason"`
}

// ResponsesUsage is the token accounting for a response.
type ResponsesUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	TotalTokens        int `json:"total_tokens"`
	InputTokensDetails struct {
		CachedTokens     int `json:"cached_tokens"`
		CacheWriteTokens int `json:"cache_write_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

// Output item types we care about.
const (
	ResponsesItemMessage            = "message"
	ResponsesItemReasoning          = "reasoning"
	ResponsesItemFunctionCall       = "function_call"
	ResponsesItemFunctionCallOutput = "function_call_output"
)

// ResponsesPhaseFinalAnswer marks the message item holding the model's answer.
// gpt-5.6 also emits "commentary" messages that narrate tool use; they are not
// part of the answer.
const ResponsesPhaseFinalAnswer = "final_answer"

// ResponsesOutputItem is one item of a response's output array.
//
// Raw holds the item exactly as the API sent it. Replaying reasoning items on a
// later turn requires byte-identical content (the API validates
// encrypted_content), and preserving the raw form also means item types this
// client does not model survive a round trip.
type ResponsesOutputItem struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	Role   string `json:"role"`
	Status string `json:"status"`
	Phase  string `json:"phase"`

	Content []ResponsesContentPart `json:"content"`

	// function_call
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`

	// reasoning
	Summary          []ResponsesContentPart `json:"summary"`
	EncryptedContent string                 `json:"encrypted_content"`

	Raw json.RawMessage `json:"-"`
}

// UnmarshalJSON decodes the item and retains the original bytes in Raw.
func (i *ResponsesOutputItem) UnmarshalJSON(data []byte) error {
	type alias ResponsesOutputItem
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*i = ResponsesOutputItem(a)
	i.Raw = append(json.RawMessage(nil), data...)
	return nil
}

// ResponsesContentPart is a part of a message's content or a reasoning summary.
type ResponsesContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Text concatenates the output_text parts of a message item.
func (i ResponsesOutputItem) Text() string {
	var b strings.Builder
	for _, p := range i.Content {
		if p.Type == "output_text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// SummaryText concatenates the parts of a reasoning item's summary.
func (i ResponsesOutputItem) SummaryText() string {
	var b strings.Builder
	for _, p := range i.Summary {
		b.WriteString(p.Text)
	}
	return b.String()
}

// NewResponsesMessageItem builds an input message Item.
func NewResponsesMessageItem(role, content string) (json.RawMessage, error) {
	return json.Marshal(map[string]string{"role": role, "content": content})
}

// NewResponsesFunctionCallOutputItem builds the Item that answers a
// function_call. callID must match the call_id of that call.
func NewResponsesFunctionCallOutputItem(callID, output string) (json.RawMessage, error) {
	return json.Marshal(map[string]string{
		"type":    ResponsesItemFunctionCallOutput,
		"call_id": callID,
		"output":  output,
	})
}

// NewResponsesFunctionCallItem builds a function_call Item for history that was
// not produced by a live response (a reconstructed transcript, say), where the
// original item bytes are no longer available.
func NewResponsesFunctionCallItem(callID, name, arguments string) (json.RawMessage, error) {
	return json.Marshal(map[string]string{
		"type":      ResponsesItemFunctionCall,
		"call_id":   callID,
		"name":      name,
		"arguments": arguments,
	})
}

// CreateResponse creates a response. It fills in the client's model when the
// request does not name one.
func (c *Client) CreateResponse(ctx context.Context, r *ResponsesRequest) (*ResponsesResponse, error) {
	if r.Model == "" {
		if c.Model == "" {
			r.Model = defaultChatModel
		} else {
			r.Model = c.Model
		}
	}
	return c.createResponse(ctx, r)
}

func (c *Client) createResponse(ctx context.Context, payload *ResponsesRequest) (*ResponsesResponse, error) {
	if payload.StreamingFunc != nil || payload.StreamingReasoningFunc != nil {
		payload.Stream = true
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.buildURL(responsesEndpoint, payload.Model), bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, err
	}
	c.setHeaders(req)

	r, err := c.httpClient.Do(req)
	if err != nil {
		return nil, sanitizeHTTPError(err)
	}
	defer r.Body.Close()

	if r.StatusCode != http.StatusOK {
		apiErr := &llms.APIError{Provider: "openai", StatusCode: r.StatusCode}
		var errResp errorMessage
		if err := json.NewDecoder(r.Body).Decode(&errResp); err != nil {
			return nil, apiErr
		}
		apiErr.Type = errResp.Error.Type
		apiErr.Message = errResp.Error.Message
		return nil, apiErr
	}

	if payload.Stream {
		return parseStreamingResponse(ctx, r, payload)
	}

	var response ResponsesResponse
	if err := json.NewDecoder(r.Body).Decode(&response); err != nil {
		return nil, err
	}
	return &response, checkResponseStatus(&response)
}

// responsesStreamEvent is one server-sent event from a streaming response.
type responsesStreamEvent struct {
	Type     string             `json:"type"`
	Delta    string             `json:"delta"`
	Response *ResponsesResponse `json:"response"`
	// Populated by top-level "error" events.
	Code    string `json:"code"`
	Message string `json:"message"`
}

// parseStreamingResponse consumes the SSE stream, forwarding text and reasoning
// deltas to the callbacks, and returns the response carried by the terminal
// event. Unlike /chat/completions there is no need to reassemble the result from
// deltas: response.completed repeats the whole response object.
func parseStreamingResponse(ctx context.Context, r *http.Response, payload *ResponsesRequest) (
	*ResponsesResponse, error,
) {
	scanner := bufio.NewScanner(r.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), responsesStreamBufferBytes)

	var final *ResponsesResponse
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		line := scanner.Text()
		// Only data lines carry payloads; the event: line repeats the type that
		// is already inside the JSON, and comments start with ':'.
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		var ev responsesStreamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			// Tolerate payloads this client does not model, matching the
			// chat-completions reader.
			continue
		}

		switch ev.Type {
		case "response.output_text.delta":
			if payload.StreamingFunc != nil && ev.Delta != "" {
				if err := payload.StreamingFunc(ctx, []byte(ev.Delta)); err != nil {
					return nil, fmt.Errorf("streaming func returned an error: %w", err)
				}
			}
		case "response.reasoning_summary_text.delta":
			if payload.StreamingReasoningFunc != nil && ev.Delta != "" {
				if err := payload.StreamingReasoningFunc(ctx, []byte(ev.Delta), nil); err != nil {
					return nil, fmt.Errorf("streaming reasoning func returned an error: %w", err)
				}
			}
		case "response.completed", "response.failed", "response.incomplete":
			final = ev.Response
		case "error":
			return nil, &llms.APIError{
				Provider: "openai",
				Type:     ev.Code,
				Message:  ev.Message,
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading streaming response: %w", err)
	}
	if final == nil {
		return nil, ErrEmptyResponse
	}
	return final, checkResponseStatus(final)
}

// checkResponseStatus turns a failed response into an error.
//
// A status of "incomplete" is deliberately not an error: it is how the Responses
// API reports what /chat/completions reports as finish_reason "length", and
// callers auto-continue from it. StopReason carries it instead; see
// ResponsesResponse.StopReason.
func checkResponseStatus(r *ResponsesResponse) error {
	if r.Error != nil && r.Error.Message != "" {
		return &llms.APIError{Provider: "openai", Type: r.Error.Code, Message: r.Error.Message}
	}
	switch r.Status {
	case "", "completed", "incomplete":
		return nil
	default:
		return fmt.Errorf("openai: response status %q", r.Status)
	}
}

// StopReason reports why generation ended, using the same vocabulary as
// /chat/completions finish_reason so callers do not need to branch on which
// endpoint served the request.
func (r *ResponsesResponse) StopReason() string {
	if r.Status == "incomplete" && r.IncompleteDetails != nil {
		switch r.IncompleteDetails.Reason {
		case "max_output_tokens":
			return "length"
		case "":
			return "incomplete"
		default:
			return r.IncompleteDetails.Reason
		}
	}
	for _, item := range r.Output {
		if item.Type == ResponsesItemFunctionCall {
			return "tool_calls"
		}
	}
	return "stop"
}

// ErrNoResponsesOutput is returned when a response carries no output items.
var ErrNoResponsesOutput = errors.New("openai: response contained no output items")
