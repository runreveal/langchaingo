package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// newResponsesServer starts a stub /v1/responses endpoint that records the
// request body and replies with the given body.
func newResponsesServer(t *testing.T, reply string) (*httptest.Server, *map[string]any) {
	t.Helper()
	captured := map[string]any{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/responses"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &captured); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, reply)
	}))
	t.Cleanup(srv.Close)
	return srv, &captured
}

func newResponsesLLM(t *testing.T, baseURL string) *LLM {
	t.Helper()
	llm, err := New(WithToken("sk-test"), WithModel("gpt-5.6-terra"), WithBaseURL(baseURL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return llm
}

const simpleReply = `{
  "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5.6-terra",
  "output": [{"id":"msg_1","type":"message","role":"assistant","status":"completed","phase":"final_answer",
              "content":[{"type":"output_text","text":"hi there"}]}],
  "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens_details": {"reasoning_tokens": 2}}
}`

func TestResponsesRequestMapping(t *testing.T) {
	srv, captured := newResponsesServer(t, simpleReply)
	llm := newResponsesLLM(t, srv.URL)

	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "be terse"),
		llms.TextParts(llms.ChatMessageTypeSystem, "and correct"),
		llms.TextParts(llms.ChatMessageTypeHuman, "hello"),
	},
		WithResponsesAPI(),
		WithReasoningEffort("high"),
		WithReasoningSummary("auto"),
		llms.WithMaxTokens(512),
		llms.WithTools([]llms.Tool{{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "SourceList",
				Description: "List sources",
				Parameters:  map[string]any{"type": "object"},
			},
		}}),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	req := *captured

	// System messages become top-level instructions, joined in order.
	if got, want := req["instructions"], "be terse\n\nand correct"; got != want {
		t.Errorf("instructions = %q, want %q", got, want)
	}
	if _, ok := req["messages"]; ok {
		t.Error("request must not carry a chat-completions messages field")
	}
	input, _ := req["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input has %d items, want 1: %v", len(input), input)
	}
	first, _ := input[0].(map[string]any)
	if first["role"] != "user" || first["content"] != "hello" {
		t.Errorf("input[0] = %v, want the user message", first)
	}

	// Reasoning, and no temperature alongside it.
	reasoning, _ := req["reasoning"].(map[string]any)
	if reasoning["effort"] != "high" || reasoning["summary"] != "auto" {
		t.Errorf("reasoning = %v", reasoning)
	}
	if _, ok := req["temperature"]; ok {
		t.Error("temperature must be omitted while reasoning is active")
	}

	// Stateless by default, with encrypted reasoning so it can be replayed.
	if req["store"] != false {
		t.Errorf("store = %v, want false", req["store"])
	}
	include, _ := req["include"].([]any)
	if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Errorf("include = %v", include)
	}
	if req["max_output_tokens"] != float64(512) {
		t.Errorf("max_output_tokens = %v, want 512", req["max_output_tokens"])
	}

	// Tools are tagged inline rather than nested under "function".
	toolsField, _ := req["tools"].([]any)
	if len(toolsField) != 1 {
		t.Fatalf("tools = %v", toolsField)
	}
	tool, _ := toolsField[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "SourceList" {
		t.Errorf("tool = %v", tool)
	}
	if _, nested := tool["function"]; nested {
		t.Error("tool must not nest the definition under a function key")
	}
	if tool["strict"] != false {
		t.Errorf("strict = %v, want an explicit false to match chat completions", tool["strict"])
	}
}

// A reasoning model rejects a non-default temperature. Leaving the effort unset
// does not disable reasoning - the server applies its own default - so
// temperature must stay off until reasoning is explicitly turned off.
func TestResponsesTemperatureRules(t *testing.T) {
	tests := []struct {
		name  string
		model string
		opts  []llms.CallOption
		want  any
	}{
		{
			name: "reasoning model, effort unset", model: "gpt-5.6-terra",
			want: nil,
		},
		{
			name: "reasoning model, effort high", model: "gpt-5.6-terra",
			opts: []llms.CallOption{WithReasoningEffort("high")}, want: nil,
		},
		{
			name: "reasoning model, effort none", model: "gpt-5.6-terra",
			opts: []llms.CallOption{WithReasoningEffort("none")}, want: 0.25,
		},
		{
			name: "non-reasoning model keeps its temperature", model: "gpt-4o",
			want: 0.25,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, captured := newResponsesServer(t, simpleReply)
			llm, err := New(WithToken("sk-test"), WithModel(tt.model), WithBaseURL(srv.URL))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			opts := append([]llms.CallOption{WithResponsesAPI(), llms.WithTemperature(0.25)}, tt.opts...)
			if _, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...); err != nil {
				t.Fatalf("GenerateContent: %v", err)
			}
			got, present := (*captured)["temperature"]
			if tt.want == nil {
				if present {
					t.Errorf("temperature = %v, want it omitted", got)
				}
				return
			}
			if got != tt.want {
				t.Errorf("temperature = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestResponsesToolRoundTripItems(t *testing.T) {
	srv, captured := newResponsesServer(t, simpleReply)
	llm := newResponsesLLM(t, srv.URL)

	history := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "how many sources?"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextContent{Text: "checking"},
			llms.ToolCall{ID: "call_1", Type: "function", FunctionCall: &llms.FunctionCall{
				Name: "SourceList", Arguments: `{"limit":10}`,
			}},
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "call_1", Name: "SourceList", Content: `{"n":2}`},
		}},
	}

	priorReasoning := []json.RawMessage{
		json.RawMessage(`{"type":"reasoning","id":"rs_1","encrypted_content":"OPAQUE"}`),
	}

	_, err := llm.GenerateContent(context.Background(), history,
		WithResponsesAPI(), WithReasoningEffort("high"), WithPriorReasoningItems(priorReasoning))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}

	input, _ := (*captured)["input"].([]any)
	types := make([]string, 0, len(input))
	for _, raw := range input {
		item, _ := raw.(map[string]any)
		if typ, ok := item["type"].(string); ok {
			types = append(types, typ)
			continue
		}
		types = append(types, "message:"+fmt.Sprint(item["role"]))
	}

	// The reasoning item must land immediately before the assistant turn it
	// belongs to, and the tool call and its output must be separate items.
	want := []string{"message:user", "reasoning", "message:assistant", "function_call", "function_call_output"}
	if strings.Join(types, ",") != strings.Join(want, ",") {
		t.Fatalf("input items = %v, want %v", types, want)
	}

	reasoningItem, _ := input[1].(map[string]any)
	if reasoningItem["encrypted_content"] != "OPAQUE" {
		t.Errorf("reasoning item was not replayed verbatim: %v", reasoningItem)
	}
	call, _ := input[3].(map[string]any)
	if call["call_id"] != "call_1" || call["name"] != "SourceList" {
		t.Errorf("function_call = %v", call)
	}
	out, _ := input[4].(map[string]any)
	if out["call_id"] != "call_1" || out["output"] != `{"n":2}` {
		t.Errorf("function_call_output = %v", out)
	}
}

const toolCallReply = `{
  "id": "resp_2", "status": "completed", "model": "gpt-5.6-terra",
  "output": [
    {"id":"rs_1","type":"reasoning","encrypted_content":"OPAQUE",
     "summary":[{"type":"summary_text","text":"weighing options"}]},
    {"id":"msg_1","type":"message","role":"assistant","phase":"commentary",
     "content":[{"type":"output_text","text":"Let me look that up."}]},
    {"id":"fc_1","type":"function_call","call_id":"call_9","name":"SourceList","arguments":"{\"limit\":100}"}
  ],
  "usage": {"input_tokens": 20, "output_tokens": 30, "total_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 12}}
}`

func TestResponsesParsesToolCallsAndReasoning(t *testing.T) {
	srv, _ := newResponsesServer(t, toolCallReply)
	llm := newResponsesLLM(t, srv.URL)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many?")},
		WithResponsesAPI())
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	c := resp.Choices[0]

	if len(c.ToolCalls) != 1 || c.ToolCalls[0].ID != "call_9" {
		t.Fatalf("tool calls = %v", c.ToolCalls)
	}
	if c.FuncCall == nil || c.FuncCall.Name != "SourceList" {
		t.Errorf("FuncCall = %v", c.FuncCall)
	}
	// No final_answer message, so the commentary stands in as the content.
	if c.Content != "Let me look that up." {
		t.Errorf("Content = %q", c.Content)
	}
	if c.ReasoningContent != "weighing options" {
		t.Errorf("ReasoningContent = %q", c.ReasoningContent)
	}
	items, ok := c.GenerationInfo[GenInfoReasoningItems].([]json.RawMessage)
	if !ok || len(items) != 1 {
		t.Fatalf("reasoning items = %v", c.GenerationInfo[GenInfoReasoningItems])
	}
	if !strings.Contains(string(items[0]), "OPAQUE") {
		t.Errorf("reasoning item lost its encrypted content: %s", items[0])
	}
	if c.GenerationInfo["ReasoningTokens"] != 12 || c.GenerationInfo["PromptTokens"] != 20 {
		t.Errorf("token usage = %v", c.GenerationInfo)
	}
	if c.GenerationInfo[GenInfoResponseID] != "resp_2" {
		t.Errorf("response id = %v", c.GenerationInfo[GenInfoResponseID])
	}
}

const bothPhasesReply = `{
  "id": "resp_3", "status": "completed",
  "output": [
    {"type":"message","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"thinking out loud"}]},
    {"type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"the answer"}]}
  ]
}`

// Commentary messages narrate tool use; folding them into the content would make
// the model appear to answer twice.
func TestResponsesPrefersFinalAnswerOverCommentary(t *testing.T) {
	srv, _ := newResponsesServer(t, bothPhasesReply)
	llm := newResponsesLLM(t, srv.URL)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "q")}, WithResponsesAPI())
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if got := resp.Choices[0].Content; got != "the answer" {
		t.Errorf("Content = %q, want only the final answer", got)
	}
}

func TestResponsesStreaming(t *testing.T) {
	events := []string{
		`{"type":"response.created","response":{"id":"resp_4","status":"in_progress"}}`,
		`{"type":"response.reasoning_summary_text.delta","delta":"weigh"}`,
		`{"type":"response.reasoning_summary_text.delta","delta":"ing"}`,
		`{"type":"response.output_text.delta","delta":"Hello"}`,
		`{"type":"response.output_text.delta","delta":", world"}`,
		`{"type":"response.completed","response":{"id":"resp_4","status":"completed",` +
			`"output":[{"type":"message","role":"assistant","phase":"final_answer",` +
			`"content":[{"type":"output_text","text":"Hello, world"}]}],` +
			`"usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12}}}`,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["stream"] != true {
			t.Errorf("stream = %v, want true", req["stream"])
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			fmt.Fprintf(w, "event: x\ndata: %s\n\n", e)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	llm := newResponsesLLM(t, srv.URL)
	var text, reasoning strings.Builder
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		WithResponsesAPI(),
		llms.WithStreamingFunc(func(_ context.Context, c []byte) error { text.Write(c); return nil }),
		llms.WithStreamingReasoningFunc(func(_ context.Context, r, _ []byte) error { reasoning.Write(r); return nil }),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if text.String() != "Hello, world" {
		t.Errorf("streamed text = %q", text.String())
	}
	if reasoning.String() != "weighing" {
		t.Errorf("streamed reasoning = %q", reasoning.String())
	}
	// The terminal event repeats the whole response, so the final content comes
	// from there rather than from reassembled deltas.
	if resp.Choices[0].Content != "Hello, world" {
		t.Errorf("final content = %q", resp.Choices[0].Content)
	}
	if resp.Choices[0].GenerationInfo["TotalTokens"] != 12 {
		t.Errorf("usage = %v", resp.Choices[0].GenerationInfo)
	}
}

// The terminal event carries every encrypted reasoning item, so it can be far
// larger than bufio.Scanner's default 64KiB line limit.
func TestResponsesStreamingHandlesLargeTerminalEvent(t *testing.T) {
	blob := strings.Repeat("A", 200_000)
	final := fmt.Sprintf(`{"type":"response.completed","response":{"id":"resp_5","status":"completed",`+
		`"output":[{"type":"reasoning","encrypted_content":%q},`+
		`{"type":"message","role":"assistant","phase":"final_answer",`+
		`"content":[{"type":"output_text","text":"ok"}]}]}}`, blob)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", `{"type":"response.output_text.delta","delta":"ok"}`)
		fmt.Fprintf(w, "data: %s\n\n", final)
	}))
	defer srv.Close()

	llm := newResponsesLLM(t, srv.URL)
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		WithResponsesAPI(),
		llms.WithStreamingFunc(func(context.Context, []byte) error { return nil }))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if resp.Choices[0].Content != "ok" {
		t.Errorf("content = %q", resp.Choices[0].Content)
	}
	items, _ := resp.Choices[0].GenerationInfo[GenInfoReasoningItems].([]json.RawMessage)
	if len(items) != 1 || !strings.Contains(string(items[0]), blob) {
		t.Error("large reasoning item did not survive the stream")
	}
}

func TestResponsesErrors(t *testing.T) {
	tests := []struct {
		name    string
		status  int
		body    string
		wantErr string
	}{
		{
			name:    "http error",
			status:  http.StatusBadRequest,
			body:    `{"error":{"type":"invalid_request_error","message":"bad model"}}`,
			wantErr: "bad model",
		},
		{
			name:    "failed status",
			status:  http.StatusOK,
			body:    `{"id":"r","status":"failed","error":{"code":"server_error","message":"boom"},"output":[]}`,
			wantErr: "boom",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.status)
				fmt.Fprint(w, tt.body)
			}))
			defer srv.Close()

			llm := newResponsesLLM(t, srv.URL)
			_, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, WithResponsesAPI())
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %v, want it to mention %q", err, tt.wantErr)
			}
		})
	}
}

func TestResponsesStreamErrorEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"error\",\"code\":\"rate_limit_exceeded\",\"message\":\"slow down\"}\n\n")
	}))
	defer srv.Close()

	llm := newResponsesLLM(t, srv.URL)
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		WithResponsesAPI(),
		llms.WithStreamingFunc(func(context.Context, []byte) error { return nil }))
	if err == nil || !strings.Contains(err.Error(), "slow down") {
		t.Fatalf("error = %v, want it to mention the stream error", err)
	}
}

func TestUseResponsesAPIRouting(t *testing.T) {
	tests := []struct {
		name    string
		model   string
		baseURL string
		opts    []llms.CallOption
		want    bool
	}{
		{name: "gpt-5.6 on the stock endpoint", model: "gpt-5.6-terra", want: true},
		{name: "gpt-5.6-sol on the stock endpoint", model: "gpt-5.6-sol", want: true},
		{name: "a later family", model: "gpt-6", want: true},
		{name: "gpt-5.5 stays on chat completions", model: "gpt-5.5", want: false},
		{name: "gpt-4o stays on chat completions", model: "gpt-4o", want: false},
		{
			name: "a custom base URL is left alone", model: "gpt-5.6-terra",
			baseURL: "https://inference.do-ai.run/v1", want: false,
		},
		{
			name: "a custom base URL can opt in", model: "gpt-5.6-terra",
			baseURL: "https://my-azure.openai.azure.com", opts: []llms.CallOption{WithResponsesAPI()}, want: true,
		},
		{
			name: "chat completions can be forced", model: "gpt-5.6-terra",
			opts: []llms.CallOption{WithChatCompletionsAPI()}, want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llmOpts := []Option{WithToken("sk-test"), WithModel(tt.model)}
			if tt.baseURL != "" {
				llmOpts = append(llmOpts, WithBaseURL(tt.baseURL))
			}
			llm, err := New(llmOpts...)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			callOpts := llms.CallOptions{}
			for _, o := range tt.opts {
				o(&callOpts)
			}
			if got := llm.useResponsesAPI(&callOpts, tt.model); got != tt.want {
				t.Errorf("useResponsesAPI(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

// A model routed to Responses must not also get the chat-completions
// reasoning-off workaround; that would silently disable reasoning.
func TestResponsesRoutedModelKeepsReasoning(t *testing.T) {
	srv, captured := newResponsesServer(t, simpleReply)
	llm := newResponsesLLM(t, srv.URL)

	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		WithResponsesAPI(), WithReasoningEffort("high"),
		llms.WithTools([]llms.Tool{{
			Type:     "function",
			Function: &llms.FunctionDefinition{Name: "T", Parameters: map[string]any{"type": "object"}},
		}}))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	reasoning, _ := (*captured)["reasoning"].(map[string]any)
	if reasoning["effort"] != "high" {
		t.Errorf("reasoning effort = %v, want it preserved alongside tools", reasoning["effort"])
	}
	if _, ok := (*captured)["reasoning_effort"]; ok {
		t.Error("the chat-completions reasoning_effort field leaked into a responses request")
	}
}

// An "incomplete" response is the Responses equivalent of finish_reason
// "length". It must surface as a stop reason so callers can auto-continue,
// not as an error.
func TestResponsesTruncationIsNotAnError(t *testing.T) {
	body := `{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
	          "output":[{"type":"message","role":"assistant","phase":"final_answer",
	                     "content":[{"type":"output_text","text":"partial"}]}]}`
	srv, _ := newResponsesServer(t, body)
	llm := newResponsesLLM(t, srv.URL)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, WithResponsesAPI())
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if got := resp.Choices[0].StopReason; got != "length" {
		t.Errorf("StopReason = %q, want %q", got, "length")
	}
	if got := resp.Choices[0].Content; got != "partial" {
		t.Errorf("Content = %q, want the partial output to be kept", got)
	}
}

func TestResponsesStopReasons(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "plain completion",
			body: simpleReply,
			want: "stop",
		},
		{
			name: "tool call",
			body: toolCallReply,
			want: "tool_calls",
		},
		{
			name: "other incomplete reason passes through",
			body: `{"id":"r","status":"incomplete","incomplete_details":{"reason":"content_filter"},
			        "output":[{"type":"message","role":"assistant","phase":"final_answer",
			                   "content":[{"type":"output_text","text":"x"}]}]}`,
			want: "content_filter",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := newResponsesServer(t, tt.body)
			llm := newResponsesLLM(t, srv.URL)
			resp, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, WithResponsesAPI())
			if err != nil {
				t.Fatalf("GenerateContent: %v", err)
			}
			if got := resp.Choices[0].StopReason; got != tt.want {
				t.Errorf("StopReason = %q, want %q", got, tt.want)
			}
		})
	}
}
