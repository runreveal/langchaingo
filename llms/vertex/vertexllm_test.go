package vertex

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// newFakeVertex returns an LLM wired to handler via an HTTP client that
// always dials the test server regardless of the hostname in the URL.
func newFakeVertex(t *testing.T, model string, handler http.HandlerFunc) *LLM {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	// Redirect any https://*-aiplatform.googleapis.com request to the test server.
	redirect := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			target, _ := url.Parse(srv.URL)
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			return http.DefaultTransport.RoundTrip(req)
		}),
	}

	llm, err := New(
		WithProject("test-project"),
		WithLocation("us-east5"),
		WithModel(model),
		WithHTTPClient(redirect),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return llm
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestNew_MissingProject(t *testing.T) {
	_, err := New(WithLocation("us-east5"), WithModel("claude-3-7-sonnet@20250219"))
	if err == nil || !strings.Contains(err.Error(), "project") {
		t.Fatalf("expected missing project error, got %v", err)
	}
}

func TestNew_MissingLocation(t *testing.T) {
	_, err := New(WithProject("p"), WithModel("claude-3-7-sonnet@20250219"))
	if err == nil || !strings.Contains(err.Error(), "location") {
		t.Fatalf("expected missing location error, got %v", err)
	}
}

func TestNew_UnknownPublisher(t *testing.T) {
	_, err := New(
		WithProject("p"),
		WithLocation("us-east5"),
		WithModel("mistral-something"),
		WithHTTPClient(http.DefaultClient),
	)
	if err == nil || !strings.Contains(err.Error(), "publisher") {
		t.Fatalf("expected unsupported publisher error, got %v", err)
	}
}

func TestGenerateContent_NonStreaming(t *testing.T) {
	var gotPath, gotBody string
	handler := func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id":"msg_1","type":"message","role":"assistant",
			"content":[{"type":"text","text":"hello world"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":5,"output_tokens":2}
		}`)
	}

	llm := newFakeVertex(t, "claude-3-7-sonnet@20250219", handler)
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("hi")}},
	})
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}

	// URL shape: :rawPredict + project/location/model encoded.
	wantPathFragment := "/v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-3-7-sonnet@20250219:rawPredict"
	if !strings.Contains(gotPath, wantPathFragment) {
		t.Errorf("path missing %q; got %q", wantPathFragment, gotPath)
	}

	// Body: has vertex version, no model field.
	if !strings.Contains(gotBody, `"anthropic_version":"vertex-2023-10-16"`) {
		t.Errorf("body missing anthropic_version vertex-2023-10-16: %s", gotBody)
	}
	if strings.Contains(gotBody, `"model":`) {
		t.Errorf("body should not contain top-level model field: %s", gotBody)
	}

	if len(resp.Choices) != 1 || resp.Choices[0].Content != "hello world" {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if resp.Choices[0].StopReason != "end_turn" {
		t.Errorf("stop reason: got %q want end_turn", resp.Choices[0].StopReason)
	}
	if resp.Choices[0].GenerationInfo["input_tokens"] != 5 {
		t.Errorf("input_tokens: got %v", resp.Choices[0].GenerationInfo["input_tokens"])
	}
}

func TestGenerateContent_GlobalLocation(t *testing.T) {
	var gotHost string
	handler := func(w http.ResponseWriter, r *http.Request) {
		gotHost = r.Host
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{}}`)
	}
	srv := httptest.NewServer(http.HandlerFunc(handler))
	t.Cleanup(srv.Close)

	redirect := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			// Capture the host the library asked for before we rewrite it.
			gotHost = req.URL.Host
			target, _ := url.Parse(srv.URL)
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			return http.DefaultTransport.RoundTrip(req)
		}),
	}

	llm, err := New(
		WithProject("p"),
		WithLocation("global"),
		WithModel("claude-3-5-haiku@20241022"),
		WithHTTPClient(redirect),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("hi")}},
	}); err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if gotHost != "aiplatform.googleapis.com" {
		t.Errorf("global location should target aiplatform.googleapis.com, got %q", gotHost)
	}
}

func TestGenerateContent_Streaming(t *testing.T) {
	var gotPath string
	handler := func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		events := []string{
			`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","usage":{"input_tokens":7,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
			`{"type":"message_stop"}`,
		}
		for _, e := range events {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", "x", e)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}

	llm := newFakeVertex(t, "claude-3-7-sonnet@20250219", handler)

	var streamed []string
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("hi")}},
	}, llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
		streamed = append(streamed, string(chunk))
		return nil
	}))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}

	if !strings.Contains(gotPath, ":streamRawPredict") {
		t.Errorf("streaming should use :streamRawPredict, got %q", gotPath)
	}
	if got := strings.Join(streamed, ""); got != "Hello" {
		t.Errorf("streamed chunks: got %q want %q", got, "Hello")
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Content != "Hello" {
		t.Fatalf("final content: %+v", resp.Choices[0])
	}
	if resp.Choices[0].StopReason != "end_turn" {
		t.Errorf("stop reason: got %q", resp.Choices[0].StopReason)
	}
	if resp.Choices[0].GenerationInfo["input_tokens"] != 7 {
		t.Errorf("input_tokens: got %v", resp.Choices[0].GenerationInfo["input_tokens"])
	}
	if resp.Choices[0].GenerationInfo["output_tokens"] != 2 {
		t.Errorf("output_tokens: got %v", resp.Choices[0].GenerationInfo["output_tokens"])
	}
}

func TestGenerateContent_ToolUse(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("bad request body: %v", err)
		}
		if _, ok := req["tools"]; !ok {
			t.Errorf("expected tools in body, got %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id":"m","type":"message","role":"assistant",
			"content":[
				{"type":"text","text":"Let me look that up."},
				{"type":"tool_use","id":"tu_1","name":"get_weather","input":{"city":"Paris"}}
			],
			"stop_reason":"tool_use",
			"usage":{"input_tokens":10,"output_tokens":4}
		}`)
	}

	llm := newFakeVertex(t, "claude-3-7-sonnet@20250219", handler)
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("weather in Paris?")}},
		},
		llms.WithTools([]llms.Tool{{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get current weather for a city",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string"},
					},
				},
			},
		}}),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}

	choice := resp.Choices[0]
	if choice.Content != "Let me look that up." {
		t.Errorf("text: got %q", choice.Content)
	}
	if len(choice.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(choice.ToolCalls))
	}
	tc := choice.ToolCalls[0]
	if tc.ID != "tu_1" || tc.FunctionCall.Name != "get_weather" {
		t.Errorf("unexpected tool call: %+v", tc)
	}
	if !strings.Contains(tc.FunctionCall.Arguments, `"city":"Paris"`) {
		t.Errorf("tool args should contain city:Paris, got %q", tc.FunctionCall.Arguments)
	}
	if choice.StopReason != "tool_use" {
		t.Errorf("stop reason: got %q", choice.StopReason)
	}
}

func TestGenerateContent_SystemAndUserMessages(t *testing.T) {
	var gotReq map[string]any
	handler := func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{}}`)
	}

	llm := newFakeVertex(t, "claude-3-7-sonnet@20250219", handler)
	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{llms.TextPart("you are terse")}},
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("hi")}},
	})
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}

	if gotReq["system"] != "you are terse" {
		t.Errorf("system prompt not passed through: %v", gotReq["system"])
	}
	messages, ok := gotReq["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("expected 1 non-system message, got %v", gotReq["messages"])
	}
	if m := messages[0].(map[string]any); m["role"] != "user" {
		t.Errorf("first message should be user, got %v", m["role"])
	}
}

func TestGenerateContent_StreamingToolUse(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		events := []string{
			`{"type":"message_start","message":{"id":"m","usage":{"input_tokens":3,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_9","name":"lookup"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"x"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"}"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
			`{"type":"message_stop"}`,
		}
		for _, e := range events {
			fmt.Fprintf(w, "data: %s\n\n", e)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}

	llm := newFakeVertex(t, "claude-3-7-sonnet@20250219", handler)
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("go")}},
		},
		llms.WithStreamingFunc(func(_ context.Context, _ []byte) error { return nil }),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if len(resp.Choices[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Choices[0].ToolCalls))
	}
	tc := resp.Choices[0].ToolCalls[0]
	if tc.ID != "tu_9" || tc.FunctionCall.Name != "lookup" {
		t.Errorf("unexpected tool call: %+v", tc)
	}
	if tc.FunctionCall.Arguments != `{"q":"x"}` {
		t.Errorf("tool args: got %q want %q", tc.FunctionCall.Arguments, `{"q":"x"}`)
	}
}
