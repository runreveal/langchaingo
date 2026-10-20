package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

const chatReply = `{"id":"chatcmpl_1","object":"chat.completion","model":"m",
  "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
  "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

// newChatServer starts a stub /chat/completions endpoint that records the
// request body.
func newChatServer(t *testing.T) (*httptest.Server, *map[string]any) {
	t.Helper()
	captured := map[string]any{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/chat/completions"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &captured); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, chatReply)
	}))
	t.Cleanup(srv.Close)
	return srv, &captured
}

var searchTool = llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
	Name: "search", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
}}

func generate(t *testing.T, baseURL, model string, opts ...llms.CallOption) {
	t.Helper()
	llm, err := New(WithToken("sk-test"), WithModel(model), WithBaseURL(baseURL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...); err != nil {
		t.Fatalf("GenerateContent(%s): %v", model, err)
	}
}

// TestResponsesReasoningEffort checks the reasoning.effort sent on
// /v1/responses for each model and option.
func TestResponsesReasoningEffort(t *testing.T) {
	tests := []struct {
		name  string
		model string
		opt   llms.CallOption
		want  string // "" means no reasoning.effort
	}{
		{"thinking mode high", "gpt-6-sol", llms.WithThinkingMode(llms.ThinkingModeHigh), "high"},
		{"thinking mode low", "gpt-5.6-terra", llms.WithThinkingMode(llms.ThinkingModeLow), "low"},
		{"thinking mode none", "gpt-6-luna", llms.WithThinkingMode(llms.ThinkingModeNone), "none"},
		// gpt-6-astra and gpt-6.1-sol have no "none", so the default is kept.
		{"none on astra", "gpt-6-astra", llms.WithThinkingMode(llms.ThinkingModeNone), ""},
		{"none on 6.1 sol", "gpt-6.1-sol", llms.WithThinkingMode(llms.ThinkingModeNone), ""},
		{"auto keeps the default", "gpt-6-sol", llms.WithThinkingMode(llms.ThinkingModeAuto), ""},
		{"explicit effort", "gpt-6-astra", WithReasoningEffort("max"), "max"},
		// An explicit effort is sent as given, so the API reports a bad level.
		{"explicit unsupported effort", "gpt-6-astra", WithReasoningEffort("none"), "none"},
		{"no option", "gpt-6-sol", nil, ""},
		// gpt-5-pro accepts only "high"; any other mode keeps the default.
		{"high on gpt-5-pro", "gpt-5-pro", llms.WithThinkingMode(llms.ThinkingModeHigh), "high"},
		{"low on gpt-5-pro", "gpt-5-pro", llms.WithThinkingMode(llms.ThinkingModeLow), ""},
		{"none on gpt-5-pro", "gpt-5-pro", llms.WithThinkingMode(llms.ThinkingModeNone), ""},
		// gpt-5.x-pro starts at medium.
		{"medium on gpt-5.5-pro", "gpt-5.5-pro", llms.WithThinkingMode(llms.ThinkingModeMedium), "medium"},
		{"low on gpt-5.5-pro", "gpt-5.5-pro", llms.WithThinkingMode(llms.ThinkingModeLow), ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, captured := newResponsesServer(t, simpleReply)
			opts := []llms.CallOption{WithResponsesAPI()}
			if tt.opt != nil {
				opts = append(opts, tt.opt)
			}
			generate(t, srv.URL, tt.model, opts...)
			reasoning, _ := (*captured)["reasoning"].(map[string]any)
			if got, _ := reasoning["effort"].(string); got != tt.want {
				t.Errorf("reasoning.effort = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestResponsesExplicitEffortBeatsThinkingMode checks WithReasoningEffort takes
// precedence over the provider-neutral thinking mode.
func TestResponsesExplicitEffortBeatsThinkingMode(t *testing.T) {
	srv, captured := newResponsesServer(t, simpleReply)
	generate(t, srv.URL, "gpt-6-sol", WithResponsesAPI(),
		llms.WithThinkingMode(llms.ThinkingModeLow), WithReasoningEffort("xhigh"))
	reasoning, _ := (*captured)["reasoning"].(map[string]any)
	if got := reasoning["effort"]; got != "xhigh" {
		t.Errorf("reasoning.effort = %v, want xhigh", got)
	}
}

// TestChatCompletionsReasoningEffort checks the reasoning_effort sent on
// /chat/completions, which used to drop both the thinking mode and
// WithReasoningEffort.
func TestChatCompletionsReasoningEffort(t *testing.T) {
	tests := []struct {
		name  string
		model string
		opts  []llms.CallOption
		want  string // "" means no reasoning_effort
	}{
		{"thinking mode medium", "gpt-5.5", []llms.CallOption{llms.WithThinkingMode(llms.ThinkingModeMedium)}, "medium"},
		{"explicit effort", "gpt-5.2", []llms.CallOption{WithReasoningEffort("xhigh")}, "xhigh"},
		{"none on gpt-5.1", "gpt-5.1", []llms.CallOption{llms.WithThinkingMode(llms.ThinkingModeNone)}, "none"},
		// gpt-5 has no "none"; its lowest level is "minimal".
		{"none on gpt-5", "gpt-5-mini", []llms.CallOption{llms.WithThinkingMode(llms.ThinkingModeNone)}, "minimal"},
		{"o-series", "o3", []llms.CallOption{llms.WithThinkingMode(llms.ThinkingModeHigh)}, "high"},
		// Unknown reasoning models get low/medium/high, never a guessed "none".
		{"unknown reasoning model", "gpt-5.3", []llms.CallOption{llms.WithThinkingMode(llms.ThinkingModeHigh)}, "high"},
		{"unknown model, none", "gpt-5.3", []llms.CallOption{llms.WithThinkingMode(llms.ThinkingModeNone)}, ""},
		{"non-reasoning model", "gpt-4o", []llms.CallOption{llms.WithThinkingMode(llms.ThinkingModeHigh)}, ""},
		{"azure deployment name", "my-deployment", []llms.CallOption{llms.WithThinkingMode(llms.ThinkingModeHigh)}, ""},

		// With tools, gpt-5.6+ takes reasoning off on this endpoint: a
		// mode-derived effort gives way to the client's "none".
		{"tools, thinking mode", "gpt-5.6-sol",
			[]llms.CallOption{llms.WithTools([]llms.Tool{searchTool}), llms.WithThinkingMode(llms.ThinkingModeHigh)}, "none"},
		{"tools, explicit effort", "gpt-5.6-sol",
			[]llms.CallOption{llms.WithTools([]llms.Tool{searchTool}), WithReasoningEffort("low")}, "low"},
		// gpt-6-astra rejects "none", so nothing is injected.
		{"tools on astra", "gpt-6-astra", []llms.CallOption{llms.WithTools([]llms.Tool{searchTool})}, ""},
		// Before gpt-5.6, tools and reasoning combine fine.
		{"tools on gpt-5.5", "gpt-5.5",
			[]llms.CallOption{llms.WithTools([]llms.Tool{searchTool}), llms.WithThinkingMode(llms.ThinkingModeHigh)}, "high"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, captured := newChatServer(t)
			generate(t, srv.URL, tt.model, append([]llms.CallOption{WithChatCompletionsAPI()}, tt.opts...)...)
			got, _ := (*captured)["reasoning_effort"].(string)
			if got != tt.want {
				t.Errorf("reasoning_effort = %q, want %q", got, tt.want)
			}
		})
	}
}
