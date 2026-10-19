package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// captureRequest runs one GenerateContent call against a local server and
// returns the JSON body and beta header that were sent to /messages.
func captureRequest(t *testing.T, model string, opts ...llms.CallOption) (map[string]any, string) {
	t.Helper()
	var (
		body map[string]any
		beta string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		beta = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_test","type":"message","role":"assistant","model":"`+model+`",
			"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithToken("test-key"), WithBaseURL(srv.URL), WithModel(model))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	opts = append([]llms.CallOption{llms.WithMaxTokens(4096)}, opts...)
	if _, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...); err != nil {
		t.Fatalf("GenerateContent(%s): %v", model, err)
	}
	return body, beta
}

func thinkingField(body map[string]any) map[string]any {
	th, _ := body["thinking"].(map[string]any)
	return th
}

func effortField(body map[string]any) string {
	oc, _ := body["output_config"].(map[string]any)
	e, _ := oc["effort"].(string)
	return e
}

// TestThinkingRequestByModel checks the thinking request each Claude family is
// sent. Adaptive models (4.6 and later) must never get budget_tokens: 4.7 and
// later answer it with a 400.
func TestThinkingRequestByModel(t *testing.T) {
	cases := []struct {
		name       string
		model      string
		opt        llms.CallOption
		wantType   string // "" means no thinking field
		wantBudget float64
		wantEffort string
	}{
		{"opus 5.5 high", "claude-opus-5-5", llms.WithThinkingMode(llms.ThinkingModeHigh), "adaptive", 0, "high"},
		{"sonnet 5.5 low", "claude-sonnet-5-5", llms.WithThinkingMode(llms.ThinkingModeLow), "adaptive", 0, "low"},
		{"fable auto", "claude-fable-5-1", llms.WithThinkingMode(llms.ThinkingModeAuto), "adaptive", 0, ""},
		{"haiku 5.5 medium", "claude-haiku-5-5", llms.WithThinkingMode(llms.ThinkingModeMedium), "adaptive", 0, "medium"},
		{"opus 4.8 explicit budget", "claude-opus-4-8", llms.WithThinkingBudget(8000), "adaptive", 0, ""},
		{
			"bedrock-style id",
			"global.anthropic.claude-opus-5-5",
			llms.WithThinkingMode(llms.ThinkingModeHigh),
			"adaptive",
			0,
			"high",
		},
		// ThinkingModeNone sends nothing: Opus 5.5 rejects thinking.type=disabled.
		{"opus 5.5 none", "claude-opus-5-5", llms.WithThinkingMode(llms.ThinkingModeNone), "", 0, ""},
		// Budget-style models keep budget_tokens (50% of 4096 max tokens).
		{"sonnet 4.5 medium", "claude-sonnet-4-5", llms.WithThinkingMode(llms.ThinkingModeMedium), "enabled", 2048, ""},
		{"haiku 4.5 budget", "claude-haiku-4-5", llms.WithThinkingBudget(3000), "enabled", 3000, ""},
		// Unknown ids keep the name-based guess and the budget request.
		{"unknown claude 4", "claude-sonnet-4-9-preview", llms.WithThinkingBudget(2000), "enabled", 2000, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			body, _ := captureRequest(t, c.model, c.opt)
			th := thinkingField(body)
			if c.wantType == "" {
				if th != nil {
					t.Fatalf("thinking = %v, want none", th)
				}
				return
			}
			if got, _ := th["type"].(string); got != c.wantType {
				t.Errorf("thinking.type = %q, want %q", got, c.wantType)
			}
			if got, _ := th["budget_tokens"].(float64); got != c.wantBudget {
				t.Errorf("thinking.budget_tokens = %v, want %v", got, c.wantBudget)
			}
			if got := effortField(body); got != c.wantEffort {
				t.Errorf("output_config.effort = %q, want %q", got, c.wantEffort)
			}
		})
	}
}

// TestInterleavedThinkingHeader checks the interleaved-thinking beta header is
// only sent to budget-style models; adaptive thinking interleaves on its own.
func TestInterleavedThinkingHeader(t *testing.T) {
	interleave := llms.WithThinking(&llms.ThinkingConfig{
		Mode: llms.ThinkingModeHigh, InterleaveThinking: true,
	})
	if _, beta := captureRequest(t, "claude-sonnet-4-5", interleave); !strings.Contains(beta, "interleaved-thinking") {
		t.Errorf("sonnet 4.5 anthropic-beta = %q, want interleaved-thinking", beta)
	}
	if _, beta := captureRequest(t, "claude-opus-5-5", interleave); strings.Contains(beta, "interleaved-thinking") {
		t.Errorf("opus 5.5 anthropic-beta = %q, want no interleaved-thinking", beta)
	}
}

// TestSamplingParamsByModel checks temperature and top_p are dropped for the
// models that reject them and passed through everywhere else.
func TestSamplingParamsByModel(t *testing.T) {
	cases := []struct {
		model    string
		wantSent bool
	}{
		{"claude-opus-5-5", false},
		{"claude-sonnet-5-5", false},
		{"claude-fable-5-1", false},
		{"claude-opus-4-8", false},
		{"claude-opus-4-6", true},
		{"claude-sonnet-4-6", true},
		{"claude-haiku-4-5", true},
		{"claude-some-future-model", true},
	}
	for _, c := range cases {
		t.Run(c.model, func(t *testing.T) {
			body, _ := captureRequest(t, c.model, llms.WithTemperature(0.3), llms.WithTopP(0.9))
			_, hasTemp := body["temperature"]
			_, hasTopP := body["top_p"]
			if hasTemp != c.wantSent || hasTopP != c.wantSent {
				t.Errorf("temperature sent=%v top_p sent=%v, want %v", hasTemp, hasTopP, c.wantSent)
			}
		})
	}
}
