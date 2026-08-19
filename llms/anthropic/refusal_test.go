package anthropic

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic/internal/anthropicclient"
)

// guardrailBody is the real shape Opus 5 / Fable 5 return when a request trips a
// safety guardrail: HTTP 200, an empty content array, stop_reason "refusal", and
// the reason in stop_details.
const guardrailBody = `{
  "id": "msg_test",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-5",
  "content": [],
  "stop_reason": "refusal",
  "stop_details": {
    "type": "refusal",
    "category": "cyber",
    "explanation": "This request triggered restrictions on violative cyber content and was blocked under Anthropic's Usage Policy."
  },
  "usage": {"input_tokens": 87, "output_tokens": 0}
}`

type stubDoer struct {
	status int
	body   string
}

func (s stubDoer) Do(*http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: s.status,
		Body:       io.NopCloser(strings.NewReader(s.body)),
		Header:     make(http.Header),
	}, nil
}

// TestGuardrailRefusalBecomesResponse is the regression test for the reported bug:
// a blocked request produced an empty message that surfaced as
// "no valid content in AI message" instead of a usable refusal.
func TestGuardrailRefusalBecomesResponse(t *testing.T) {
	llm, err := New(
		WithToken("test-token"),
		WithModel("claude-opus-5"),
		WithHTTPClient(stubDoer{status: http.StatusOK, body: guardrailBody}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how do I hack a wifi router")})
	if err != nil {
		t.Fatalf("expected a refusal response, got error: %v", err)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("Choices = %d, want 1", len(resp.Choices))
	}
	choice := resp.Choices[0]

	if choice.StopReason != StopReasonRefusal {
		t.Errorf("StopReason = %q, want %q", choice.StopReason, StopReasonRefusal)
	}
	if !strings.Contains(choice.Content, "Usage Policy") {
		t.Errorf("Content = %q, want the guardrail explanation", choice.Content)
	}
	if got := choice.GenerationInfo["RefusalCategory"]; got != "cyber" {
		t.Errorf("RefusalCategory = %v, want cyber", got)
	}
}

// TestGuardrailRefusalWithoutExplanation covers a refusal that carries no
// stop_details: the response must still be non-empty so it does not poison history.
func TestGuardrailRefusalWithoutExplanation(t *testing.T) {
	body := `{"id":"m","type":"message","role":"assistant","content":[],` +
		`"stop_reason":"refusal","usage":{"input_tokens":10,"output_tokens":0}}`
	llm, err := New(WithToken("t"), WithModel("claude-opus-5"),
		WithHTTPClient(stubDoer{status: http.StatusOK, body: body}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "x")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Choices[0].Content == "" {
		t.Error("refusal without explanation produced empty content")
	}
	if resp.Choices[0].StopReason != StopReasonRefusal {
		t.Errorf("StopReason = %q, want refusal", resp.Choices[0].StopReason)
	}
}

// TestEmptyResponseStillErrorsWhenNotRefusal makes sure the new path is scoped:
// an empty response that is not a refusal keeps returning ErrEmptyResponse.
func TestEmptyResponseStillErrorsWhenNotRefusal(t *testing.T) {
	body := `{"id":"m","type":"message","role":"assistant","content":[],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":0}}`
	llm, _ := New(WithToken("t"), WithModel("claude-opus-5"),
		WithHTTPClient(stubDoer{status: http.StatusOK, body: body}))

	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "x")})
	if err == nil {
		t.Fatal("expected ErrEmptyResponse for a non-refusal empty response")
	}
}

// TestHandleAIMessageToleratesEmpty is the second half of the fix: a blocked
// assistant turn already sitting in history must replay without failing the whole
// request. This reproduces the reported "no valid content in AI message".
func TestHandleAIMessageToleratesEmpty(t *testing.T) {
	// An assistant turn whose only text part is empty — what an unfixed refusal
	// persisted to history looks like.
	msg := llms.MessageContent{
		Role:  llms.ChatMessageTypeAI,
		Parts: []llms.ContentPart{llms.TextContent{Text: ""}},
	}

	got, err := handleAIMessage(msg)
	if err != nil {
		t.Fatalf("handleAIMessage on an empty assistant turn should not error, got: %v", err)
	}
	contents, ok := got.Content.([]anthropicclient.Content)
	if !ok || len(contents) == 0 {
		t.Fatalf("expected a placeholder content block, got %#v", got.Content)
	}
}
