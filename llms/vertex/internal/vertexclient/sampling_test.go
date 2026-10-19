package vertexclient

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

type captureTransport struct{ body map[string]any }

func (c *captureTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &c.body); err != nil {
		return nil, err
	}
	resp := `{"id":"msg","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(resp)),
		Request:    r,
	}, nil
}

// TestSamplingParamsByModel checks Claude on Vertex drops temperature, top_p and
// top_k for the models that reject them.
func TestSamplingParamsByModel(t *testing.T) {
	cases := map[string]bool{
		"claude-opus-5-5":          false,
		"claude-sonnet-5-5":        false,
		"claude-opus-4-8":          false,
		"claude-sonnet-4-6":        true,
		"claude-opus-4-5@20251101": true,
		// Unknown to the model catalog: passed through as before.
		"claude-some-future-model": true,
	}
	for model, wantSent := range cases {
		t.Run(model, func(t *testing.T) {
			tr := &captureTransport{}
			c := NewClient(&http.Client{Transport: tr}, "proj", "global")
			_, err := c.CreateCompletion(context.Background(), "", model,
				[]Message{{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"}},
				llms.CallOptions{Temperature: 0.3, TopP: 0.9, TopK: 5, MaxTokens: 100})
			if err != nil {
				t.Fatalf("CreateCompletion: %v", err)
			}
			for _, k := range []string{"temperature", "top_p", "top_k"} {
				if _, sent := tr.body[k]; sent != wantSent {
					t.Errorf("%s sent = %v, want %v", k, sent, wantSent)
				}
			}
		})
	}
}
