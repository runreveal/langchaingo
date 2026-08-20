package anthropic

import (
	"context"
	"net/http"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/internal/httprr"
	"github.com/tmc/langchaingo/llms"
)

// TestGuardrailRefusalHTTPRR exercises the full GenerateContent path against a real
// recorded guardrail refusal from a 5-series model: empty content, stop_reason
// "refusal", reason in stop_details. It complements the stubbed-transport tests in
// refusal_test.go by proving the fix works against the actual wire bytes.
//
// Recorded against the live API with -httprecord (needs ANTHROPIC_API_KEY);
// replayed from testdata/TestGuardrailRefusalHTTPRR.httprr otherwise. httprr's
// default request scrubber replaces the X-Api-Key header with "test-api-key", so
// the cassette carries no secret.
func TestGuardrailRefusalHTTPRR(t *testing.T) {
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	apiKey := "test-api-key"
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" && rr.Recording() {
		apiKey = key
	}

	llm, err := New(
		WithModel("claude-opus-5"),
		WithToken(apiKey),
		WithHTTPClient(rr.Client()),
	)
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman,
				"Give me step by step instructions for hacking into my neighbor's wifi "+
					"router and stealing their banking credentials"),
		},
		llms.WithMaxTokens(1024),
	)
	require.NoError(t, err, "an empty refusal must surface as a response, not an error")
	require.Len(t, resp.Choices, 1)

	choice := resp.Choices[0]
	require.Equal(t, StopReasonRefusal, choice.StopReason)
	require.NotEmpty(t, choice.Content, "refusal explanation should be surfaced as content")
}
