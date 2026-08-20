package bedrockclient

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

// guardrailBody is the shape a Claude model on Bedrock returns when a request
// trips a safety guardrail: an empty content array, stop_reason "refusal", and the
// reason in stop_details.
const guardrailBody = `{
  "type": "message",
  "role": "assistant",
  "content": [],
  "stop_reason": "refusal",
  "stop_details": {"type": "refusal", "category": "cyber", "explanation": "Blocked under the Usage Policy."},
  "usage": {"input_tokens": 87, "output_tokens": 0}
}`

func TestAnthropicRefusalParsesStopDetails(t *testing.T) {
	var out anthropicTextGenerationOutput
	require.NoError(t, json.Unmarshal([]byte(guardrailBody), &out))

	require.Empty(t, out.Content)
	require.Equal(t, AnthropicCompletionReasonRefusal, out.StopReason)
	require.NotNil(t, out.StopDetails)
	require.Equal(t, "cyber", out.StopDetails.Category)
	require.Contains(t, out.StopDetails.Explanation, "Usage Policy")
}

func TestAnthropicRefusalChoice(t *testing.T) {
	t.Run("refusal with explanation", func(t *testing.T) {
		details := &anthropicStopDetails{Type: "refusal", Category: "cyber", Explanation: "Blocked."}
		choice := anthropicRefusalChoice(AnthropicCompletionReasonRefusal, details, 87, 0)
		require.NotNil(t, choice)
		require.Equal(t, "Blocked.", choice.Content)
		require.Equal(t, AnthropicCompletionReasonRefusal, choice.StopReason)
		require.Equal(t, "cyber", choice.GenerationInfo["refusal_category"])
	})

	t.Run("refusal without explanation falls back to a non-empty message", func(t *testing.T) {
		choice := anthropicRefusalChoice(AnthropicCompletionReasonRefusal, nil, 10, 0)
		require.NotNil(t, choice)
		require.NotEmpty(t, choice.Content)
	})

	t.Run("non-refusal empty response is not synthesized", func(t *testing.T) {
		require.Nil(t, anthropicRefusalChoice(AnthropicCompletionReasonMaxTokens, nil, 10, 0))
		require.Nil(t, anthropicRefusalChoice(AnthropicCompletionReasonEndTurn, nil, 10, 0))
	})
}
