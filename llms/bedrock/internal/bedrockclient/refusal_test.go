package bedrockclient

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

// guardrailBody is a real InvokeModel response body captured from
// us.anthropic.claude-opus-5 on Bedrock (us-east-1) for a request that tripped the
// cyber safety guardrail: empty content, stop_reason "refusal", reason in
// stop_details. The extra usage fields Bedrock adds (cache_creation, service_tier,
// output_tokens_details) are kept verbatim to confirm the struct tolerates them.
const guardrailBody = `{
  "model": "claude-opus-5",
  "id": "msg_bdrk_test",
  "type": "message",
  "role": "assistant",
  "content": [],
  "stop_reason": "refusal",
  "stop_sequence": null,
  "stop_details": {"type": "refusal", "category": "cyber", "explanation": "This request triggered restrictions on violative cyber content and was blocked under Anthropic's Usage Policy. To learn more, see https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback."},
  "usage": {"input_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}, "output_tokens": 4, "output_tokens_details": {"thinking_tokens": 0}, "service_tier": "standard"}
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
