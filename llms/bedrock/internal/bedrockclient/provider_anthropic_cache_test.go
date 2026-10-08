package bedrockclient

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
)

// requestJSON builds the InvokeModel body and decodes it back into a generic map
// so assertions check the exact wire shape Bedrock receives.
func requestJSON(t *testing.T, messages []Message, options llms.CallOptions) map[string]any {
	t.Helper()
	input, err := buildAnthropicInput(messages, options)
	require.NoError(t, err)
	body, err := json.Marshal(input)
	require.NoError(t, err)
	var out map[string]any
	require.NoError(t, json.Unmarshal(body, &out))
	return out
}

func TestBuildAnthropicInput_NoCacheOrBetaByDefault(t *testing.T) {
	body := requestJSON(t, []Message{
		{Role: llms.ChatMessageTypeSystem, Type: "text", Content: "You are helpful"},
		{Role: llms.ChatMessageTypeHuman, Type: "text", Content: "Hi"},
	}, llms.CallOptions{})

	require.Equal(t, "You are helpful", body["system"], "uncached system prompt stays a plain string")
	require.NotContains(t, body, "anthropic_beta")
	require.NotContains(t, body, "context_management")
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	require.NotContains(t, content[0].(map[string]any), "cache_control")
}

func TestBuildAnthropicInput_NoSystemPromptOmitsField(t *testing.T) {
	body := requestJSON(t, []Message{
		{Role: llms.ChatMessageTypeHuman, Type: "text", Content: "Hi"},
	}, llms.CallOptions{})
	require.NotContains(t, body, "system")
}

func TestBuildAnthropicInput_CacheBreakpoints(t *testing.T) {
	ephemeral := &llms.CacheControl{Type: "ephemeral", Duration: 5 * time.Minute}
	body := requestJSON(t, []Message{
		{Role: llms.ChatMessageTypeSystem, Type: "text", Content: "System prefix", CacheControl: ephemeral},
		{Role: llms.ChatMessageTypeHuman, Type: "text", Content: "Investigate"},
		{
			Role: llms.ChatMessageTypeAI, Type: "tool_call",
			ToolCallID: "t1", ToolName: "search", ToolArgs: `{"q":"x"}`,
		},
		{
			Role: llms.ChatMessageTypeTool, Type: "tool_result", ToolUseID: "t1", Content: "rows",
			CacheControl: &llms.CacheControl{Type: "ephemeral", Duration: time.Hour},
		},
	}, llms.CallOptions{})

	require.Equal(t, []any{map[string]any{
		"type":          "text",
		"text":          "System prefix",
		"cache_control": map[string]any{"type": "ephemeral"},
	}}, body["system"], "a cached system prompt is sent as a block array; the 5m TTL is left implicit")

	msgs := body["messages"].([]any)
	require.Len(t, msgs, 3)
	toolResult := msgs[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	require.Equal(t, "tool_result", toolResult["type"])
	require.Equal(t, map[string]any{"type": "ephemeral", "ttl": "1h"}, toolResult["cache_control"])

	// Only the blocks that asked for a breakpoint carry one.
	human := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	require.NotContains(t, human, "cache_control")
}

func TestBuildAnthropicInput_Compaction(t *testing.T) {
	opts := llms.CallOptions{Metadata: map[string]any{
		MetadataKeyAnthropicBeta: []string{"some-beta", AnthropicBetaCompaction},
		MetadataKeyCompaction: &CompactionConfig{
			TriggerTokens:        10_000,
			Instructions:         "Keep IOCs",
			PauseAfterCompaction: true,
		},
	}}
	body := requestJSON(t, []Message{{Role: llms.ChatMessageTypeHuman, Type: "text", Content: "Hi"}}, opts)

	require.Equal(t, []any{"some-beta", AnthropicBetaCompaction}, body["anthropic_beta"], "betas are deduplicated")
	require.Equal(t, map[string]any{
		"edits": []any{map[string]any{
			"type":                   "compact_20260112",
			"trigger":                map[string]any{"type": "input_tokens", "value": float64(50_000)},
			"instructions":           "Keep IOCs",
			"pause_after_compaction": true,
		}},
	}, body["context_management"], "a trigger below the API minimum is raised to it")
}

func TestParseAnthropicResponse_CacheUsage(t *testing.T) {
	resp, err := parseAnthropicResponse([]byte(`{
		"type": "message", "role": "assistant",
		"content": [{"type": "text", "text": "done"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 12, "output_tokens": 30,
			"cache_creation_input_tokens": 200, "cache_read_input_tokens": 9000
		}
	}`))
	require.NoError(t, err)
	info := resp.Choices[0].GenerationInfo
	require.Equal(t, 12, info["input_tokens"])
	require.Equal(t, 30, info["output_tokens"])
	require.Equal(t, 200, info["CacheCreationInputTokens"])
	require.Equal(t, 9000, info["CacheReadInputTokens"])
}

func TestParseAnthropicResponse_CompactionPause(t *testing.T) {
	resp, err := parseAnthropicResponse([]byte(`{
		"type": "message", "role": "assistant",
		"content": [{"type": "compaction", "content": "<summary>analyst found 3 IOCs</summary>"}],
		"stop_reason": "compaction",
		"usage": {
			"input_tokens": 0, "output_tokens": 0,
			"iterations": [{"type": "compaction", "input_tokens": 180000, "output_tokens": 3500}]
		}
	}`))
	require.NoError(t, err)
	choice := resp.Choices[0]
	require.Equal(t, AnthropicCompletionReasonCompaction, choice.StopReason)
	require.Equal(t, "<summary>analyst found 3 IOCs</summary>", choice.GenerationInfo["CompactionSummary"])
	require.Equal(t, 180000, choice.GenerationInfo["CompactionInputTokens"])
	require.Equal(t, 3500, choice.GenerationInfo["CompactionOutputTokens"])
	require.Empty(t, choice.Content)
	require.Empty(t, choice.ToolCalls)
}
