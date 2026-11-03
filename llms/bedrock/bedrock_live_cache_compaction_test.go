package bedrock_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/bedrock"
)

// These tests call real Bedrock and cost money (~$0.50 per run on Sonnet 4.6).
// They run only when BEDROCK_LIVE_MODEL is set, using the default AWS credential
// chain and region:
//
//	AWS_PROFILE=... AWS_REGION=us-east-1 \
//	BEDROCK_LIVE_MODEL=us.anthropic.claude-sonnet-4-6 \
//	go test ./llms/bedrock/ -run TestLive -v
func liveLLM(t *testing.T) (*bedrock.LLM, string) {
	t.Helper()
	model := os.Getenv("BEDROCK_LIVE_MODEL")
	if model == "" {
		t.Skip("set BEDROCK_LIVE_MODEL to run live Bedrock tests")
	}
	llm, err := bedrock.New(bedrock.WithModel(model), bedrock.WithModelProvider("anthropic"))
	require.NoError(t, err)
	return llm, model
}

// filler returns roughly n tokens of distinct, non-repeating text (~10 tokens a line).
func filler(n int) string {
	var sb strings.Builder
	for i := 0; i < n/10; i++ {
		fmt.Fprintf(&sb, "Log %d: user u%d logged in from 10.0.%d.%d at step %d.\n", i, i%97, i%251, i%13, i)
	}
	return sb.String()
}

func TestLiveCompactionFires(t *testing.T) {
	llm, model := liveLLM(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// ~60k input tokens, above the 50k minimum trigger.
	msgs := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "You are a security analyst."),
		llms.TextParts(llms.ChatMessageTypeHuman, filler(60_000)+"\nSummarize which users logged in most."),
	}
	resp, err := llm.GenerateContent(ctx, msgs,
		llms.WithModel(model),
		llms.WithMaxTokens(4096),
		bedrock.WithCompaction(&bedrock.CompactionConfig{
			TriggerTokens:        50_000,
			PauseAfterCompaction: true,
			Instructions:         "Summarize the conversation. Respond with text only.",
		}),
	)
	require.NoError(t, err)
	choice := resp.Choices[0]
	t.Logf("stop_reason=%q generationInfo=%v", choice.StopReason, choice.GenerationInfo)

	require.Equal(t, "compaction", choice.StopReason, "compaction should pause the request")
	summary, _ := choice.GenerationInfo["CompactionSummary"].(string)
	require.NotEmpty(t, summary, "the summary should be surfaced in GenerationInfo")
	t.Logf("summary: %.300s", summary)
}

func TestLivePromptCacheHits(t *testing.T) {
	llm, model := liveLLM(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// A unique prefix per run so the first call is a guaranteed cache write.
	system := fmt.Sprintf("Run %d. You are a security analyst.\n%s", time.Now().UnixNano(), filler(5_000))
	msgs := []llms.MessageContent{
		{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{
			llms.WithCacheControl(llms.TextPart(system), &llms.CacheControl{Type: "ephemeral"}),
		}},
		llms.TextParts(llms.ChatMessageTypeHuman, "Reply with OK."),
	}

	var reads []any
	for i := range 2 {
		resp, err := llm.GenerateContent(ctx, msgs, llms.WithModel(model), llms.WithMaxTokens(16))
		require.NoError(t, err)
		info := resp.Choices[0].GenerationInfo
		t.Logf("call %d: %v", i+1, info)
		reads = append(reads, info["CacheReadInputTokens"])
		if i == 0 {
			require.Positive(t, info["CacheCreationInputTokens"], "first call should write the cache")
		}
	}
	require.Positive(t, reads[1], "second call should read the cached system prompt")
}
