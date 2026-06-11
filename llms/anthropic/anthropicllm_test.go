package anthropic

import (
	"os"
	"testing"

	"github.com/tmc/langchaingo/llms"
	anthropicclient "github.com/tmc/langchaingo/llms/anthropic/internal/anthropicclient"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name     string
		envToken string
		opts     []Option
		wantErr  bool
	}{
		{
			name:     "with token from env",
			envToken: "test-token",
			opts:     []Option{},
			wantErr:  false,
		},
		{
			name:     "with token option",
			envToken: "",
			opts:     []Option{WithToken("test-token")},
			wantErr:  false,
		},
		{
			name:     "missing token",
			envToken: "",
			opts:     []Option{},
			wantErr:  true,
		},
		{
			name:     "with all options",
			envToken: "test-token",
			opts: []Option{
				WithModel("claude-3-opus-20240229"),
				WithBaseURL("https://api.example.com"),
				WithAnthropicBetaHeader("max-tokens-3-5-sonnet-2024-07-15"),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			os.Setenv("ANTHROPIC_API_KEY", tt.envToken)

			llm, err := New(tt.opts...)
			if (err != nil) != tt.wantErr {
				t.Errorf("New() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && llm == nil {
				t.Error("New() returned nil LLM without error")
			}
		})
	}
}

func TestProcessMessages(t *testing.T) {
	tests := []struct {
		name       string
		messages   []llms.MessageContent
		wantLen    int
		wantSystem string
		wantErr    bool
	}{
		{
			name: "basic text message",
			messages: []llms.MessageContent{
				{
					Role: llms.ChatMessageTypeHuman,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "Hello"},
					},
				},
			},
			wantLen:    1,
			wantSystem: "",
			wantErr:    false,
		},
		{
			name: "system message",
			messages: []llms.MessageContent{
				{
					Role: llms.ChatMessageTypeSystem,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "You are helpful"},
					},
				},
				{
					Role: llms.ChatMessageTypeHuman,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "Hi"},
					},
				},
			},
			wantLen:    1,
			wantSystem: "You are helpful",
			wantErr:    false,
		},
		{
			name: "ai and human messages",
			messages: []llms.MessageContent{
				{
					Role: llms.ChatMessageTypeHuman,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "Hello"},
					},
				},
				{
					Role: llms.ChatMessageTypeAI,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "Hi there!"},
					},
				},
			},
			wantLen:    2,
			wantSystem: "",
			wantErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, systemPrompt, err := processMessages(tt.messages)
			if (err != nil) != tt.wantErr {
				t.Errorf("processMessages() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if len(result) != tt.wantLen {
					t.Errorf("processMessages() returned %d messages, want %d", len(result), tt.wantLen)
				}
				if systemPrompt != tt.wantSystem {
					t.Errorf("processMessages() system prompt = %q, want %q", systemPrompt, tt.wantSystem)
				}
			}
		})
	}
}

func TestHandleAIMessage(t *testing.T) {
	toolCall := llms.ToolCall{
		ID:           "toolu_1",
		FunctionCall: &llms.FunctionCall{Name: "get_weather", Arguments: `{"location":"SF"}`},
	}

	tests := []struct {
		name      string
		parts     []llms.ContentPart
		wantTypes []string // expected content block types, in order
		wantErr   bool
	}{
		{
			name:      "text only",
			parts:     []llms.ContentPart{llms.TextContent{Text: "hello"}},
			wantTypes: []string{"text"},
		},
		{
			name:      "tool call only",
			parts:     []llms.ContentPart{toolCall},
			wantTypes: []string{"tool_use"},
		},
		{
			name:      "text followed by tool call",
			parts:     []llms.ContentPart{llms.TextContent{Text: "let me check"}, toolCall},
			wantTypes: []string{"text", "tool_use"},
		},
		{
			name:      "empty text is skipped alongside tool call",
			parts:     []llms.ContentPart{llms.TextContent{Text: ""}, toolCall},
			wantTypes: []string{"tool_use"},
		},
		{
			name:    "unsupported part type errors",
			parts:   []llms.ContentPart{llms.ImageURLContent{URL: "http://example.com/x.png"}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: tt.parts}
			got, err := handleAIMessage(msg)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("handleAIMessage() expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("handleAIMessage() unexpected error: %v", err)
			}
			if got.Role != RoleAssistant {
				t.Errorf("role = %q, want %q", got.Role, RoleAssistant)
			}
			contents, ok := got.Content.([]anthropicclient.Content)
			if !ok {
				t.Fatalf("content is %T, want []anthropicclient.Content", got.Content)
			}
			if len(contents) != len(tt.wantTypes) {
				t.Fatalf("got %d content blocks, want %d", len(contents), len(tt.wantTypes))
			}
			for i, want := range tt.wantTypes {
				if got := contents[i].GetType(); got != want {
					t.Errorf("content[%d] type = %q, want %q", i, got, want)
				}
			}
		})
	}
}

func TestToolsToTools(t *testing.T) {
	tools := []llms.Tool{
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The location to get weather for",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	result := toolsToTools(tools)

	if len(result) != 1 {
		t.Fatalf("toolsToTools() returned %d tools, want 1", len(result))
	}
	if result[0].Name != "get_weather" {
		t.Errorf("toolsToTools() tool name = %q, want %q", result[0].Name, "get_weather")
	}
	if result[0].Description != "Get the weather for a location" {
		t.Errorf("toolsToTools() tool description = %q, want %q", result[0].Description, "Get the weather for a location")
	}
}

func TestOptions(t *testing.T) {
	t.Run("WithModel", func(t *testing.T) {
		opts := &options{}
		WithModel("claude-3-opus")(opts)
		if opts.model != "claude-3-opus" {
			t.Errorf("WithModel() got %s, want claude-3-opus", opts.model)
		}
	})

	t.Run("WithToken", func(t *testing.T) {
		opts := &options{}
		WithToken("test-token")(opts)
		if opts.token != "test-token" {
			t.Errorf("WithToken() got %s, want test-token", opts.token)
		}
	})

	t.Run("WithBaseURL", func(t *testing.T) {
		opts := &options{}
		WithBaseURL("https://test.com")(opts)
		if opts.baseURL != "https://test.com" {
			t.Errorf("WithBaseURL() got %s, want https://test.com", opts.baseURL)
		}
	})

	t.Run("WithAnthropicBetaHeader", func(t *testing.T) {
		opts := &options{}
		WithAnthropicBetaHeader("test-beta")(opts)
		if opts.anthropicBetaHeader != "test-beta" {
			t.Errorf("WithAnthropicBetaHeader() got %s, want test-beta", opts.anthropicBetaHeader)
		}
	})

	t.Run("WithLegacyTextCompletionsAPI", func(t *testing.T) {
		opts := &options{}
		WithLegacyTextCompletionsAPI()(opts)
		if !opts.useLegacyTextCompletionsAPI {
			t.Error("WithLegacyTextCompletionsAPI() did not set flag")
		}
	})
}

func TestEphemeralCacheOptions(t *testing.T) {
	cache := EphemeralCache()
	if cache.Type != "ephemeral" {
		t.Errorf("EphemeralCache() type = %s, want ephemeral", cache.Type)
	}

	cacheOneHour := EphemeralCacheOneHour()
	if cacheOneHour.Type != "ephemeral" {
		t.Errorf("EphemeralCacheOneHour() type = %s, want ephemeral", cacheOneHour.Type)
	}
}

func TestWithCompaction(t *testing.T) {
	t.Run("default trigger", func(t *testing.T) {
		cfg := &CompactionConfig{}
		opt := WithCompaction(cfg)

		var opts llms.CallOptions
		opt(&opts)

		if opts.Metadata == nil {
			t.Fatal("metadata should be initialized")
		}
		stored, ok := opts.Metadata["anthropic:compaction"].(*CompactionConfig)
		if !ok || stored == nil {
			t.Fatal("compaction config should be stored in metadata")
		}
		if stored.TriggerTokens != 0 {
			t.Errorf("TriggerTokens = %d, want 0 (default applied at request time)", stored.TriggerTokens)
		}
	})

	t.Run("custom config", func(t *testing.T) {
		cfg := &CompactionConfig{
			TriggerTokens:        150000,
			PauseAfterCompaction: true,
			Instructions:         "custom summary instructions",
		}
		opt := WithCompaction(cfg)

		var opts llms.CallOptions
		opt(&opts)

		stored := opts.Metadata["anthropic:compaction"].(*CompactionConfig)
		if stored.TriggerTokens != 150000 {
			t.Errorf("TriggerTokens = %d, want 150000", stored.TriggerTokens)
		}
		if !stored.PauseAfterCompaction {
			t.Error("PauseAfterCompaction should be true")
		}
		if stored.Instructions != "custom summary instructions" {
			t.Errorf("Instructions = %q, want custom", stored.Instructions)
		}
	})
}

func TestProcessAnthropicResponse_Compaction(t *testing.T) {
	// Test via JSON unmarshal since anthropicclient is internal
	jsonData := `{
		"content": [{"type": "compaction", "content": "Summary of the conversation so far."}],
		"id": "msg_test",
		"model": "claude-sonnet-4-6",
		"role": "assistant",
		"stop_reason": "compaction",
		"stop_sequence": null,
		"type": "message",
		"usage": {"input_tokens": 180000, "output_tokens": 3500}
	}`

	var result anthropicclient.MessageResponsePayload
	err := result.UnmarshalJSON([]byte(jsonData))
	if err != nil {
		t.Fatalf("UnmarshalJSON() error = %v", err)
	}

	resp, err := processAnthropicResponse(&result)
	if err != nil {
		t.Fatalf("processAnthropicResponse() error = %v", err)
	}

	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}

	choice := resp.Choices[0]
	if choice.StopReason != "compaction" {
		t.Errorf("StopReason = %q, want compaction", choice.StopReason)
	}
	if choice.Content != "Summary of the conversation so far." {
		t.Errorf("Content = %q, want summary text", choice.Content)
	}

	summary, ok := choice.GenerationInfo["CompactionSummary"].(string)
	if !ok || summary != "Summary of the conversation so far." {
		t.Errorf("CompactionSummary = %q, want summary text", summary)
	}

	inputTokens, ok := choice.GenerationInfo["InputTokens"].(int)
	if !ok || inputTokens != 180000 {
		t.Errorf("InputTokens = %v, want 180000", choice.GenerationInfo["InputTokens"])
	}
}

func TestCall(t *testing.T) {
	// Test that Call delegates to GenerateContent
	t.Skip("Call() requires integration testing with mock client")
}

func TestGenerateMessagesContent_EmptyContent(t *testing.T) {
	// This test demonstrates the need for checking len(result.Content) == 0
	// Without the fix, accessing result.Content[0] would panic when Anthropic
	// returns a response with nil or empty content (addresses issue #993)
	t.Skip("Requires mock client - would demonstrate panic without len(result.Content) == 0 check")
}
