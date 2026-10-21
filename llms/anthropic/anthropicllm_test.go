package anthropic

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic/internal/anthropicclient"
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
			t.Setenv("ANTHROPIC_API_KEY", tt.envToken)

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

func TestSupportsReasoningForCurrentClaudeModels(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{model: "claude-opus-5-5", want: true},
		{model: "claude-sonnet-5", want: true},
		{model: "claude-fable-5-1", want: true},
		{model: "claude-mythos-5-1", want: true},
		{model: "global.anthropic.claude-opus-5-5", want: true},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := supportsReasoningForModel(tt.model); got != tt.want {
				t.Errorf("supportsReasoningForModel(%q) = %t, want %t", tt.model, got, tt.want)
			}
		})
	}
}

func TestProcessMessages(t *testing.T) {
	tests := []struct {
		name       string
		messages   []llms.MessageContent
		wantLen    int
		wantSystem any
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
		{
			name: "cached system message becomes a cache_control block",
			messages: []llms.MessageContent{
				{
					Role: llms.ChatMessageTypeSystem,
					Parts: []llms.ContentPart{
						llms.WithCacheControl(
							llms.TextContent{Text: "You are helpful"},
							&llms.CacheControl{Type: "ephemeral"},
						),
					},
				},
				{
					Role:  llms.ChatMessageTypeHuman,
					Parts: []llms.ContentPart{llms.TextContent{Text: "Hi"}},
				},
			},
			wantLen: 1,
			wantSystem: []anthropicclient.TextContent{
				{
					Type:         "text",
					Text:         "You are helpful",
					CacheControl: &anthropicclient.CacheControl{Type: "ephemeral"},
				},
			},
			wantErr: false,
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
				if !reflect.DeepEqual(systemPrompt, tt.wantSystem) {
					t.Errorf("processMessages() system prompt = %#v, want %#v", systemPrompt, tt.wantSystem)
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

func TestProcessMessagesSystem(t *testing.T) {
	tests := []struct {
		name     string
		messages []llms.MessageContent
		want     any
	}{
		{
			name: "multi-part system concatenates",
			messages: []llms.MessageContent{
				{
					Role: llms.ChatMessageTypeSystem,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "You are helpful."},
						llms.TextContent{Text: " Be brief."},
					},
				},
			},
			want: "You are helpful. Be brief.",
		},
		{
			name: "multiple system messages concatenate",
			messages: []llms.MessageContent{
				{
					Role:  llms.ChatMessageTypeSystem,
					Parts: []llms.ContentPart{llms.TextContent{Text: "One."}},
				},
				{
					Role:  llms.ChatMessageTypeSystem,
					Parts: []llms.ContentPart{llms.TextContent{Text: "Two."}},
				},
			},
			want: "One.Two.",
		},
		{
			name: "cache control produces block list",
			messages: []llms.MessageContent{
				{
					Role: llms.ChatMessageTypeSystem,
					Parts: []llms.ContentPart{
						llms.WithCacheControl(
							llms.TextContent{Text: "Large context"},
							&llms.CacheControl{Type: "ephemeral"},
						),
						llms.TextContent{Text: "Question preamble"},
					},
				},
			},
			want: []anthropicclient.TextContent{
				{Type: "text", Text: "Large context", CacheControl: &anthropicclient.CacheControl{Type: "ephemeral"}},
				{Type: "text", Text: "Question preamble"},
			},
		},
		{
			name: "duration maps to 1h ttl and empty type defaults",
			messages: []llms.MessageContent{
				{
					Role: llms.ChatMessageTypeSystem,
					Parts: []llms.ContentPart{
						llms.WithCacheControl(
							llms.TextContent{Text: "Cached"},
							&llms.CacheControl{Duration: time.Hour},
						),
					},
				},
			},
			want: []anthropicclient.TextContent{
				{Type: "text", Text: "Cached", CacheControl: &anthropicclient.CacheControl{Type: "ephemeral", TTL: "1h"}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, system, err := processMessages(tt.messages)
			if err != nil {
				t.Fatalf("processMessages() error = %v", err)
			}
			if !reflect.DeepEqual(system, tt.want) {
				t.Errorf("processMessages() system = %#v, want %#v", system, tt.want)
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

func TestProcessAnthropicResponse_AggregatesMessageBlocks(t *testing.T) {
	tests := []struct {
		name             string
		content          string
		wantText         string
		wantThinking     string
		wantSignature    string
		wantToolNames    []string
		wantOutputTokens int
	}{
		{
			name:             "text only",
			content:          `[{"type":"text","text":"answer"}]`,
			wantText:         "answer",
			wantOutputTokens: 6,
		},
		{
			name: "thinking before text",
			content: `[
				{"type":"thinking","thinking":"analysis","signature":"signed"},
				{"type":"text","text":"answer"}
			]`,
			wantText:         "answer",
			wantThinking:     "analysis",
			wantSignature:    "signed",
			wantOutputTokens: 7,
		},
		{
			name: "text and parallel tools",
			content: `[
				{"type":"text","text":"checking"},
				{"type":"tool_use","id":"tool-1","name":"first","input":{"value":1}},
				{"type":"tool_use","id":"tool-2","name":"second","input":{"value":2}}
			]`,
			wantText:         "checking",
			wantToolNames:    []string{"first", "second"},
			wantOutputTokens: 8,
		},
		{
			name: "multiple text blocks",
			content: `[
				{"type":"text","text":"first"},
				{"type":"text","text":" second"}
			]`,
			wantText:         "first second",
			wantOutputTokens: 9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jsonData := fmt.Sprintf(`{
				"content": %s,
				"id": "msg_test",
				"model": "claude-test",
				"role": "assistant",
				"stop_reason": "end_turn",
				"type": "message",
				"usage": {"input_tokens": 5, "output_tokens": %d}
			}`, tt.content, tt.wantOutputTokens)

			var result anthropicclient.MessageResponsePayload
			if err := result.UnmarshalJSON([]byte(jsonData)); err != nil {
				t.Fatalf("UnmarshalJSON() error = %v", err)
			}

			resp, err := processAnthropicResponse(&result)
			if err != nil {
				t.Fatalf("processAnthropicResponse() error = %v", err)
			}
			if len(resp.Choices) != 1 {
				t.Fatalf("len(Choices) = %d, want 1", len(resp.Choices))
			}
			choice := resp.Choices[0]
			if choice.Content != tt.wantText {
				t.Errorf("Content = %q, want %q", choice.Content, tt.wantText)
			}
			if got, _ := choice.GenerationInfo["ThinkingContent"].(string); got != tt.wantThinking {
				t.Errorf("ThinkingContent = %q, want %q", got, tt.wantThinking)
			}
			if got, _ := choice.GenerationInfo["ThinkingSignature"].(string); got != tt.wantSignature {
				t.Errorf("ThinkingSignature = %q, want %q", got, tt.wantSignature)
			}
			if got, _ := choice.GenerationInfo["OutputTokens"].(int); got != tt.wantOutputTokens {
				t.Errorf("OutputTokens = %d, want %d", got, tt.wantOutputTokens)
			}
			if len(choice.ToolCalls) != len(tt.wantToolNames) {
				t.Fatalf("len(ToolCalls) = %d, want %d", len(choice.ToolCalls), len(tt.wantToolNames))
			}
			for i, wantName := range tt.wantToolNames {
				if got := choice.ToolCalls[i].FunctionCall.Name; got != wantName {
					t.Errorf("ToolCalls[%d].Name = %q, want %q", i, got, wantName)
				}
			}
		})
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

func TestSupportsReasoningForModel(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"claude-fable-5", true},
		{"claude-opus-4-8", true},
		{"claude-opus-4-7", true},
		{"claude-opus-4-6", true},
		{"claude-sonnet-4-6", true},
		{"claude-3-7-sonnet-20250219", true},
		{"claude-3-haiku-20240307", false},
		{"claude-3-5-sonnet-20240620", false},
		// The empty name gets newest-generation defaults; the provider
		// resolves the client default before lookup.
		{"", true},
	}
	for _, tt := range tests {
		if got := supportsReasoningForModel(tt.model); got != tt.want {
			t.Errorf("supportsReasoningForModel(%q) = %v, want %v", tt.model, got, tt.want)
		}
	}
}

func TestAdaptiveThinkingOnly(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"claude-fable-5", true},
		{"claude-opus-4-7", true},
		{"claude-opus-4-8", true},
		{"claude-opus-4-6", false},
		{"claude-sonnet-4-6", false},
		{"claude-3-7-sonnet-20250219", false},
		{"claude-3-haiku-20240307", false},
		// The empty name gets newest-generation defaults; the provider
		// resolves the client default before lookup.
		{"", true},
	}
	for _, tt := range tests {
		if got := adaptiveThinkingOnly(tt.model); got != tt.want {
			t.Errorf("adaptiveThinkingOnly(%q) = %v, want %v", tt.model, got, tt.want)
		}
	}
}

func TestExtractThinkingOptions(t *testing.T) {
	tests := []extractThinkingOptionsTest{
		{
			name:         "no thinking config",
			model:        "claude-fable-5",
			config:       nil,
			wantThinking: nil,
		},
		{
			// A client without a model targets the client default,
			// which supports thinking; the config must not be dropped.
			name:         "empty model resolves default",
			model:        "",
			config:       &llms.ThinkingConfig{Mode: llms.ThinkingModeMedium},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "enabled", BudgetTokens: 2048},
		},
		{
			name:         "fable auto mode uses adaptive",
			model:        "claude-fable-5",
			config:       &llms.ThinkingConfig{Mode: llms.ThinkingModeAuto},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "adaptive"},
		},
		{
			name:         "fable drops explicit budget",
			model:        "claude-fable-5",
			config:       &llms.ThinkingConfig{BudgetTokens: 8192},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "adaptive"},
		},
		{
			name:         "fable mode none omits thinking entirely",
			model:        "claude-fable-5",
			config:       &llms.ThinkingConfig{Mode: llms.ThinkingModeNone},
			wantThinking: nil,
		},
		{
			name:         "fable skips interleaved thinking header",
			model:        "claude-fable-5",
			config:       &llms.ThinkingConfig{Mode: llms.ThinkingModeAuto, InterleaveThinking: true},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "adaptive"},
		},
		{
			name:         "opus 4.7 uses adaptive",
			model:        "claude-opus-4-7",
			config:       &llms.ThinkingConfig{BudgetTokens: 4096},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "adaptive"},
		},
		{
			name:         "older model keeps budget",
			model:        "claude-sonnet-4-6",
			config:       &llms.ThinkingConfig{BudgetTokens: 8192},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "enabled", BudgetTokens: 8192},
		},
		{
			name:         "older model keeps interleaved thinking header",
			model:        "claude-sonnet-4-6",
			config:       &llms.ThinkingConfig{BudgetTokens: 8192, InterleaveThinking: true},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "enabled", BudgetTokens: 8192},
			wantHeaders:  []string{"interleaved-thinking-2025-05-14"},
		},
		{
			name:         "older model clamps budget to minimum",
			model:        "claude-3-7-sonnet-20250219",
			config:       &llms.ThinkingConfig{BudgetTokens: 100},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "enabled", BudgetTokens: 1024},
		},
		{
			name:         "non-reasoning model ignores thinking",
			model:        "claude-3-haiku-20240307",
			config:       &llms.ThinkingConfig{BudgetTokens: 8192},
			wantThinking: nil,
		},
	}
	runExtractThinkingOptionsTests(t, tests)
}

func TestExtractThinkingOptionsEffortDisplay(t *testing.T) {
	tests := []extractThinkingOptionsTest{
		{
			name:         "fable maps effort to output config",
			model:        "claude-fable-5",
			config:       &llms.ThinkingConfig{Mode: llms.ThinkingModeAuto, Effort: "xhigh"},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "adaptive"},
			wantOutput:   &anthropicclient.OutputConfig{Effort: "xhigh"},
		},
		{
			name:         "fable passes display through",
			model:        "claude-fable-5",
			config:       &llms.ThinkingConfig{Mode: llms.ThinkingModeAuto, Display: "summarized"},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "adaptive", Display: "summarized"},
		},
		{
			name:         "fable effort without thinking mode",
			model:        "claude-fable-5",
			config:       &llms.ThinkingConfig{Mode: llms.ThinkingModeNone, Effort: "low"},
			wantThinking: nil,
			wantOutput:   &anthropicclient.OutputConfig{Effort: "low"},
		},
		{
			name:         "sonnet 4.6 maps effort alongside budget",
			model:        "claude-sonnet-4-6",
			config:       &llms.ThinkingConfig{BudgetTokens: 8192, Effort: "high"},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "enabled", BudgetTokens: 8192},
			wantOutput:   &anthropicclient.OutputConfig{Effort: "high"},
		},
		{
			name:         "budget-only model drops effort",
			model:        "claude-3-7-sonnet-20250219",
			config:       &llms.ThinkingConfig{BudgetTokens: 8192, Effort: "high"},
			wantThinking: &anthropicclient.ThinkingConfig{Type: "enabled", BudgetTokens: 8192},
		},
	}
	runExtractThinkingOptionsTests(t, tests)
}

type extractThinkingOptionsTest struct {
	name         string
	model        string
	config       *llms.ThinkingConfig
	wantThinking *anthropicclient.ThinkingConfig
	wantHeaders  []string
	wantOutput   *anthropicclient.OutputConfig
}

func runExtractThinkingOptionsTests(t *testing.T, tests []extractThinkingOptionsTest) {
	t.Helper()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := &LLM{model: tt.model}
			opts := &llms.CallOptions{MaxTokens: 4096}
			if tt.config != nil {
				opts.Metadata = map[string]any{"thinking_config": tt.config}
			}

			headers, thinking, output := extractThinkingOptions(o, opts)
			assertThinkingConfig(t, thinking, tt.wantThinking)
			assertBetaHeaders(t, headers, tt.wantHeaders)
			assertOutputConfig(t, output, tt.wantOutput)
		})
	}
}

func assertThinkingConfig(t *testing.T, got, want *anthropicclient.ThinkingConfig) {
	t.Helper()
	if want == nil {
		if got != nil {
			t.Fatalf("extractThinkingOptions() thinking = %+v, want nil", got)
		}
		return
	}
	if got == nil {
		t.Fatalf("extractThinkingOptions() thinking = nil, want %+v", want)
	}
	if got.Type != want.Type {
		t.Errorf("thinking.Type = %q, want %q", got.Type, want.Type)
	}
	if got.BudgetTokens != want.BudgetTokens {
		t.Errorf("thinking.BudgetTokens = %d, want %d", got.BudgetTokens, want.BudgetTokens)
	}
	if got.Display != want.Display {
		t.Errorf("thinking.Display = %q, want %q", got.Display, want.Display)
	}
}

func assertOutputConfig(t *testing.T, got, want *anthropicclient.OutputConfig) {
	t.Helper()
	if want == nil {
		if got != nil {
			t.Fatalf("extractThinkingOptions() output = %+v, want nil", got)
		}
		return
	}
	if got == nil {
		t.Fatalf("extractThinkingOptions() output = nil, want %+v", want)
	}
	if got.Effort != want.Effort {
		t.Errorf("output.Effort = %q, want %q", got.Effort, want.Effort)
	}
}

func assertBetaHeaders(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("extractThinkingOptions() headers = %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("headers[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestModelCapabilities(t *testing.T) {
	tests := []struct {
		model string
		want  capabilities
	}{
		{"claude-fable-5", adaptiveCaps},
		{"claude-opus-4-8", adaptiveCaps},
		{"claude-opus-4-7", adaptiveCaps},
		{"claude-opus-4-6", capabilities{thinking: true, sampling: true, effort: true}},
		{"claude-sonnet-4-6", capabilities{thinking: true, sampling: true, effort: true}},
		{"claude-opus-4-20250514", budgetCaps},
		{"claude-3-7-sonnet-20250219", budgetCaps},
		{"claude-3-5-sonnet-20240620", legacyCaps},
		{"claude-3-haiku-20240307", legacyCaps},
		{"claude-2.1", legacyCaps},
		// Unlisted 4-x family members: minors past the last listed
		// release get newest-generation defaults, earlier ones keep
		// budget-era behavior.
		{"claude-opus-4-9", adaptiveCaps},
		{"claude-opus-4-5", budgetCaps},
		{"claude-opus-4-1-20250805", budgetCaps},
		{"claude-opus-4", budgetCaps},
		{"claude-sonnet-4-7", adaptiveCaps},
		{"claude-sonnet-4-5-20250929", budgetCaps},
		{"claude-haiku-4-5", budgetCaps},
		{"claude-haiku-4-6", adaptiveCaps},
		// Unknown Claude names get newest-generation defaults. The
		// empty name cannot reach here from the provider:
		// resolvedModel substitutes the client default first.
		{"", adaptiveCaps},
		{"claude-fable-6", adaptiveCaps},
		{"claude-omega-7", adaptiveCaps},
		{"claude-sonnet-5", adaptiveCaps},
		// Names that are not Claude models at all — gateway aliases,
		// fine-tunes, compatible endpoints — keep the long-standing
		// wire contract so caller sampling parameters are not dropped.
		{"my-gateway/llama-3-70b", budgetCaps},
		{"gpt-4o", budgetCaps},
		{"some-finetune-v2", budgetCaps},
	}
	for _, tt := range tests {
		if got := modelCapabilities(tt.model); got != tt.want {
			t.Errorf("modelCapabilities(%q) = %+v, want %+v", tt.model, got, tt.want)
		}
	}
}

func TestHandleAIMessageAllParts(t *testing.T) {
	msg := llms.MessageContent{
		Role: llms.ChatMessageTypeAI,
		Parts: []llms.ContentPart{
			llms.ThinkingContent{Thinking: "let me think", Signature: "sig123"},
			llms.RedactedThinkingContent{Data: "opaque"},
			llms.TextContent{Text: "calling the tool"},
			llms.ToolCall{
				ID: "tool-1",
				FunctionCall: &llms.FunctionCall{
					Name:      "get_weather",
					Arguments: `{"location":"sf"}`,
				},
			},
		},
	}

	got, err := handleAIMessage(msg)
	if err != nil {
		t.Fatalf("handleAIMessage() error = %v", err)
	}
	if got.Role != RoleAssistant {
		t.Errorf("role = %q, want %q", got.Role, RoleAssistant)
	}
	contents, ok := got.Content.([]anthropicclient.Content)
	if !ok {
		t.Fatalf("content type = %T, want []anthropicclient.Content", got.Content)
	}
	if len(contents) != 4 {
		t.Fatalf("len(contents) = %d, want 4", len(contents))
	}
	wantTypes := []string{"thinking", "redacted_thinking", "text", "tool_use"}
	for i, want := range wantTypes {
		if contents[i].GetType() != want {
			t.Errorf("contents[%d].GetType() = %q, want %q", i, contents[i].GetType(), want)
		}
	}
	thinking, ok := contents[0].(*anthropicclient.ThinkingContent)
	if !ok {
		t.Fatalf("contents[0] type = %T, want *ThinkingContent", contents[0])
	}
	if thinking.Signature != "sig123" {
		t.Errorf("thinking signature = %q, want %q", thinking.Signature, "sig123")
	}
}

func TestHandleToolMessageAllParts(t *testing.T) {
	msg := llms.MessageContent{
		Role: llms.ChatMessageTypeTool,
		Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "tool-1", Content: "sunny"},
			llms.ToolCallResponse{ToolCallID: "tool-2", Content: "72F"},
		},
	}

	got, err := handleToolMessage(msg)
	if err != nil {
		t.Fatalf("handleToolMessage() error = %v", err)
	}
	if got.Role != RoleUser {
		t.Errorf("role = %q, want %q", got.Role, RoleUser)
	}
	contents, ok := got.Content.([]anthropicclient.Content)
	if !ok {
		t.Fatalf("content type = %T, want []anthropicclient.Content", got.Content)
	}
	if len(contents) != 2 {
		t.Fatalf("len(contents) = %d, want 2", len(contents))
	}
	second, ok := contents[1].(anthropicclient.ToolResultContent)
	if !ok {
		t.Fatalf("contents[1] type = %T, want ToolResultContent", contents[1])
	}
	if second.ToolUseID != "tool-2" || second.Content != "72F" {
		t.Errorf("contents[1] = %+v, want tool-2/72F", second)
	}
}

func TestProcessAnthropicResponseParts(t *testing.T) {
	result := &anthropicclient.MessageResponsePayload{
		Content: []anthropicclient.Content{
			&anthropicclient.ThinkingContent{Type: "thinking", Thinking: "hmm", Signature: "sig"},
			&anthropicclient.RedactedThinkingContent{Type: "redacted_thinking", Data: "opaque"},
			&anthropicclient.TextContent{Type: "text", Text: "answer"},
			&anthropicclient.ToolUseContent{Type: "tool_use", ID: "t1", Name: "f", Input: map[string]interface{}{"a": "b"}},
		},
	}

	resp, err := processAnthropicResponse(result)
	if err != nil {
		t.Fatalf("processAnthropicResponse() error = %v", err)
	}
	// The blocks of one response are one choice, with Parts in block order.
	if len(resp.Choices) != 1 {
		t.Fatalf("len(choices) = %d, want 1", len(resp.Choices))
	}
	choice := resp.Choices[0]
	if len(choice.Parts) != 4 {
		t.Fatalf("len(Parts) = %d, want 4", len(choice.Parts))
	}

	thinkingPart, ok := choice.Parts[0].(llms.ThinkingContent)
	if !ok {
		t.Fatalf("Parts[0] type = %T, want llms.ThinkingContent", choice.Parts[0])
	}
	if thinkingPart.Thinking != "hmm" || thinkingPart.Signature != "sig" {
		t.Errorf("thinking part = %+v, want hmm/sig", thinkingPart)
	}
	if choice.ReasoningContent != "hmm" {
		t.Errorf("ReasoningContent = %q, want %q", choice.ReasoningContent, "hmm")
	}
	if _, ok := choice.Parts[1].(llms.RedactedThinkingContent); !ok {
		t.Fatalf("Parts[1] type = %T, want llms.RedactedThinkingContent", choice.Parts[1])
	}
	if _, ok := choice.Parts[2].(llms.TextContent); !ok {
		t.Fatalf("Parts[2] type = %T, want llms.TextContent", choice.Parts[2])
	}
	if _, ok := choice.Parts[3].(llms.ToolCall); !ok {
		t.Fatalf("Parts[3] type = %T, want llms.ToolCall", choice.Parts[3])
	}
	if choice.Content != "answer" || len(choice.ToolCalls) != 1 {
		t.Errorf("Content = %q, ToolCalls = %d; want answer, 1", choice.Content, len(choice.ToolCalls))
	}

	// Round-trip: the assistant message must convert back into a valid
	// request message with all four blocks in order.
	aiParts := resp.AssistantMessage().Parts
	aiMsg, err := handleAIMessage(llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: aiParts})
	if err != nil {
		t.Fatalf("handleAIMessage(round-trip) error = %v", err)
	}
	contents := aiMsg.Content.([]anthropicclient.Content)
	if len(contents) != 4 {
		t.Fatalf("round-trip len(contents) = %d, want 4", len(contents))
	}
	rt, ok := contents[0].(*anthropicclient.ThinkingContent)
	if !ok || rt.Signature != "sig" {
		t.Errorf("round-trip thinking = %+v, want signature preserved", contents[0])
	}
}
