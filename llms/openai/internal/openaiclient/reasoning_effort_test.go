package openaiclient

import (
	"encoding/json"
	"testing"
)

func TestGPTVersion(t *testing.T) {
	tests := []struct {
		model string
		major int
		minor int
		ok    bool
	}{
		{model: "gpt-5.6-terra", major: 5, minor: 6, ok: true},
		{model: "gpt-5.6-sol", major: 5, minor: 6, ok: true},
		{model: "GPT-5.6-Luna", major: 5, minor: 6, ok: true},
		{model: "gpt-5.6", major: 5, minor: 6, ok: true},
		{model: "gpt-5.5", major: 5, minor: 5, ok: true},
		{model: "gpt-5", major: 5, minor: 0, ok: true},
		{model: "gpt-6-orion", major: 6, minor: 0, ok: true},
		{model: "gpt-4", major: 4, minor: 0, ok: true},
		{model: "gpt-4-turbo", major: 4, minor: 0, ok: true},
		// "4o" is not a number, so it is not a versioned id we can reason about.
		{model: "gpt-4o"},
		{model: "gpt-4o-mini"},
		{model: "o3"},
		{model: "my-azure-deployment"},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			major, minor, ok := gptVersion(tt.model)
			if ok != tt.ok || major != tt.major || minor != tt.minor {
				t.Errorf("gptVersion(%q) = (%d, %d, %v), want (%d, %d, %v)",
					tt.model, major, minor, ok, tt.major, tt.minor, tt.ok)
			}
		})
	}
}

func TestNeedsReasoningOffForTools(t *testing.T) {
	tests := map[string]bool{
		"gpt-5.6-terra":       true,
		"gpt-5.6-sol":         true,
		"gpt-5.7":             true,
		"gpt-6":               true,
		"gpt-5.5":             false,
		"gpt-5.4-mini":        false,
		"gpt-5":               false,
		"gpt-4o":              false,
		"o3":                  false,
		"my-azure-deployment": false,
	}
	for model, want := range tests {
		t.Run(model, func(t *testing.T) {
			if got := needsReasoningOffForTools(model); got != want {
				t.Errorf("needsReasoningOffForTools(%q) = %v, want %v", model, got, want)
			}
		})
	}
}

// TestMarshalReasoningEffortWithTools covers the gpt-5.6 rejection of function
// tools on /v1/chat/completions unless reasoning_effort is "none".
func TestMarshalReasoningEffortWithTools(t *testing.T) {
	tool := Tool{Type: ToolTypeFunction, Function: FunctionDefinition{Name: "search"}}

	tests := []struct {
		name    string
		request ChatRequest
		want    string // "" means the field must be absent
	}{
		{
			name:    "gpt-5.6 with tools disables reasoning",
			request: ChatRequest{Model: "gpt-5.6-terra", Tools: []Tool{tool}},
			want:    "none",
		},
		{
			name:    "gpt-5.6 with deprecated functions disables reasoning",
			request: ChatRequest{Model: "gpt-5.6-sol", Functions: []FunctionDefinition{{Name: "search"}}},
			want:    "none",
		},
		{
			name:    "gpt-5.6 without tools keeps the server default",
			request: ChatRequest{Model: "gpt-5.6-terra"},
			want:    "",
		},
		{
			name:    "explicit effort is not overridden",
			request: ChatRequest{Model: "gpt-5.6-terra", Tools: []Tool{tool}, ReasoningEffort: "high"},
			want:    "high",
		},
		{
			name:    "older reasoning models must not receive none",
			request: ChatRequest{Model: "gpt-5.5", Tools: []Tool{tool}},
			want:    "",
		},
		{
			name:    "non-reasoning models are untouched",
			request: ChatRequest{Model: "gpt-4o", Tools: []Tool{tool}},
			want:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.request)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("Unmarshal: %v", err)
			}
			effort, present := got["reasoning_effort"]
			if tt.want == "" {
				if present {
					t.Errorf("reasoning_effort = %v, want absent; body: %s", effort, data)
				}
				return
			}
			if effort != tt.want {
				t.Errorf("reasoning_effort = %v, want %q; body: %s", effort, tt.want, data)
			}
		})
	}
}

// TestMarshalReasoningEffortDoesNotMutateRequest guards the value-receiver copy:
// marshalling must not leave reasoning_effort set on the caller's request.
func TestMarshalReasoningEffortDoesNotMutateRequest(t *testing.T) {
	req := ChatRequest{
		Model: "gpt-5.6-terra",
		Tools: []Tool{{Type: ToolTypeFunction, Function: FunctionDefinition{Name: "search"}}},
	}
	if _, err := json.Marshal(req); err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if req.ReasoningEffort != "" {
		t.Errorf("ReasoningEffort = %q, want it left empty on the caller's request", req.ReasoningEffort)
	}
}
