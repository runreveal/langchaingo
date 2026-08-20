package vertex

import (
	"errors"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// TestMapError covers Vertex classification. Transport failures arrive as
// *llms.APIError (classified by status + Anthropic error type); GCP auth/config
// problems raised before the request arrive as plain messages. Deterministic — no
// GCP credentials or network needed.
func TestMapError(t *testing.T) {
	tests := []struct {
		name string
		in   error
		want llms.ErrorCode
	}{
		{
			name: "structured APIError wins on status/type",
			in:   &llms.APIError{Provider: "vertex", StatusCode: 429, Type: "rate_limit_error"},
			want: llms.ErrCodeRateLimit,
		},
		{
			name: "structured overloaded",
			in:   &llms.APIError{Provider: "vertex", StatusCode: llms.StatusCodeOverloaded, Type: "overloaded_error"},
			want: llms.ErrCodeProviderUnavailable,
		},
		{
			name: "missing GCP credentials",
			in:   errors.New("could not find default credentials"),
			want: llms.ErrCodeAuthentication,
		},
		{
			name: "resource exhausted",
			in:   errors.New("vertex anthropic: status 429: RESOURCE_EXHAUSTED: quota exceeded"),
			want: llms.ErrCodeRateLimit,
		},
		{
			name: "model not allowlisted",
			in:   errors.New("publisher model is not allowlisted for this project"),
			want: llms.ErrCodeResourceNotFound,
		},
		{
			name: "unknown passes through",
			in:   errors.New("some unexpected failure"),
			want: llms.ErrCodeUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mapped := MapError(tt.in)
			var stdErr *llms.Error
			if !errors.As(mapped, &stdErr) {
				t.Fatalf("expected *llms.Error, got %T", mapped)
			}
			if stdErr.Code != tt.want {
				t.Errorf("MapError(%v) code = %v, want %v", tt.in, stdErr.Code, tt.want)
			}
		})
	}
}

func TestMapErrorNil(t *testing.T) {
	if MapError(nil) != nil {
		t.Error("MapError(nil) should be nil")
	}
}
