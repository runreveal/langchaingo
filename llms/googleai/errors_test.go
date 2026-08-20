package googleai

import (
	"errors"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// TestMapError covers the Google AI classification table directly. The googleai
// client returns gRPC/googleapi errors (not a langchaingo *llms.APIError), so
// classification is by message; this feeds representative real-world messages and
// asserts the code. It needs no client, cassette, or API key, which keeps it
// deterministic where the httprr error test cannot be (its cassette needs a live
// GOOGLE_API_KEY to re-record).
func TestMapError(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want llms.ErrorCode
	}{
		{"permission denied", "rpc error: code = PermissionDenied desc = PERMISSION_DENIED", llms.ErrCodeAuthentication},
		{"invalid key", "API key not valid. Please pass a valid API key.", llms.ErrCodeAuthentication},
		{"resource exhausted", "rpc error: code = ResourceExhausted desc = RESOURCE_EXHAUSTED: quota", llms.ErrCodeRateLimit},
		{"safety block", "finish reason: SAFETY", llms.ErrCodeContentFilter},
		{"recitation block", "blocked due to RECITATION", llms.ErrCodeContentFilter},
		{"model not found", "models/gemini-bogus is not found for API version v1beta", llms.ErrCodeResourceNotFound},
		{"unavailable", "the model is overloaded, please try again later", llms.ErrCodeProviderUnavailable},
		{"unknown passes through", "something unexpected happened", llms.ErrCodeUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mapped := MapError(errors.New(tt.in))
			var stdErr *llms.Error
			if !errors.As(mapped, &stdErr) {
				t.Fatalf("expected *llms.Error, got %T", mapped)
			}
			if stdErr.Code != tt.want {
				t.Errorf("MapError(%q) code = %v, want %v", tt.in, stdErr.Code, tt.want)
			}
			// The original message must survive classification (it feeds logs).
			if !errors.Is(mapped, stdErr.Unwrap()) {
				t.Error("mapped error should retain its cause")
			}
		})
	}
}

// TestMapErrorNil keeps the nil contract explicit.
func TestMapErrorNil(t *testing.T) {
	if MapError(nil) != nil {
		t.Error("MapError(nil) should be nil")
	}
}
