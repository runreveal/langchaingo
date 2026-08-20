package bedrock

import (
	"errors"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// TestMapError covers the Bedrock classification table. Bedrock surfaces AWS SDK
// errors whose text carries the service's exception names (ThrottlingException,
// ModelNotReadyException, ...), which are stable machine-readable identifiers.
// This is deterministic — no AWS credentials or network needed.
func TestMapError(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want llms.ErrorCode
	}{
		{"throttling", "operation error Bedrock: InvokeModel, ThrottlingException: Too many requests", llms.ErrCodeRateLimit},
		{"access denied", "AccessDeniedException: not authorized to perform bedrock:InvokeModel", llms.ErrCodeAuthentication},
		{"expired token", "ExpiredTokenException: the security token included in the request is expired", llms.ErrCodeAuthentication},
		{"model not found", "ResourceNotFoundException: could not find model", llms.ErrCodeResourceNotFound},
		{"model not ready", "ModelNotReadyException: model is not ready", llms.ErrCodeProviderUnavailable},
		{"service unavailable", "ServiceUnavailableException: try again later", llms.ErrCodeProviderUnavailable},
		{"model timeout", "ModelTimeoutException: the request timed out", llms.ErrCodeTimeout},
		{"validation", "ValidationException: malformed input request", llms.ErrCodeInvalidRequest},
		{"token limit", "ValidationException: too many input tokens", llms.ErrCodeTokenLimit},
		{"unknown passes through", "some unexpected failure", llms.ErrCodeUnknown},
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
		})
	}
}

func TestMapErrorNil(t *testing.T) {
	if MapError(nil) != nil {
		t.Error("MapError(nil) should be nil")
	}
}
