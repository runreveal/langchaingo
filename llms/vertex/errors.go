package vertex

import (
	"errors"
	"strings"

	"github.com/tmc/langchaingo/llms"
)

// ErrMissingProject is returned when the GCP project ID is not configured.
var ErrMissingProject = errors.New("vertex: missing GCP project (use WithProject)")

// ErrMissingLocation is returned when the GCP location/region is not configured.
var ErrMissingLocation = errors.New("vertex: missing GCP location (use WithLocation)")

// ErrUnsupportedPublisher is returned when the configured model ID does not map
// to a known Vertex publisher model family.
var ErrUnsupportedPublisher = errors.New("vertex: unsupported publisher for model")

type errorMapping struct {
	patterns []string
	code     llms.ErrorCode
	message  string
}

var vertexErrorMappings = []errorMapping{
	{
		patterns: []string{"permission_denied", "permission denied", "unauthenticated", "invalid_grant", "access denied"},
		code:     llms.ErrCodeAuthentication,
		message:  "Invalid or missing GCP credentials",
	},
	{
		patterns: []string{"resource_exhausted", "quota exceeded", "rate limit", "429"},
		code:     llms.ErrCodeRateLimit,
		message:  "Request rate limit or quota exceeded",
	},
	{
		patterns: []string{"not_found", "model not found", "publisher model"},
		code:     llms.ErrCodeResourceNotFound,
		message:  "Model not found or not accessible",
	},
	{
		patterns: []string{"invalid_argument", "malformed", "400"},
		code:     llms.ErrCodeInvalidRequest,
		message:  "Invalid request parameters",
	},
	{
		patterns: []string{"deadline_exceeded", "timeout"},
		code:     llms.ErrCodeTimeout,
		message:  "Request timeout",
	},
	{
		patterns: []string{"unavailable", "internal error", "500", "503"},
		code:     llms.ErrCodeProviderUnavailable,
		message:  "Vertex AI service error",
	},
}

// MapError maps Vertex AI errors to standardized langchaingo error codes.
func MapError(err error) error {
	if err == nil {
		return nil
	}

	errStr := strings.ToLower(err.Error())

	for _, mapping := range vertexErrorMappings {
		for _, pattern := range mapping.patterns {
			if strings.Contains(errStr, pattern) {
				return llms.NewError(mapping.code, "vertex", mapping.message).WithCause(err)
			}
		}
	}

	mapper := llms.NewErrorMapper("vertex")
	return mapper.Map(err)
}
