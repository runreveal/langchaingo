package vertex

import (
	"errors"

	"github.com/tmc/langchaingo/llms"
)

// ErrMissingProject is returned when the GCP project ID is not configured.
var ErrMissingProject = errors.New("vertex: missing GCP project (use WithProject)")

// ErrMissingLocation is returned when the GCP location/region is not configured.
var ErrMissingLocation = errors.New("vertex: missing GCP location (use WithLocation)")

// ErrUnsupportedPublisher is returned when the configured model ID does not map
// to a known Vertex publisher model family.
var ErrUnsupportedPublisher = errors.New("vertex: unsupported publisher for model")

// vertexErrorMappings classifies errors that reach MapError without structure.
//
// Errors from the Vertex HTTP client are *llms.APIError and are classified from
// their status code and Anthropic error type, so these patterns only cover what
// arrives as plain text: transport failures and GCP auth problems raised before
// the request is sent.
//
// Note the absence of bare status digits — see llms/errors_mapper.go for why.
var vertexErrorMappings = []llms.PatternMapping{
	{
		Patterns: []string{
			"could not find default credentials", "unauthenticated",
			"permission_denied", "invalid authentication", "token expired",
		},
		Code:    llms.ErrCodeAuthentication,
		Summary: "Invalid or missing GCP credentials",
	},
	{
		Patterns: []string{"resource_exhausted", "rate limit", "too many requests", "quota exceeded"},
		Code:     llms.ErrCodeRateLimit,
		Summary:  "Rate limit exceeded",
	},
	{
		Patterns: []string{"model not found", "was not found", "is not allowlisted"},
		Code:     llms.ErrCodeResourceNotFound,
		Summary:  "Model not found or not enabled for this project",
	},
	{
		Patterns: []string{"prompt is too long", "maximum tokens", "context window", "request_too_large"},
		Code:     llms.ErrCodeTokenLimit,
		Summary:  "Token limit exceeded",
	},
	{
		Patterns: []string{"blocked", "safety", "content filter"},
		Code:     llms.ErrCodeContentFilter,
		Summary:  "Content blocked by safety filter",
	},
	{
		Patterns: []string{"billing", "insufficient"},
		Code:     llms.ErrCodeQuotaExceeded,
		Summary:  "GCP quota or billing limit reached",
	},
	{
		Patterns: []string{"overloaded", "service unavailable", "unavailable"},
		Code:     llms.ErrCodeProviderUnavailable,
		Summary:  "Vertex AI service temporarily unavailable",
	},
	{
		Patterns: []string{"invalid_argument", "invalid request"},
		Code:     llms.ErrCodeInvalidRequest,
		Summary:  "Invalid request",
	},
}

// MapError maps Vertex AI-specific errors to standardized error codes.
func MapError(err error) error {
	return llms.MapProviderError("vertex", err, vertexErrorMappings)
}
