package anthropic

import (
	"github.com/tmc/langchaingo/llms"
)

// anthropicErrorMappings classifies errors that reach MapError without structure.
//
// Errors from the Anthropic HTTP client are *llms.APIError and are classified from
// their status code and error type, so these patterns only cover what arrives as
// plain text: transport failures, and errors surfaced by streaming events.
//
// Note the absence of bare status digits ("401", "429", ...). Matching those against
// message text misreads dated model ids and request ids; status codes travel on
// *llms.APIError instead.
var anthropicErrorMappings = []llms.PatternMapping{
	{
		Patterns: []string{"invalid api key", "invalid x-api-key", "authentication failed"},
		Code:     llms.ErrCodeAuthentication,
		Summary:  "Invalid or missing API key",
	},
	{
		Patterns: []string{"rate limit", "too many requests"},
		Code:     llms.ErrCodeRateLimit,
		Summary:  "Rate limit exceeded",
	},
	{
		Patterns: []string{"model not found", "invalid model"},
		Code:     llms.ErrCodeResourceNotFound,
		Summary:  "Model not found",
	},
	{
		Patterns: []string{"prompt is too long", "maximum tokens", "context window", "request_too_large"},
		Code:     llms.ErrCodeTokenLimit,
		Summary:  "Token limit exceeded",
	},
	{
		Patterns: []string{"content blocked", "safety violation"},
		Code:     llms.ErrCodeContentFilter,
		Summary:  "Content blocked by safety filter",
	},
	{
		Patterns: []string{"credit limit", "credit balance", "quota exceeded"},
		Code:     llms.ErrCodeQuotaExceeded,
		Summary:  "API quota exceeded",
	},
	{
		Patterns: []string{"overloaded", "service unavailable"},
		Code:     llms.ErrCodeProviderUnavailable,
		Summary:  "Anthropic service temporarily unavailable",
	},
	{
		Patterns: []string{"invalid request"},
		Code:     llms.ErrCodeInvalidRequest,
		Summary:  "Invalid request",
	},
}

// MapError maps Anthropic-specific errors to standardized error codes.
func MapError(err error) error {
	return llms.MapProviderError("anthropic", err, anthropicErrorMappings)
}
