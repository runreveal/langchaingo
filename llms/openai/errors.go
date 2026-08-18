package openai

import (
	"github.com/tmc/langchaingo/llms"
)

// openaiErrorMappings classifies errors that reach MapError without structure.
//
// Errors from the OpenAI HTTP client are *llms.APIError and are classified from
// their status code and error type, so these patterns only cover what arrives as
// plain text — notably the generic messages sanitizeHTTPError substitutes for
// transport failures.
//
// Note the absence of bare status digits ("401", "429", ...). Matching those
// against message text misreads dated model ids ("gpt-4o-2024-05-13") and request
// ids; status codes travel on *llms.APIError instead.
var openaiErrorMappings = []llms.PatternMapping{
	{
		Patterns: []string{"invalid_api_key", "incorrect api key", "invalid authentication"},
		Code:     llms.ErrCodeAuthentication,
		Summary:  "Invalid or missing API key",
	},
	{
		Patterns: []string{"rate limit", "too many requests"},
		Code:     llms.ErrCodeRateLimit,
		Summary:  "Rate limit exceeded",
	},
	{
		Patterns: []string{"model_not_found", "does not exist"},
		Code:     llms.ErrCodeResourceNotFound,
		Summary:  "Model not found",
	},
	{
		Patterns: []string{"context_length_exceeded", "maximum context length", "reduce the length"},
		Code:     llms.ErrCodeTokenLimit,
		Summary:  "Token limit exceeded",
	},
	{
		Patterns: []string{"content_filter", "content policy"},
		Code:     llms.ErrCodeContentFilter,
		Summary:  "Content blocked by content filter",
	},
	{
		Patterns: []string{"insufficient_quota", "exceeded your current quota", "billing"},
		Code:     llms.ErrCodeQuotaExceeded,
		Summary:  "API quota exceeded",
	},
	{
		Patterns: []string{"request timeout", "deadline"},
		Code:     llms.ErrCodeTimeout,
		Summary:  "Request timed out",
	},
	{
		Patterns: []string{"service unavailable", "server_error", "engine is currently overloaded"},
		Code:     llms.ErrCodeProviderUnavailable,
		Summary:  "OpenAI service temporarily unavailable",
	},
}

// MapError maps OpenAI-specific errors to standardized error codes.
func MapError(err error) error {
	return llms.MapProviderError("openai", err, openaiErrorMappings)
}
