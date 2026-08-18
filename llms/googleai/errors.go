package googleai

import (
	"github.com/tmc/langchaingo/llms"
)

// googleAIErrorMappings classifies Google AI errors.
//
// The Google client returns gRPC/googleapi errors rather than a langchaingo
// *llms.APIError, so classification here is by message. The patterns favour
// Google's own uppercase reason codes (SAFETY, RECITATION, RESOURCE_EXHAUSTED),
// which are machine-readable identifiers rather than prose.
var googleAIErrorMappings = []llms.PatternMapping{
	{
		Patterns: []string{"api key not valid", "api_key_invalid", "unauthenticated", "permission_denied"},
		Code:     llms.ErrCodeAuthentication,
		Summary:  "Invalid or missing API key",
	},
	{
		Patterns: []string{"resource_exhausted", "rate limit", "too many requests"},
		Code:     llms.ErrCodeRateLimit,
		Summary:  "Rate limit exceeded",
	},
	{
		Patterns: []string{"is not found", "not_found", "is not supported for generatecontent"},
		Code:     llms.ErrCodeResourceNotFound,
		Summary:  "Model not found",
	},
	{
		Patterns: []string{"safety", "recitation", "prohibited_content", "blocked"},
		Code:     llms.ErrCodeContentFilter,
		Summary:  "Content blocked by safety filter",
	},
	{
		Patterns: []string{"token count", "exceeds the maximum number of tokens", "input token count"},
		Code:     llms.ErrCodeTokenLimit,
		Summary:  "Token limit exceeded",
	},
	{
		Patterns: []string{"quota", "billing"},
		Code:     llms.ErrCodeQuotaExceeded,
		Summary:  "API quota exceeded",
	},
	{
		Patterns: []string{"unavailable", "internal error", "overloaded"},
		Code:     llms.ErrCodeProviderUnavailable,
		Summary:  "Google AI service temporarily unavailable",
	},
	{
		Patterns: []string{"invalid_argument", "invalid request"},
		Code:     llms.ErrCodeInvalidRequest,
		Summary:  "Invalid request",
	},
}

// MapError maps Google AI-specific errors to standardized error codes.
func MapError(err error) error {
	return llms.MapProviderError("googleai", err, googleAIErrorMappings)
}
