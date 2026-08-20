package bedrock

import (
	"github.com/tmc/langchaingo/llms"
)

// bedrockErrorMappings classifies AWS Bedrock errors.
//
// Unlike the HTTP providers, Bedrock surfaces AWS SDK errors whose text carries the
// service's own exception names ("ThrottlingException", "ModelNotReadyException").
// Those names are machine-readable identifiers rather than prose, so matching them
// is reliable in a way that matching status digits is not.
var bedrockErrorMappings = []llms.PatternMapping{
	{
		Patterns: []string{"accessdeniedexception", "unauthorized", "invalid security token", "expiredtoken"},
		Code:     llms.ErrCodeAuthentication,
		Summary:  "Invalid or missing AWS credentials",
	},
	{
		Patterns: []string{"throttlingexception", "toomanyrequestsexception"},
		Code:     llms.ErrCodeRateLimit,
		Summary:  "Request rate limit exceeded",
	},
	{
		Patterns: []string{"resourcenotfoundexception", "model not found"},
		Code:     llms.ErrCodeResourceNotFound,
		Summary:  "Model not found or not accessible",
	},
	{
		Patterns: []string{"serviceunavailableexception", "serviceexception", "internalservererror"},
		Code:     llms.ErrCodeProviderUnavailable,
		Summary:  "AWS Bedrock service error",
	},
	{
		Patterns: []string{"modelnotreadyexception"},
		Code:     llms.ErrCodeProviderUnavailable,
		Summary:  "Model not ready for invocation",
	},
	{
		Patterns: []string{"modeltimeoutexception"},
		Code:     llms.ErrCodeTimeout,
		Summary:  "Model invocation timeout",
	},
	{
		Patterns: []string{"servicequotaexceededexception"},
		Code:     llms.ErrCodeQuotaExceeded,
		Summary:  "Service quota exceeded",
	},
	{
		Patterns: []string{"payload size", "token limit", "too many input tokens"},
		Code:     llms.ErrCodeTokenLimit,
		Summary:  "Input size or token limit exceeded",
	},
	{
		Patterns: []string{"validationexception", "malformed"},
		Code:     llms.ErrCodeInvalidRequest,
		Summary:  "Invalid request parameters",
	},
}

// MapError maps AWS Bedrock-specific errors to standardized error codes.
func MapError(err error) error {
	return llms.MapProviderError("bedrock", err, bedrockErrorMappings)
}
