package llms

import (
	"context"
	"errors"
	"net"
	"strings"
)

// ErrorMapper helps map provider-specific errors to standardized errors.
type ErrorMapper struct {
	provider string
	matchers []ErrorMatcher
}

// ErrorMatcher matches an error and returns the appropriate error code.
type ErrorMatcher struct {
	// Match returns true if this matcher handles the error
	Match func(error) bool
	// Code is the error code to use
	Code ErrorCode
	// CodeFor optionally derives the code from the error itself, for matchers whose
	// code is not fixed. When set it takes precedence over Code.
	CodeFor func(error) ErrorCode
	// Transform optionally transforms the error message
	Transform func(error) string
}

// NewErrorMapper creates a new error mapper for a provider.
func NewErrorMapper(provider string) *ErrorMapper {
	return &ErrorMapper{
		provider: provider,
		matchers: defaultMatchers(),
	}
}

// defaultMatchers returns the default set of error matchers.
func defaultMatchers() []ErrorMatcher {
	matchers := []ErrorMatcher{}

	// Structured errors first: an *APIError carries the real status code and the
	// provider's error type, so it never needs to be guessed at from message text.
	matchers = append(matchers, apiErrorMatcher())

	// Add context error matchers
	matchers = append(matchers, contextErrorMatchers()...)

	// Add string pattern matchers
	matchers = append(matchers, stringPatternMatchers()...)

	return matchers
}

// apiErrorMatcher matches a provider *APIError and classifies it from its
// structured fields.
func apiErrorMatcher() ErrorMatcher {
	return ErrorMatcher{
		Match: func(err error) bool {
			var apiErr *APIError
			return errors.As(err, &apiErr)
		},
		CodeFor: func(err error) ErrorCode {
			var apiErr *APIError
			if errors.As(err, &apiErr) {
				return apiErr.ErrorCode()
			}
			return ErrCodeUnknown
		},
	}
}

// contextErrorMatchers returns matchers for context-related errors.
func contextErrorMatchers() []ErrorMatcher {
	return []ErrorMatcher{
		{
			Match: func(err error) bool {
				return errors.Is(err, context.Canceled)
			},
			Code: ErrCodeCanceled,
		},
		{
			Match: func(err error) bool {
				return errors.Is(err, context.DeadlineExceeded)
			},
			Code: ErrCodeTimeout,
		},
		{
			Match: func(err error) bool {
				var netErr net.Error
				return errors.As(err, &netErr) && netErr.Timeout()
			},
			Code: ErrCodeTimeout,
		},
	}
}

// stringPatternMatchers returns matchers based on error string patterns.
func stringPatternMatchers() []ErrorMatcher {
	// Define pattern groups for easier maintenance.
	//
	// These deliberately contain no bare status-code digits. Matching "429" or
	// "400" anywhere in the message misclassifies ordinary text that happens to
	// contain those digits — a dated model id like "claude-3-5-sonnet-20240429"
	// reads as a rate limit, and a request id like "req_011CQ400ZxyZ" reads as a
	// bad request. Status codes are carried structurally by *APIError instead, and
	// matched ahead of these by apiErrorMatcher.
	authPatterns := []string{"unauthorized", "authentication", "api key"}
	rateLimitPatterns := []string{"rate limit", "too many requests"}
	invalidPatterns := []string{"invalid request", "bad request"}
	notFoundPatterns := []string{"not found"}
	quotaPatterns := []string{"quota", "limit exceeded", "insufficient"}
	contentPatterns := []string{"content filter", "safety", "blocked", "inappropriate"}
	tokenPatterns := []string{"token limit", "maximum context", "context length", "too long"}
	unavailablePatterns := []string{"service unavailable", "internal server", "overloaded"}
	notImplPatterns := []string{"not implemented", "not supported", "unsupported"}

	return []ErrorMatcher{
		makeStringMatcher(authPatterns, ErrCodeAuthentication),
		makeStringMatcher(rateLimitPatterns, ErrCodeRateLimit),
		makeStringMatcher(invalidPatterns, ErrCodeInvalidRequest),
		makeStringMatcher(notFoundPatterns, ErrCodeResourceNotFound),
		makeStringMatcher(quotaPatterns, ErrCodeQuotaExceeded),
		makeStringMatcher(contentPatterns, ErrCodeContentFilter),
		makeStringMatcher(tokenPatterns, ErrCodeTokenLimit),
		makeStringMatcher(unavailablePatterns, ErrCodeProviderUnavailable),
		makeStringMatcher(notImplPatterns, ErrCodeNotImplemented),
	}
}

// makeStringMatcher creates an ErrorMatcher that checks for any of the given patterns.
func makeStringMatcher(patterns []string, code ErrorCode) ErrorMatcher {
	return ErrorMatcher{
		Match: func(err error) bool {
			s := strings.ToLower(err.Error())
			for _, pattern := range patterns {
				if strings.Contains(s, pattern) {
					return true
				}
			}
			return false
		},
		Code: code,
	}
}

// AddMatcher adds a custom error matcher.
func (m *ErrorMapper) AddMatcher(matcher ErrorMatcher) *ErrorMapper {
	// Prepend custom matchers so they take precedence
	m.matchers = append([]ErrorMatcher{matcher}, m.matchers...)
	return m
}

// WrapError wraps an error with standardized error information.
func (m *ErrorMapper) WrapError(err error) error {
	if err == nil {
		return nil
	}

	// Check if already wrapped
	var stdErr *Error
	if errors.As(err, &stdErr) {
		return err
	}

	// Find matching error code
	code := ErrCodeUnknown
	message := err.Error()

	for _, matcher := range m.matchers {
		if matcher.Match(err) {
			code = matcher.Code
			if matcher.CodeFor != nil {
				code = matcher.CodeFor(err)
			}
			if matcher.Transform != nil {
				message = matcher.Transform(err)
			}
			break
		}
	}

	return NewError(code, m.provider, message).WithCause(err)
}

// Map is an alias for WrapError for consistency with provider error mappers.
func (m *ErrorMapper) Map(err error) error {
	return m.WrapError(err)
}

// PatternMapping associates message substrings with a standard error code. It is
// the fallback classification path, used only for errors that carry no structure.
type PatternMapping struct {
	// Patterns are lowercase substrings matched against the error message.
	Patterns []string

	// Code is the error code to assign on a match.
	Code ErrorCode

	// Summary is a short human-readable description of the failure class. It is
	// recorded under Details["summary"] rather than replacing the message, because
	// the provider's original text is what diagnostics and logs need.
	Summary string
}

// MapProviderError classifies err for the named provider. Structured information is
// always preferred: an error that is already an *Error is returned untouched, and an
// *APIError is classified from its status code and error type. Only untyped errors
// fall through to the provider's message patterns.
//
// The returned error preserves the original message; the matched Summary is attached
// as a detail. Providers call this from their MapError functions.
func MapProviderError(provider string, err error, mappings []PatternMapping) error {
	if err == nil {
		return nil
	}

	// Already classified somewhere below us; don't reclassify.
	var stdErr *Error
	if errors.As(err, &stdErr) {
		return err
	}

	// Structured transport error: status code and provider error type beat any
	// amount of message matching.
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return NewError(apiErr.ErrorCode(), provider, apiErr.Error()).WithCause(err)
	}

	msg := strings.ToLower(err.Error())
	for _, m := range mappings {
		for _, pattern := range m.Patterns {
			if strings.Contains(msg, pattern) {
				return NewError(m.Code, provider, err.Error()).
					WithCause(err).
					WithDetail("summary", m.Summary)
			}
		}
	}

	return NewErrorMapper(provider).Map(err)
}

// OpenAIErrorMapper creates an error mapper with OpenAI-specific patterns.
func OpenAIErrorMapper() *ErrorMapper {
	mapper := NewErrorMapper("openai")

	// Add OpenAI-specific matchers
	mapper.AddMatcher(ErrorMatcher{
		Match: func(err error) bool {
			s := err.Error()
			return strings.Contains(s, "invalid_api_key")
		},
		Code: ErrCodeAuthentication,
		Transform: func(_ error) string {
			return "Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable."
		},
	})

	mapper.AddMatcher(ErrorMatcher{
		Match: func(err error) bool {
			s := err.Error()
			return strings.Contains(s, "model_not_found")
		},
		Code: ErrCodeResourceNotFound,
		Transform: func(_ error) string {
			return "Model not found. Please check the model name and your API access."
		},
	})

	return mapper
}

// AnthropicErrorMapper creates an error mapper with Anthropic-specific patterns.
func AnthropicErrorMapper() *ErrorMapper {
	mapper := NewErrorMapper("anthropic")

	// Add Anthropic-specific matchers
	mapper.AddMatcher(ErrorMatcher{
		Match: func(err error) bool {
			s := err.Error()
			return strings.Contains(s, "invalid_x_api_key")
		},
		Code: ErrCodeAuthentication,
		Transform: func(_ error) string {
			return "Invalid Anthropic API key. Please check your ANTHROPIC_API_KEY environment variable."
		},
	})

	mapper.AddMatcher(ErrorMatcher{
		Match: func(err error) bool {
			s := err.Error()
			return strings.Contains(s, "credit_balance")
		},
		Code: ErrCodeQuotaExceeded,
		Transform: func(_ error) string {
			return "Anthropic API credit balance exceeded."
		},
	})

	return mapper
}

// GoogleAIErrorMapper creates an error mapper with Google AI-specific patterns.
func GoogleAIErrorMapper() *ErrorMapper {
	mapper := NewErrorMapper("googleai")

	// Add Google AI-specific matchers
	mapper.AddMatcher(ErrorMatcher{
		Match: func(err error) bool {
			s := err.Error()
			return strings.Contains(s, "API key not valid")
		},
		Code: ErrCodeAuthentication,
		Transform: func(_ error) string {
			return "Invalid Google AI API key. Please check your GOOGLE_API_KEY environment variable."
		},
	})

	mapper.AddMatcher(ErrorMatcher{
		Match: func(err error) bool {
			s := err.Error()
			return strings.Contains(s, "RECITATION") || strings.Contains(s, "SAFETY")
		},
		Code: ErrCodeContentFilter,
	})

	return mapper
}
