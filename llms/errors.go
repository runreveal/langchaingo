package llms

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// ErrorCode represents a standardized error code for LLM operations.
type ErrorCode string

const (
	// ErrCodeUnknown indicates an unknown error.
	ErrCodeUnknown ErrorCode = "unknown"

	// ErrCodeAuthentication indicates an authentication failure.
	ErrCodeAuthentication ErrorCode = "authentication"

	// ErrCodeRateLimit indicates a rate limit has been exceeded.
	ErrCodeRateLimit ErrorCode = "rate_limit"

	// ErrCodeInvalidRequest indicates the request was invalid.
	ErrCodeInvalidRequest ErrorCode = "invalid_request"

	// ErrCodeResourceNotFound indicates a requested resource was not found.
	ErrCodeResourceNotFound ErrorCode = "resource_not_found"

	// ErrCodeTimeout indicates the operation timed out.
	ErrCodeTimeout ErrorCode = "timeout"

	// ErrCodeCanceled indicates the operation was canceled.
	ErrCodeCanceled ErrorCode = "canceled"

	// ErrCodeQuotaExceeded indicates a quota has been exceeded.
	ErrCodeQuotaExceeded ErrorCode = "quota_exceeded"

	// ErrCodeContentFilter indicates content was blocked by safety filters.
	ErrCodeContentFilter ErrorCode = "content_filter"

	// ErrCodeTokenLimit indicates the token limit was exceeded.
	ErrCodeTokenLimit ErrorCode = "token_limit"

	// ErrCodeProviderUnavailable indicates the provider service is unavailable.
	ErrCodeProviderUnavailable ErrorCode = "provider_unavailable"

	// ErrCodeNotImplemented indicates a feature is not implemented.
	ErrCodeNotImplemented ErrorCode = "not_implemented"
)

// Error represents a standardized error from an LLM provider.
type Error struct {
	// Code is the standardized error code.
	Code ErrorCode

	// Message is a human-readable error message.
	Message string

	// Provider is the name of the provider that generated the error.
	Provider string

	// Details contains provider-specific error details.
	Details map[string]interface{}

	// Cause is the underlying error, if any.
	Cause error
}

// Error implements the error interface.
func (e *Error) Error() string {
	if e.Provider != "" {
		return fmt.Sprintf("%s: %s: %s", e.Provider, e.Code, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// Unwrap returns the underlying error.
func (e *Error) Unwrap() error {
	return e.Cause
}

// Is implements errors.Is support.
func (e *Error) Is(target error) bool {
	if target == nil {
		return false
	}

	// Check if target is an *Error with the same code
	if te, ok := target.(*Error); ok {
		return e.Code == te.Code
	}

	// Check against common errors
	switch e.Code {
	case ErrCodeCanceled:
		return errors.Is(target, context.Canceled)
	case ErrCodeTimeout:
		return errors.Is(target, context.DeadlineExceeded)
	}

	return false
}

// NewError creates a new standardized error.
func NewError(code ErrorCode, provider, message string) *Error {
	return &Error{
		Code:     code,
		Provider: provider,
		Message:  message,
		Details:  make(map[string]interface{}),
	}
}

// WithCause adds an underlying error cause.
func (e *Error) WithCause(cause error) *Error {
	e.Cause = cause
	return e
}

// WithDetail adds a detail to the error.
func (e *Error) WithDetail(key string, value interface{}) *Error {
	if e.Details == nil {
		e.Details = make(map[string]interface{})
	}
	e.Details[key] = value
	return e
}

// StatusCodeOverloaded is the non-standard status code Anthropic returns when its
// API is temporarily overloaded. It has no net/http constant.
const StatusCodeOverloaded = 529

// APIError is a transport-level error from a provider's HTTP API. It preserves the
// HTTP status code and, when the provider sends one, its own machine-readable error
// type, so callers can classify a failure without pattern-matching on message text.
//
// Providers return it from their HTTP clients; ErrorMapper recognizes it and uses
// the structured fields in preference to any string matching.
type APIError struct {
	// Provider is the name of the provider that returned the error.
	Provider string

	// StatusCode is the HTTP status code of the response.
	StatusCode int

	// Type is the provider's machine-readable error type, when the response body
	// carries one (e.g. Anthropic "overloaded_error", OpenAI "insufficient_quota").
	Type string

	// Message is the human-readable message from the response body.
	Message string
}

// Error implements the error interface. The "API returned unexpected status code"
// prefix is preserved from the untyped errors this replaces so existing callers
// matching on it keep working.
func (e *APIError) Error() string {
	msg := fmt.Sprintf("API returned unexpected status code: %d", e.StatusCode)
	if e.Type != "" {
		msg += ": " + e.Type
	}
	if e.Message != "" {
		msg += ": " + e.Message
	}
	return msg
}

// ErrorCode maps the API error to a standardized error code, preferring the
// provider's own error type over the HTTP status when both are present.
func (e *APIError) ErrorCode() ErrorCode {
	if code, ok := errorTypeToCode(e.Type); ok {
		return code
	}
	// A 4xx can mean several distinct things that share a status. Consult the
	// message before falling back to the status alone: an oversized prompt and a
	// malformed request are both 400, but only one is worth surfacing as a token
	// limit (and only one is worth retrying against a larger-context model).
	if e.StatusCode >= 400 && e.StatusCode < 500 {
		if code, ok := semanticCodeFromMessage(e.Message); ok {
			return code
		}
	}
	return StatusCodeToErrorCode(e.StatusCode)
}

// errorTypeToCode maps a provider's machine-readable error type to a standard
// code. Anthropic and OpenAI share most of these spellings.
func errorTypeToCode(errType string) (ErrorCode, bool) {
	switch strings.ToLower(errType) {
	case "authentication_error", "invalid_api_key", "permission_error":
		return ErrCodeAuthentication, true
	case "rate_limit_error", "rate_limit_exceeded":
		return ErrCodeRateLimit, true
	case "overloaded_error", "api_error", "server_error":
		return ErrCodeProviderUnavailable, true
	case "not_found_error", "model_not_found":
		return ErrCodeResourceNotFound, true
	case "billing_error", "insufficient_quota":
		return ErrCodeQuotaExceeded, true
	case "request_too_large", "context_length_exceeded":
		return ErrCodeTokenLimit, true
	case "content_filter", "content_policy_violation":
		return ErrCodeContentFilter, true
	case "timeout_error":
		return ErrCodeTimeout, true
	case "invalid_request_error":
		// Deliberately not authoritative: providers use this as a catch-all for
		// oversized prompts and filtered content too, so let the message decide.
		return ErrCodeUnknown, false
	}
	return ErrCodeUnknown, false
}

// semanticCodeFromMessage recognizes the few failures whose meaning lives in the
// message rather than the status code or error type.
func semanticCodeFromMessage(msg string) (ErrorCode, bool) {
	s := strings.ToLower(msg)
	for _, p := range []string{
		"prompt is too long", "token limit", "maximum context", "context length",
		"context window", "too many tokens", "exceeds the context",
	} {
		if strings.Contains(s, p) {
			return ErrCodeTokenLimit, true
		}
	}
	for _, p := range []string{"content filter", "safety", "blocked", "content policy"} {
		if strings.Contains(s, p) {
			return ErrCodeContentFilter, true
		}
	}
	return ErrCodeUnknown, false
}

// StatusCodeToErrorCode maps an HTTP status code to a standardized error code.
func StatusCodeToErrorCode(status int) ErrorCode {
	switch status {
	case http.StatusUnauthorized, http.StatusForbidden:
		return ErrCodeAuthentication
	case http.StatusNotFound:
		return ErrCodeResourceNotFound
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		return ErrCodeTimeout
	case http.StatusTooManyRequests:
		return ErrCodeRateLimit
	case http.StatusPaymentRequired:
		return ErrCodeQuotaExceeded
	case http.StatusRequestEntityTooLarge:
		return ErrCodeTokenLimit
	case StatusCodeOverloaded:
		return ErrCodeProviderUnavailable
	}
	switch {
	case status >= 500:
		return ErrCodeProviderUnavailable
	case status >= 400:
		return ErrCodeInvalidRequest
	}
	return ErrCodeUnknown
}

// IsAuthenticationError returns true if the error is an authentication error.
func IsAuthenticationError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == ErrCodeAuthentication
}

// IsRateLimitError returns true if the error is a rate limit error.
func IsRateLimitError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == ErrCodeRateLimit
}

// IsInvalidRequestError returns true if the error is an invalid request error.
func IsInvalidRequestError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == ErrCodeInvalidRequest
}

// IsTimeoutError returns true if the error is a timeout error.
func IsTimeoutError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == ErrCodeTimeout
}

// IsCanceledError returns true if the error is a cancellation error.
func IsCanceledError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == ErrCodeCanceled
}

// IsQuotaExceededError returns true if the error is a quota exceeded error.
func IsQuotaExceededError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == ErrCodeQuotaExceeded
}

// IsContentFilterError returns true if the error is a content filter error.
func IsContentFilterError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == ErrCodeContentFilter
}

// IsTokenLimitError returns true if the error is a token limit error.
func IsTokenLimitError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == ErrCodeTokenLimit
}

// IsProviderUnavailableError returns true if the error is a provider unavailable error.
func IsProviderUnavailableError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == ErrCodeProviderUnavailable
}

// IsNotImplementedError returns true if the error is a not implemented error.
func IsNotImplementedError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == ErrCodeNotImplemented
}

// Common error variables for easy comparison.
var (
	// ErrAuthentication is returned when authentication fails.
	ErrAuthentication = &Error{Code: ErrCodeAuthentication}

	// ErrRateLimit is returned when rate limit is exceeded.
	ErrRateLimit = &Error{Code: ErrCodeRateLimit}

	// ErrInvalidRequest is returned for invalid requests.
	ErrInvalidRequest = &Error{Code: ErrCodeInvalidRequest}

	// ErrTimeout is returned when an operation times out.
	ErrTimeout = &Error{Code: ErrCodeTimeout}

	// ErrCanceled is returned when an operation is canceled.
	ErrCanceled = &Error{Code: ErrCodeCanceled}

	// ErrQuotaExceeded is returned when quota is exceeded.
	ErrQuotaExceeded = &Error{Code: ErrCodeQuotaExceeded}

	// ErrContentFilter is returned when content is filtered.
	ErrContentFilter = &Error{Code: ErrCodeContentFilter}

	// ErrTokenLimit is returned when token limit is exceeded.
	ErrTokenLimit = &Error{Code: ErrCodeTokenLimit}

	// ErrProviderUnavailable is returned when provider is unavailable.
	ErrProviderUnavailable = &Error{Code: ErrCodeProviderUnavailable}

	// ErrNotImplemented is returned when a feature is not implemented.
	ErrNotImplemented = &Error{Code: ErrCodeNotImplemented}
)
