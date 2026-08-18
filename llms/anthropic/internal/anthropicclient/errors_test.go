package anthropicclient

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// stubDoer returns a canned HTTP response, so the error path can be exercised
// without a live API key or a recorded cassette.
type stubDoer struct {
	status int
	body   string
}

func (s stubDoer) Do(*http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: s.status,
		Body:       io.NopCloser(strings.NewReader(s.body)),
		Header:     make(http.Header),
	}, nil
}

func TestCreateMessageMapsAPIErrors(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		body       string
		wantCode   llms.ErrorCode
		wantType   string
		wantDetail string // substring of the provider message that must survive
	}{
		{
			name:       "rate limited",
			status:     http.StatusTooManyRequests,
			body:       `{"type":"error","error":{"type":"rate_limit_error","message":"Number of requests has exceeded your rate limit"}}`,
			wantCode:   llms.ErrCodeRateLimit,
			wantType:   "rate_limit_error",
			wantDetail: "exceeded your rate limit",
		},
		{
			name:       "overloaded returns 529, which has no net/http constant",
			status:     llms.StatusCodeOverloaded,
			body:       `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
			wantCode:   llms.ErrCodeProviderUnavailable,
			wantType:   "overloaded_error",
			wantDetail: "Overloaded",
		},
		{
			name:       "oversized prompt is a token limit despite being a 400",
			status:     http.StatusBadRequest,
			body:       `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}`,
			wantCode:   llms.ErrCodeTokenLimit,
			wantType:   "invalid_request_error",
			wantDetail: "250000 tokens",
		},
		{
			name:       "malformed request stays a bad request",
			status:     http.StatusBadRequest,
			body:       `{"type":"error","error":{"type":"invalid_request_error","message":"messages: unexpected role"}}`,
			wantCode:   llms.ErrCodeInvalidRequest,
			wantType:   "invalid_request_error",
			wantDetail: "unexpected role",
		},
		{
			name:       "bad key",
			status:     http.StatusUnauthorized,
			body:       `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`,
			wantCode:   llms.ErrCodeAuthentication,
			wantType:   "authentication_error",
			wantDetail: "invalid x-api-key",
		},
		{
			name:     "unparseable body still yields the status",
			status:   http.StatusInternalServerError,
			body:     `<html>502 Bad Gateway</html>`,
			wantCode: llms.ErrCodeProviderUnavailable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := New("test-token", "claude-sonnet-4-6", DefaultBaseURL,
				WithHTTPClient(stubDoer{status: tt.status, body: tt.body}))
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			_, err = c.CreateMessage(context.Background(), &MessageRequest{
				Model:    "claude-sonnet-4-6",
				Messages: []ChatMessage{{Role: "user", Content: "hi"}},
			})
			if err == nil {
				t.Fatal("expected an error")
			}

			var apiErr *llms.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("expected *llms.APIError, got %T: %v", err, err)
			}
			if apiErr.StatusCode != tt.status {
				t.Errorf("StatusCode = %d, want %d", apiErr.StatusCode, tt.status)
			}
			if apiErr.Type != tt.wantType {
				t.Errorf("Type = %q, want %q", apiErr.Type, tt.wantType)
			}
			if got := apiErr.ErrorCode(); got != tt.wantCode {
				t.Errorf("ErrorCode() = %v, want %v", got, tt.wantCode)
			}
			if tt.wantDetail != "" && !strings.Contains(err.Error(), tt.wantDetail) {
				t.Errorf("Error() = %q, lost provider detail %q", err.Error(), tt.wantDetail)
			}
		})
	}
}
