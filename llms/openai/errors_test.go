package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// stubDoer returns a canned HTTP response so the error path can be exercised
// without a network call or API key.
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

type openaiErrCase struct {
	name     string
	status   int
	body     string
	wantCode llms.ErrorCode
	wantText string // provider detail that must survive classification
}

func openaiErrCases() []openaiErrCase {
	return []openaiErrCase{
		{
			name:     "rate limited",
			status:   http.StatusTooManyRequests,
			body:     `{"error":{"type":"rate_limit_exceeded","message":"Rate limit reached for gpt-4o"}}`,
			wantCode: llms.ErrCodeRateLimit,
			wantText: "Rate limit reached",
		},
		{
			name:     "insufficient quota",
			status:   http.StatusTooManyRequests,
			body:     `{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}`,
			wantCode: llms.ErrCodeQuotaExceeded,
			wantText: "exceeded your current quota",
		},
		{
			name:     "bad key",
			status:   http.StatusUnauthorized,
			body:     `{"error":{"type":"invalid_request_error","code":"invalid_api_key","message":"Incorrect API key provided"}}`,
			wantCode: llms.ErrCodeAuthentication,
			wantText: "Incorrect API key",
		},
		{
			name:     "context length exceeded is a token limit, not a bad request",
			status:   http.StatusBadRequest,
			body:     `{"error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"maximum context length is 128000 tokens"}}`,
			wantCode: llms.ErrCodeTokenLimit,
			wantText: "maximum context length",
		},
		{
			name:     "server overloaded",
			status:   http.StatusServiceUnavailable,
			body:     `{"error":{"type":"server_error","message":"The engine is currently overloaded"}}`,
			wantCode: llms.ErrCodeProviderUnavailable,
			wantText: "overloaded",
		},
	}
}

// TestGenerateContentClassifiesAPIErrors verifies both halves of the change on the
// OpenAI path: chat.go builds a typed *llms.APIError from a non-200 response, and
// GenerateContent routes it through MapError so callers get a classified
// *llms.Error. Uses a stubbed transport, so no key or network is needed.
func TestGenerateContentClassifiesAPIErrors(t *testing.T) {
	for _, tt := range openaiErrCases() {
		t.Run(tt.name, func(t *testing.T) {
			llm, err := New(
				WithToken("test-token"),
				WithModel("gpt-4o"),
				WithHTTPClient(stubDoer{status: tt.status, body: tt.body}),
			)
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			_, err = llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
			if err == nil {
				t.Fatal("expected an error")
			}

			var stdErr *llms.Error
			if !errors.As(err, &stdErr) {
				t.Fatalf("expected a classified *llms.Error, got %T: %v", err, err)
			}
			if stdErr.Code != tt.wantCode {
				t.Errorf("Code = %v, want %v (err: %v)", stdErr.Code, tt.wantCode, err)
			}
			if !strings.Contains(err.Error(), tt.wantText) {
				t.Errorf("Error() = %q, lost provider detail %q", err.Error(), tt.wantText)
			}
		})
	}
}
