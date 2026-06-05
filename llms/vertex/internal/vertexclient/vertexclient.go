// Package vertexclient is the internal HTTP client for Vertex AI publisher
// models. It is structured to mirror llms/bedrock/internal/bedrockclient:
// a neutral Message representation, a publisher dispatcher, and one
// provider_<family>.go file per model family.
package vertexclient

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/tmc/langchaingo/llms"
)

// Client is a Vertex AI client for publisher-model REST APIs.
type Client struct {
	httpClient *http.Client
	project    string
	location   string
}

// Message is the neutral representation of a chat message passed through the
// dispatcher. Each provider converts []Message into its own wire format.
// Copied from llms/bedrock/internal/bedrockclient.Message to keep the two
// cloud-Claude implementations structurally symmetric.
type Message struct {
	Role    llms.ChatMessageType
	Content string
	// Type may be "text", "image", "tool_call", or "tool_result"
	Type string
	// MimeType is the MIME type of binary content.
	MimeType string
	// Tool call fields (assistant → tool request)
	ToolCallID string `json:"tool_call_id,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`
	ToolArgs   string `json:"tool_args,omitempty"`
	// Tool result fields (tool → assistant response)
	ToolUseID string `json:"tool_use_id,omitempty"`
}

// NewClient creates a new Vertex client bound to a project + location using
// the given HTTP client. The caller is responsible for ensuring the HTTP
// client handles GCP authentication (typically via oauth2.Transport).
func NewClient(httpClient *http.Client, project, location string) *Client {
	return &Client{
		httpClient: httpClient,
		project:    project,
		location:   location,
	}
}

// GetPublisher returns the Vertex publisher name for a given model ID.
// Mirrors the shape of bedrockclient.getProvider.
func GetPublisher(modelID string) string {
	switch {
	case strings.Contains(modelID, "claude"):
		return "anthropic"
	}
	return ""
}

// CreateCompletion dispatches to the correct per-publisher implementation.
func (c *Client) CreateCompletion(ctx context.Context,
	publisher string,
	modelID string,
	messages []Message,
	options llms.CallOptions,
) (*llms.ContentResponse, error) {
	if publisher == "" {
		publisher = GetPublisher(modelID)
	}
	switch publisher {
	case "anthropic":
		return createAnthropicCompletion(ctx, c, modelID, messages, options)
	default:
		return nil, errors.New("unsupported publisher")
	}
}

// endpointHost returns the regional hostname for this client's location.
// The "global" location uses the unprefixed aiplatform.googleapis.com endpoint.
func (c *Client) endpointHost() string {
	if c.location == "global" {
		return "aiplatform.googleapis.com"
	}
	return c.location + "-aiplatform.googleapis.com"
}

func getMaxTokens(maxTokens, defaultValue int) int {
	if maxTokens <= 0 {
		return defaultValue
	}
	return maxTokens
}
