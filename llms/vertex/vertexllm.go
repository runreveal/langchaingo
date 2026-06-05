// Package vertex provides a langchaingo provider for Anthropic Claude models
// served through Google Vertex AI.
//
// It connects to the Vertex AI publisher-model REST endpoint at
// {location}-aiplatform.googleapis.com and sends requests in Anthropic's
// Messages API format (with Vertex's small body/URL deltas).
//
// Prereqs for use: the GCP project must have the Vertex AI API enabled and
// must have access to the Anthropic publisher models in Model Garden.
//
// Structure mirrors llms/bedrock: a top-level llms.Model implementation that
// wraps an internal client with per-publisher provider files.
package vertex

import (
	"context"
	"errors"

	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/vertex/internal/vertexclient"
	"google.golang.org/api/option"
	htransport "google.golang.org/api/transport/http"
)

// defaultCloudPlatformScope is the OAuth2 scope required for Vertex AI calls.
const defaultCloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

// LLM is a Vertex AI LLM implementation. Currently supports Anthropic Claude
// publisher models; dispatches by model ID.
type LLM struct {
	publisher        string
	modelID          string
	client           *vertexclient.Client
	CallbacksHandler callbacks.Handler
}

var _ llms.Model = (*LLM)(nil)

// New creates a new Vertex LLM using the caller's ambient context.
func New(opts ...Option) (*LLM, error) {
	return NewWithContext(context.Background(), opts...)
}

// NewWithContext creates a new Vertex LLM, using ctx for any auth bootstrap.
func NewWithContext(ctx context.Context, opts ...Option) (*LLM, error) {
	o := &options{}
	for _, opt := range opts {
		opt(o)
	}

	if o.project == "" {
		return nil, ErrMissingProject
	}
	if o.location == "" {
		return nil, ErrMissingLocation
	}

	httpClient := o.httpClient
	if httpClient == nil {
		clientOpts := append([]option.ClientOption{option.WithScopes(defaultCloudPlatformScope)}, o.clientOpts...)
		c, _, err := htransport.NewClient(ctx, clientOpts...)
		if err != nil {
			return nil, err
		}
		httpClient = c
	}

	publisher := o.publisher
	if publisher == "" && o.modelID != "" {
		publisher = vertexclient.GetPublisher(o.modelID)
	}
	if publisher == "" {
		return nil, ErrUnsupportedPublisher
	}

	return &LLM{
		publisher:        publisher,
		modelID:          o.modelID,
		client:           vertexclient.NewClient(httpClient, o.project, o.location),
		CallbacksHandler: o.callbackHandler,
	}, nil
}

// Call implements llms.Model.
func (l *LLM) Call(ctx context.Context, prompt string, opts ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, l, prompt, opts...)
}

// GenerateContent implements llms.Model.
func (l *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, opts ...llms.CallOption) (*llms.ContentResponse, error) {
	if l.CallbacksHandler != nil {
		l.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	callOpts := llms.CallOptions{Model: l.modelID}
	for _, opt := range opts {
		opt(&callOpts)
	}

	converted, err := processMessages(messages)
	if err != nil {
		return nil, err
	}

	res, err := l.client.CreateCompletion(ctx, l.publisher, callOpts.Model, converted, callOpts)
	if err != nil {
		if l.CallbacksHandler != nil {
			l.CallbacksHandler.HandleLLMError(ctx, err)
		}
		return nil, err
	}

	if l.CallbacksHandler != nil {
		l.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, res)
	}
	return res, nil
}

// processMessages converts langchaingo's MessageContent into the neutral
// Message shape the vertexclient package dispatches on. Mirrors
// llms/bedrock.processMessages.
func processMessages(messages []llms.MessageContent) ([]vertexclient.Message, error) {
	out := make([]vertexclient.Message, 0, len(messages))
	for _, m := range messages {
		for _, part := range m.Parts {
			switch p := part.(type) {
			case llms.TextContent:
				out = append(out, vertexclient.Message{
					Role:    m.Role,
					Content: p.Text,
					Type:    "text",
				})
			case llms.BinaryContent:
				out = append(out, vertexclient.Message{
					Role:     m.Role,
					Content:  string(p.Data),
					MimeType: p.MIMEType,
					Type:     "image",
				})
			case llms.ToolCall:
				out = append(out, vertexclient.Message{
					Role:       m.Role,
					Type:       "tool_call",
					ToolCallID: p.ID,
					ToolName:   p.FunctionCall.Name,
					ToolArgs:   p.FunctionCall.Arguments,
				})
			case llms.ToolCallResponse:
				out = append(out, vertexclient.Message{
					Role:      m.Role,
					Content:   p.Content,
					Type:      "tool_result",
					ToolUseID: p.ToolCallID,
				})
			default:
				return nil, errors.New("unsupported message type")
			}
		}
	}
	return out, nil
}

