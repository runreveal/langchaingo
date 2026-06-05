package vertex

import (
	"net/http"

	"github.com/tmc/langchaingo/callbacks"
	"golang.org/x/oauth2"
	"google.golang.org/api/option"
)

// Option is a configuration option for the Vertex LLM.
type Option func(*options)

type options struct {
	project         string
	location        string
	modelID         string
	publisher       string
	callbackHandler callbacks.Handler

	// clientOpts are passed through to google.golang.org/api/transport/http
	// when building the authenticated HTTP client. They handle credentials,
	// custom HTTP transports, and testing hooks like a custom endpoint.
	clientOpts []option.ClientOption

	// httpClient, if set, overrides the HTTP client entirely and bypasses
	// GCP auth bootstrap. Useful for tests that want full control.
	httpClient *http.Client
}

// WithProject sets the GCP project ID. Required.
func WithProject(project string) Option {
	return func(o *options) {
		o.project = project
	}
}

// WithLocation sets the GCP region (e.g. "us-east5", "europe-west1"), or
// "global" for the global endpoint. Required.
func WithLocation(location string) Option {
	return func(o *options) {
		o.location = location
	}
}

// WithModel sets the Vertex publisher model ID (e.g. "claude-3-7-sonnet@20250219").
func WithModel(modelID string) Option {
	return func(o *options) {
		o.modelID = modelID
	}
}

// WithPublisher overrides publisher detection. Normally inferred from the
// model ID (e.g. "claude-*" → "anthropic").
func WithPublisher(publisher string) Option {
	return func(o *options) {
		o.publisher = publisher
	}
}

// WithCredentialsFile authenticates using a GCP service-account JSON file.
func WithCredentialsFile(path string) Option {
	return func(o *options) {
		if path == "" {
			return
		}
		o.clientOpts = append(o.clientOpts, option.WithCredentialsFile(path))
	}
}

// WithCredentialsJSON authenticates using in-memory GCP service-account JSON.
func WithCredentialsJSON(data []byte) Option {
	return func(o *options) {
		if len(data) == 0 {
			return
		}
		o.clientOpts = append(o.clientOpts, option.WithCredentialsJSON(data))
	}
}

// WithTokenSource authenticates using a caller-supplied oauth2.TokenSource.
func WithTokenSource(ts oauth2.TokenSource) Option {
	return func(o *options) {
		if ts == nil {
			return
		}
		o.clientOpts = append(o.clientOpts, option.WithTokenSource(ts))
	}
}

// WithClientOption appends a raw google.golang.org/api/option.ClientOption
// for cases not covered by the higher-level helpers above.
func WithClientOption(opt option.ClientOption) Option {
	return func(o *options) {
		if opt == nil {
			return
		}
		o.clientOpts = append(o.clientOpts, opt)
	}
}

// WithHTTPClient injects a pre-built HTTP client, bypassing GCP auth bootstrap.
// Useful for tests pointing at an httptest.Server.
func WithHTTPClient(client *http.Client) Option {
	return func(o *options) {
		o.httpClient = client
	}
}

// WithCallback sets a callbacks.Handler for generation lifecycle events.
func WithCallback(handler callbacks.Handler) Option {
	return func(o *options) {
		o.callbackHandler = handler
	}
}
