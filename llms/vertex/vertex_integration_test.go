package vertex_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"regexp"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/internal/httprr"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/vertex"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// These tests use the httprr record/replay pattern (same as llms/bedrock).
//
// To record: choose a project, authenticate with ADC, then run:
//
//	gcloud auth application-default login
//	VERTEX_PROJECT_ID=<your-project> \
//	    go test -v -run TestVertexAnthropic \
//	      ./llms/vertex/... -httprecord=TestVertexAnthropic
//
// You can also override the model and location used during recording:
//
//	VERTEX_PROJECT_ID=<p> VERTEX_MODEL=claude-opus-4-7@20251101 \
//	    VERTEX_LOCATION=us-east5 go test ...
//
// The resulting testdata/*.httprr files can be committed; in replay mode
// (no -httprecord flag) they let CI verify wire-format correctness without
// any GCP credentials.
//
// Heads-up: the recorded URL contains your GCP project ID. The scrubber
// below rewrites it to "redacted-project" so the recording is safe to
// share, and the replay path rewrites outgoing request URLs to match.

const (
	recordedProject = "redacted-project"
	defaultLocation = "us-east5"
	defaultModel    = "claude-3-5-haiku@20241022"
)

func testLocation() string {
	if v := os.Getenv("VERTEX_LOCATION"); v != "" {
		return v
	}
	return defaultLocation
}

func testModel() string {
	if v := os.Getenv("VERTEX_MODEL"); v != "" {
		return v
	}
	return defaultModel
}

// projectIDFromEnv returns the real project ID in recording mode, or the
// placeholder string used in committed recordings for replay mode.
func projectIDFromEnv(rr *httprr.RecordReplay) string {
	if rr.Recording() {
		if p := os.Getenv("VERTEX_PROJECT_ID"); p != "" {
			return p
		}
	}
	return recordedProject
}

// buildHTTPClient returns an http.Client that feeds requests through rr.
// In recording mode, requests are first authenticated via ADC; in replay
// mode, rr is used directly (no auth needed because recordings are scrubbed).
func buildHTTPClient(t *testing.T, ctx context.Context, rr *httprr.RecordReplay) *http.Client {
	t.Helper()

	// In recording mode, add an auth layer and a request scrubber that
	// rewrites the real project ID to the placeholder before RR sees it.
	if rr.Recording() {
		realProject := os.Getenv("VERTEX_PROJECT_ID")
		if realProject == "" {
			t.Fatal("VERTEX_PROJECT_ID must be set when recording")
		}

		ts, err := google.DefaultTokenSource(ctx, "https://www.googleapis.com/auth/cloud-platform")
		if err != nil {
			t.Fatalf("google.DefaultTokenSource: %v", err)
		}

		// Scrub Authorization header and project ID from both request and response.
		rr.ScrubReq(func(req *http.Request) error {
			req.Header.Del("Authorization")
			req.URL.Path = replaceProject(req.URL.Path, realProject)
			return nil
		})
		projectRE := regexp.MustCompile(regexp.QuoteMeta(realProject))
		rr.ScrubResp(func(buf *bytes.Buffer) error {
			replaced := projectRE.ReplaceAll(buf.Bytes(), []byte(recordedProject))
			buf.Reset()
			buf.Write(replaced)
			return nil
		})

		return &http.Client{Transport: &oauth2.Transport{Base: rr, Source: ts}}
	}

	// Replay mode: no auth; recordings were scrubbed of Authorization.
	return &http.Client{Transport: rr}
}

func replaceProject(path, real string) string {
	return regexp.MustCompile(`/projects/`+regexp.QuoteMeta(real)+`/`).
		ReplaceAllString(path, "/projects/"+recordedProject+"/")
}

func TestVertexAnthropicGenerate(t *testing.T) {
	ctx := context.Background()
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "VERTEX_PROJECT_ID")

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	if !rr.Recording() {
		t.Parallel()
	}

	httpClient := buildHTTPClient(t, ctx, rr)
	llm, err := vertex.New(
		vertex.WithProject(projectIDFromEnv(rr)),
		vertex.WithLocation(testLocation()),
		vertex.WithModel(testModel()),
		vertex.WithHTTPClient(httpClient),
	)
	require.NoError(t, err)

	resp, err := llm.GenerateContent(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Reply with exactly the word PING."),
		},
		llms.WithMaxTokens(16),
		llms.WithTemperature(0),
	)
	require.NoError(t, err)
	require.NotEmpty(t, resp.Choices)
	require.NotEmpty(t, resp.Choices[0].Content, "expected non-empty text response")
	t.Logf("model said: %q", resp.Choices[0].Content)
}

func TestVertexAnthropicToolCalling(t *testing.T) {
	ctx := context.Background()
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "VERTEX_PROJECT_ID")

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	if !rr.Recording() {
		t.Parallel()
	}

	httpClient := buildHTTPClient(t, ctx, rr)
	llm, err := vertex.New(
		vertex.WithProject(projectIDFromEnv(rr)),
		vertex.WithLocation(testLocation()),
		vertex.WithModel(testModel()),
		vertex.WithHTTPClient(httpClient),
	)
	require.NoError(t, err)

	weatherTool := llms.Tool{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "get_weather",
			Description: "Get the current weather for a location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The city and state, e.g. San Francisco, CA",
					},
				},
				"required": []string{"location"},
			},
		},
	}

	resp, err := llm.GenerateContent(ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "What's the weather in New York?"),
		},
		llms.WithTools([]llms.Tool{weatherTool}),
		llms.WithMaxTokens(256),
		llms.WithTemperature(0),
	)
	require.NoError(t, err)
	require.NotEmpty(t, resp.Choices)

	choice := resp.Choices[0]
	require.NotEmpty(t, choice.ToolCalls, "expected a tool_use response")

	tc := choice.ToolCalls[0]
	require.Equal(t, "function", tc.Type)
	require.Equal(t, "get_weather", tc.FunctionCall.Name)

	var args map[string]any
	require.NoError(t, json.Unmarshal([]byte(tc.FunctionCall.Arguments), &args))
	require.Contains(t, args, "location")
}
