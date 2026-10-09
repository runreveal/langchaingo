package ollama

import (
	"net"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/tmc/langchaingo/testing/llmtest"
)

func TestLLM(t *testing.T) {
	serverURL := os.Getenv("OLLAMA_HOST")
	if serverURL == "" {
		serverURL = "http://localhost:11434"
	}
	skipIfServerUnavailable(t, serverURL)

	llm, err := New(
		WithServerURL(serverURL),
		WithModel("gpt-oss:20b"),
	)
	if err != nil {
		t.Skipf("Ollama not available: %v", err)
	}

	llmtest.TestLLM(t, llm)
}

// skipIfServerUnavailable skips tests that need a live Ollama server when none is reachable.
func skipIfServerUnavailable(t *testing.T, serverURL string) {
	t.Helper()
	host := serverURL
	if u, err := url.Parse(serverURL); err == nil && u.Host != "" {
		host = u.Host
	}
	conn, err := net.DialTimeout("tcp", host, time.Second)
	if err != nil {
		t.Skipf("Ollama not available at %s: %v", serverURL, err)
	}
	conn.Close()
}
