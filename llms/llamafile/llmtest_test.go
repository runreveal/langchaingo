package llamafile

import (
	"net"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/langchaingo/testing/llmtest"
)

func TestLLM(t *testing.T) {
	if !isLlamafileAvailable() {
		t.Skip("llamafile is not available")
	}

	// Llamafile uses LLAMAFILE_HOST environment variable for server URL
	// If not set, it defaults to http://127.0.0.1:8080
	host := "127.0.0.1:8080"
	if v := os.Getenv("LLAMAFILE_HOST"); v != "" {
		_, host, _ = strings.Cut(v, "://")
		if host == "" {
			host = v
		}
	}
	conn, err := net.DialTimeout("tcp", host, time.Second)
	if err != nil {
		t.Skipf("Llamafile server not available at %s: %v", host, err)
	}
	conn.Close()

	llm, err := New()
	if err != nil {
		t.Skipf("Llamafile server not available: %v", err)
	}

	llmtest.TestLLM(t, llm)
}
