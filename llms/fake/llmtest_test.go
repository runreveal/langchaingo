package fake

import (
	"testing"

	"github.com/tmc/langchaingo/testing/llmtest"
)

func TestLLM(t *testing.T) {
	// Fake LLM doesn't need API keys
	llm := NewFakeLLM([]string{"Hello from the fake LLM."})

	// Test basic functionality only
	llmtest.TestLLM(t, llm)
}
