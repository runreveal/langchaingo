package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// stubChatReply is one scripted /chat/completions reply. Reasoning is returned
// under ReasoningField, the way the provider being imitated names it.
type stubChatReply struct {
	ReasoningField string
	Reasoning      string
	Content        string
	ToolCalls      []llms.ToolCall
}

func (r stubChatReply) finishReason() string {
	if len(r.ToolCalls) > 0 {
		return "tool_calls"
	}
	return "stop"
}

// json renders the reply as a non-streaming chat completion.
func (r stubChatReply) json(t *testing.T) string {
	t.Helper()
	msg := map[string]any{"role": "assistant", "content": r.Content}
	if r.Reasoning != "" {
		msg[r.ReasoningField] = r.Reasoning
	}
	if len(r.ToolCalls) > 0 {
		calls := make([]any, len(r.ToolCalls))
		for i, tc := range r.ToolCalls {
			calls[i] = map[string]any{
				"id": tc.ID, "type": "function",
				"function": map[string]any{"name": tc.FunctionCall.Name, "arguments": tc.FunctionCall.Arguments},
			}
		}
		msg["tool_calls"] = calls
	}
	body, err := json.Marshal(map[string]any{
		"id":      "chatcmpl-1",
		"object":  "chat.completion",
		"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": r.finishReason()}},
		"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
	})
	if err != nil {
		t.Fatalf("marshal reply: %v", err)
	}
	return string(body)
}

// sse renders the reply as a chat completion stream. Reasoning arrives before
// content and tool calls, split over two chunks, and each tool call's arguments
// follow its name in a separate chunk, as OpenAI-compatible servers send them.
func (r stubChatReply) sse(t *testing.T) string {
	t.Helper()
	var sb strings.Builder
	chunk := func(delta map[string]any, finish any) {
		body, err := json.Marshal(map[string]any{
			"id":      "chatcmpl-1",
			"object":  "chat.completion.chunk",
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
		})
		if err != nil {
			t.Fatalf("marshal chunk: %v", err)
		}
		fmt.Fprintf(&sb, "data: %s\n\n", body)
	}

	chunk(map[string]any{"role": "assistant", "content": ""}, nil)
	if r.Reasoning != "" {
		half := len(r.Reasoning) / 2
		chunk(map[string]any{r.ReasoningField: r.Reasoning[:half]}, nil)
		chunk(map[string]any{r.ReasoningField: r.Reasoning[half:]}, nil)
	}
	if r.Content != "" {
		chunk(map[string]any{"content": r.Content}, nil)
	}
	for i, tc := range r.ToolCalls {
		chunk(map[string]any{"tool_calls": []any{map[string]any{
			"index": i, "id": tc.ID, "type": "function",
			"function": map[string]any{"name": tc.FunctionCall.Name, "arguments": ""},
		}}}, nil)
		chunk(map[string]any{"tool_calls": []any{map[string]any{
			"index": i, "function": map[string]any{"arguments": tc.FunctionCall.Arguments},
		}}}, nil)
	}
	chunk(map[string]any{}, r.finishReason())
	sb.WriteString("data: [DONE]\n\n")
	return sb.String()
}

// newChatServer starts a stub /chat/completions endpoint that answers with the
// scripted replies in order and records each request body.
func newChatServer(t *testing.T, replies ...stubChatReply) (*httptest.Server, *[]map[string]any) {
	t.Helper()
	var requests []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/chat/completions"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		body, _ := io.ReadAll(r.Body)
		req := map[string]any{}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		if len(requests) >= len(replies) {
			t.Errorf("unexpected request %d, only %d replies scripted", len(requests)+1, len(replies))
			http.Error(w, "no reply scripted", http.StatusInternalServerError)
			return
		}
		reply := replies[len(requests)]
		requests = append(requests, req)

		if req["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, reply.sse(t))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, reply.json(t))
	}))
	t.Cleanup(srv.Close)
	return srv, &requests
}

func newChatLLM(t *testing.T, baseURL string) *LLM {
	t.Helper()
	llm, err := New(WithToken("sk-test"), WithModel("glm-5-3"), WithBaseURL(baseURL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return llm
}

func stubToolCall(id, args string) llms.ToolCall {
	return llms.ToolCall{ID: id, Type: "function", FunctionCall: &llms.FunctionCall{Name: "Echo", Arguments: args}}
}

func toolResult(id, content string) llms.MessageContent {
	return llms.MessageContent{
		Role:  llms.ChatMessageTypeTool,
		Parts: []llms.ContentPart{llms.ToolCallResponse{ToolCallID: id, Name: "Echo", Content: content}},
	}
}

var echoTools = llms.WithTools([]llms.Tool{{
	Type:     "function",
	Function: &llms.FunctionDefinition{Name: "Echo", Parameters: map[string]any{"type": "object"}},
}})

// requestMessages returns the messages of a recorded request as JSON objects.
func requestMessages(t *testing.T, req map[string]any) []map[string]any {
	t.Helper()
	raw, _ := req["messages"].([]any)
	msgs := make([]map[string]any, len(raw))
	for i, m := range raw {
		msg, ok := m.(map[string]any)
		if !ok {
			t.Fatalf("messages[%d] is %T, want an object", i, m)
		}
		msgs[i] = msg
	}
	return msgs
}

// reasoningKeys are the names OpenAI-compatible servers give the reasoning of an
// assistant message.
var reasoningKeys = []string{"reasoning_content", "reasoning"}

// assertReasoning checks that a request message carries the reasoning under
// field and under no other name. An empty want means no reasoning at all.
func assertReasoning(t *testing.T, msg map[string]any, field, want string) {
	t.Helper()
	for _, key := range reasoningKeys {
		got, present := msg[key]
		switch {
		case key == field && want != "":
			if got != want {
				t.Errorf("%s = %q, want %q", key, got, want)
			}
		case present:
			t.Errorf("message carries %s = %q, want it absent: %v", key, got, msg)
		}
	}
}

func assertToolCallIDs(t *testing.T, msg map[string]any, want ...string) {
	t.Helper()
	calls, _ := msg["tool_calls"].([]any)
	var got []string
	for _, c := range calls {
		call, _ := c.(map[string]any)
		id, _ := call["id"].(string)
		got = append(got, id)
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("tool call ids = %v, want %v", got, want)
	}
}

// A provider's reasoning must be readable whichever name it uses for the field,
// and must go back in that same field, on the assistant message that made the
// tool calls, when the caller replays the message.
func TestChatCompletionsReasoningRoundTrip(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name      string
		field     string
		streaming bool
	}{
		{name: "reasoning_content", field: ReasoningFieldReasoningContent},
		{name: "reasoning_content streaming", field: ReasoningFieldReasoningContent, streaming: true},
		{name: "reasoning", field: ReasoningFieldReasoning},
		{name: "reasoning streaming", field: ReasoningFieldReasoning, streaming: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			const reasoning = "The table is empty, so widen the time range."
			callA, callB := stubToolCall("call_a", `{"q":"a"}`), stubToolCall("call_b", `{"q":"b"}`)
			srv, requests := newChatServer(t,
				stubChatReply{ReasoningField: tc.field, Reasoning: reasoning, ToolCalls: []llms.ToolCall{callA, callB}},
				stubChatReply{Content: "done"},
			)
			llm := newChatLLM(t, srv.URL)

			opts := []llms.CallOption{echoTools}
			if tc.streaming {
				opts = append(opts, llms.WithStreamingFunc(func(context.Context, []byte) error { return nil }))
			}
			history := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "count the rows")}

			resp, err := llm.GenerateContent(context.Background(), history, opts...)
			if err != nil {
				t.Fatalf("GenerateContent: %v", err)
			}
			choice := resp.Choices[0]
			if choice.ReasoningContent != reasoning {
				t.Errorf("ReasoningContent = %q, want %q", choice.ReasoningContent, reasoning)
			}
			if got := choice.GenerationInfo[GenInfoReasoningField]; got != tc.field {
				t.Errorf("GenerationInfo[%s] = %v, want %q", GenInfoReasoningField, got, tc.field)
			}
			if len(choice.ToolCalls) != 2 {
				t.Fatalf("got %d tool calls, want 2", len(choice.ToolCalls))
			}

			history = append(history,
				llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
					llms.ThinkingContent{Thinking: choice.ReasoningContent}, choice.ToolCalls[0],
				}},
				toolResult("call_a", "0 rows"),
				llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{choice.ToolCalls[1]}},
				toolResult("call_b", "0 rows"),
			)
			opts = append(opts, WithReasoningField(tc.field))
			if _, err := llm.GenerateContent(context.Background(), history, opts...); err != nil {
				t.Fatalf("GenerateContent replay: %v", err)
			}

			if len(*requests) != 2 {
				t.Fatalf("got %d requests, want 2", len(*requests))
			}
			msgs := requestMessages(t, (*requests)[1])
			if len(msgs) != 5 {
				t.Fatalf("replay sent %d messages, want 5: %v", len(msgs), msgs)
			}
			assertReasoning(t, msgs[1], tc.field, reasoning)
			assertToolCallIDs(t, msgs[1], "call_a")
			if msgs[2]["role"] != "tool" || msgs[2]["tool_call_id"] != "call_a" {
				t.Errorf("messages[2] = %v, want the result of call_a", msgs[2])
			}
			assertReasoning(t, msgs[3], tc.field, "")
			assertToolCallIDs(t, msgs[3], "call_b")
			if msgs[4]["role"] != "tool" || msgs[4]["tool_call_id"] != "call_b" {
				t.Errorf("messages[4] = %v, want the result of call_b", msgs[4])
			}
		})
	}
}

// Without WithReasoningField, replayed reasoning uses the reasoning_content
// name, which is what this client has always read.
func TestChatCompletionsReasoningFieldDefault(t *testing.T) {
	t.Parallel()
	srv, requests := newChatServer(t, stubChatReply{Content: "done"})
	llm := newChatLLM(t, srv.URL)

	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "count the rows"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.ThinkingContent{Thinking: "check the table"}, stubToolCall("call_a", "{}"),
		}},
		toolResult("call_a", "0 rows"),
	}, echoTools)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	msgs := requestMessages(t, (*requests)[0])
	assertReasoning(t, msgs[1], ReasoningFieldReasoningContent, "check the table")
	assertToolCallIDs(t, msgs[1], "call_a")
}

// A provider that returns no reasoning reports no reasoning field, and a
// message replayed without a thinking part carries none.
func TestChatCompletionsWithoutReasoning(t *testing.T) {
	t.Parallel()
	call := stubToolCall("call_a", "{}")
	srv, requests := newChatServer(t,
		stubChatReply{ToolCalls: []llms.ToolCall{call}},
		stubChatReply{Content: "done"},
	)
	llm := newChatLLM(t, srv.URL)
	history := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "count the rows")}

	resp, err := llm.GenerateContent(context.Background(), history, echoTools)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	choice := resp.Choices[0]
	if choice.ReasoningContent != "" {
		t.Errorf("ReasoningContent = %q, want none", choice.ReasoningContent)
	}
	if field, ok := choice.GenerationInfo[GenInfoReasoningField]; ok {
		t.Errorf("GenerationInfo[%s] = %v, want it absent", GenInfoReasoningField, field)
	}

	history = append(history,
		llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{choice.ToolCalls[0]}},
		toolResult("call_a", "0 rows"),
	)
	if _, err := llm.GenerateContent(context.Background(), history, echoTools); err != nil {
		t.Fatalf("GenerateContent replay: %v", err)
	}
	msgs := requestMessages(t, (*requests)[1])
	assertReasoning(t, msgs[1], "", "")
	assertToolCallIDs(t, msgs[1], "call_a")
}
