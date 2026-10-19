package modelcatalog

import (
	"slices"
	"testing"
)

func TestLookup(t *testing.T) {
	cases := []struct {
		model string
		want  string
	}{
		{"claude-opus-4-8", "claude-opus-4-8"},
		{"global.anthropic.claude-opus-4-8", "claude-opus-4-8"},
		{"us.anthropic.claude-opus-4-6-v1", "claude-opus-4-6"},
		{"claude-opus-4-8@20260101", "claude-opus-4-8"},
		{"arn:aws:bedrock:us-east-2:123456789012:inference-profile/us.anthropic.claude-sonnet-5", "claude-sonnet-5"},
		{"Global.Anthropic.Claude-Sonnet-5", "claude-sonnet-5"},
		// The longest prefix wins over a broader family.
		{"claude-opus-5-5", "claude-opus-5-5"},
		{"global.anthropic.claude-opus-5-5", "claude-opus-5-5"},
		{"claude-opus-5", "claude-opus-5"},
		{"claude-sonnet-5-5", "claude-sonnet-5-5"},
		{"claude-sonnet-5", "claude-sonnet-5"},
		{"claude-haiku-5-5", "claude-haiku-5-5"},
		{"claude-fable-5-1", "claude-fable-5"},
		{"claude-mythos-5-1", "claude-mythos-5"},
		{"global.anthropic.claude-opus-4-5-20251101-v1:0", "claude-opus-4-5"},
		{"claude-opus-4-5@20251101", "claude-opus-4-5"},
		{"anthropic.claude-opus-4-1-20250805-v1:0", "claude-opus-4"},
		{"us.anthropic.claude-opus-4-20250514-v1:0", "claude-opus-4"},
		{"global.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"},
		{"anthropic.claude-sonnet-4-20250514-v1:0", "claude-sonnet-4"},
		{"claude-haiku-4-5-20251001", "claude-haiku-4-5"},
		{"anthropic.claude-3-haiku-20240307-v1:0", "claude-3-haiku"},
		{"gpt-4o-mini", "gpt-4o"},
		{"gpt-4-turbo", "gpt-4o"},
		{"gpt-4", "gpt-4"},
		{"gpt-5", "gpt-5"},
		{"gpt-5-mini", "gpt-5"},
		{"gpt-5.1", "gpt-5.1"},
		{"gpt-5.2", "gpt-5.2"},
		{"gpt-5.4", "gpt-5.4"},
		{"gpt-5.5", "gpt-5.4"},
		{"gpt-5.6", "gpt-5.6"},
		{"gpt-5.6-terra", "gpt-5.6"},
		{"gpt-6-sol", "gpt-6"},
		{"gpt-6-luna", "gpt-6"},
		{"gpt-6-astra", "gpt-6-astra"},
		{"gpt-6.1-sol", "gpt-6.1"},
		{"o3", "o-series"},
		{"o4-mini", "o-series"},
		{"o1-preview", "o-series"},
		{"gemini-2.5-flash-lite", "gemini-2.5-flash"},
		{"gemini-3-pro", "gemini"},
		{"us.amazon.nova-micro-v1:0", "nova-micro"},
	}
	for _, c := range cases {
		s, ok := Lookup(c.model)
		if !ok {
			t.Errorf("Lookup(%q): no match, want %q", c.model, c.want)
			continue
		}
		if s.ID != c.want {
			t.Errorf("Lookup(%q) = %q, want %q", c.model, s.ID, c.want)
		}
	}
}

// TestLookupUnknown checks that ids the catalog does not describe stay unknown,
// so providers fall back to their own heuristics instead of inheriting a wrong
// family's limits.
func TestLookupUnknown(t *testing.T) {
	for _, m := range []string{
		"",
		"llama3.1",
		"my-azure-deployment",
		"anthropic.claude-sonnet-9-99",
		// A prefix must end at a boundary, so versions continue past it.
		"gpt-5.3",
		"gpt-5.7",
		"gpt-5.10",
		"gpt-4.1",
		"gpt-7",
		"o10-mini",
		"claude-opus-55",
	} {
		if s, ok := Lookup(m); ok {
			t.Errorf("Lookup(%q) = %q, want no match", m, s.ID)
		}
	}
}

func TestIsGeoInferenceProfile(t *testing.T) {
	cases := map[string]bool{
		"us.anthropic.claude-sonnet-5":                                       true,
		"eu.anthropic.claude-fable-5-1":                                      true,
		"jp.anthropic.claude-opus-5-5":                                       true,
		"apac.anthropic.claude-sonnet-5":                                     true,
		"global.anthropic.claude-sonnet-5":                                   false,
		"anthropic.claude-sonnet-5":                                          false,
		"claude-sonnet-5":                                                    false,
		"arn:aws:bedrock:x:y:inference-profile/au.anthropic.claude-sonnet-5": true,
	}
	for model, want := range cases {
		if got := IsGeoInferenceProfile(model); got != want {
			t.Errorf("IsGeoInferenceProfile(%q) = %v, want %v", model, got, want)
		}
	}
}

// TestPrice checks that the geo premium applies to every geo inference
// profile, not a hand-maintained list of prefixes, and never to families
// without one.
func TestPrice(t *testing.T) {
	cases := []struct {
		model           string
		wantIn, wantOut float64
	}{
		{"claude-opus-5-5", 4, 20},
		{"claude-opus-5", 5, 25},
		{"claude-sonnet-5-5", 2, 10},
		{"global.anthropic.claude-fable-5-1", 10, 50},
		{"us.anthropic.claude-fable-5-1", 11, 55},
		{"eu.anthropic.claude-fable-5-1", 11, 55},
		{"au.anthropic.claude-fable-5-1", 11, 55},
		{"jp.anthropic.claude-opus-5-5", 4.4, 22},
		{"apac.anthropic.claude-sonnet-5", 2.2, 11},
		{"claude-mythos-5-1", 10, 50},
		// No premium on families that don't carry one.
		{"us.anthropic.claude-sonnet-4-6", 3, 15},
		{"us.amazon.nova-pro-v1:0", 0.8, 3.2},
	}
	for _, c := range cases {
		in, out, ok := Price(c.model)
		if !ok {
			t.Errorf("Price(%q): unpriced", c.model)
			continue
		}
		if in != c.wantIn || out != c.wantOut {
			t.Errorf("Price(%q) = $%v/$%v, want $%v/$%v", c.model, in, out, c.wantIn, c.wantOut)
		}
	}
	if _, _, ok := Price("gpt-6-sol"); ok {
		t.Error("Price(gpt-6-sol): want unpriced, the catalog has no OpenAI rates")
	}
}

func TestPriceForPromptLongContext(t *testing.T) {
	cases := []struct {
		prompt          int
		wantIn, wantOut float64
	}{
		{0, 0.10, 0.50},
		{100_000, 0.10, 0.50},
		{100_001, 0.50, 2.50},
	}
	for _, c := range cases {
		in, out, ok := PriceForPrompt("claude-haiku-5-5", c.prompt)
		if !ok || in != c.wantIn || out != c.wantOut {
			t.Errorf("PriceForPrompt(haiku-5-5, %d) = $%v/$%v (%v), want $%v/$%v",
				c.prompt, in, out, ok, c.wantIn, c.wantOut)
		}
	}
}

// TestClaudeRequestSurface pins the facts providers build requests from: which
// Claude models reject budget_tokens and which reject sampling parameters.
// Getting either wrong turns every thinking or temperature request into a 400.
func TestClaudeRequestSurface(t *testing.T) {
	cases := []struct {
		model       string
		style       ReasoningStyle
		temperature bool
	}{
		{"claude-fable-5-1", ReasoningAdaptive, false},
		{"claude-mythos-5-1", ReasoningAdaptive, false},
		{"claude-opus-5-5", ReasoningAdaptive, false},
		{"claude-opus-5", ReasoningAdaptive, false},
		{"claude-sonnet-5-5", ReasoningAdaptive, false},
		{"claude-sonnet-5", ReasoningAdaptive, false},
		{"claude-haiku-5-5", ReasoningAdaptive, false},
		{"claude-opus-4-8", ReasoningAdaptive, false},
		{"claude-opus-4-7", ReasoningAdaptive, false},
		{"claude-opus-4-6", ReasoningAdaptive, true},
		{"claude-sonnet-4-6", ReasoningAdaptive, true},
		{"claude-opus-4-5", ReasoningBudgetTokens, true},
		{"claude-sonnet-4-5", ReasoningBudgetTokens, true},
		{"claude-haiku-4-5", ReasoningBudgetTokens, true},
		{"claude-3-5-sonnet-20240620", ReasoningNone, true},
	}
	for _, c := range cases {
		s, ok := Lookup(c.model)
		if !ok {
			t.Errorf("Lookup(%q): no match", c.model)
			continue
		}
		if s.Reasoning.Style != c.style {
			t.Errorf("%s reasoning = %s, want %s", c.model, s.Reasoning.Style, c.style)
		}
		if s.Temperature != c.temperature {
			t.Errorf("%s temperature = %v, want %v", c.model, s.Temperature, c.temperature)
		}
	}
}

func TestSupportsEffort(t *testing.T) {
	s, _ := Lookup("claude-opus-4-6")
	if !s.Reasoning.SupportsEffort("high") || s.Reasoning.SupportsEffort("xhigh") {
		t.Errorf("opus-4-6 efforts = %v, want high but not xhigh", s.Reasoning.Efforts)
	}
}

func TestParseRejectsInvalidCatalog(t *testing.T) {
	cases := map[string]string{
		"unknown field": `
models:
  - {id: a, match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none}, temprature: true}`,
		"missing context window": `
models:
  - {id: a, match: [a], vendor: x, api: messages, reasoning: {style: none}}`,
		"unknown api": `
models:
  - {id: a, match: [a], vendor: x, api: grpc, contextWindow: 1, reasoning: {style: none}}`,
		"effort without levels": `
models:
  - {id: a, match: [a], vendor: x, api: responses, contextWindow: 1, reasoning: {style: effort}}`,
		"adaptive without levels": `
models:
  - {id: a, match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: adaptive}}`,
		"levels on budget style": `
models:
  - {id: a, match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: budget_tokens, efforts: [low]}}`,
		"duplicate match": `
models:
  - {id: a, match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none}}
  - {id: b, match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none}}`,
		"duplicate id": `
models:
  - {id: a, match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none}}
  - {id: a, match: [b], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none}}`,
		"uppercase match": `
models:
  - {id: a, match: [Claude], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none}}`,
		"missing id": `
models:
  - {match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none}}`,
		"missing match": `
models:
  - {id: a, vendor: x, api: messages, contextWindow: 1, reasoning: {style: none}}`,
		"empty match": `
models:
  - {id: a, match: [""], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none}}`,
		"missing vendor": `
models:
  - {id: a, match: [a], api: messages, contextWindow: 1, reasoning: {style: none}}`,
		"negative max output": `
models:
  - {id: a, match: [a], vendor: x, api: messages, contextWindow: 1, maxOutputTokens: -1, reasoning: {style: none}}`,
		"unknown reasoning style": `
models:
  - {id: a, match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: telepathy}}`,
		"negative pricing": `
models:
  - {id: a, match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none},
     pricing: {input: -1, output: 1}}`,
		"negative geo premium": `
models:
  - {id: a, match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none},
     pricing: {input: 1, output: 1, geoPremium: -0.1}}`,
		"not yaml": `models: [`,
		"long context without threshold": `
models:
  - {id: a, match: [a], vendor: x, api: messages, contextWindow: 1, reasoning: {style: none},
     pricing: {input: 1, output: 1, longContext: {input: 2, output: 2}}}`,
	}
	for name, data := range cases {
		if _, err := parse([]byte(data)); err == nil {
			t.Errorf("%s: parse succeeded, want error", name)
		}
	}
}

// TestOpenAIEffortLevels pins each OpenAI family's effort levels to its model
// page. "none" matters most: the openai client sends it to turn reasoning off
// for tool calls on /chat/completions, and gpt-6-astra and gpt-6.1-sol reject it.
func TestOpenAIEffortLevels(t *testing.T) {
	cases := map[string][]string{
		"gpt-5":       {"minimal", "low", "medium", "high"},
		"gpt-5.1":     {"none", "low", "medium", "high"},
		"gpt-5.2":     {"none", "low", "medium", "high", "xhigh"},
		"gpt-5.5":     {"none", "low", "medium", "high", "xhigh"},
		"gpt-5.6-sol": {"none", "low", "medium", "high", "xhigh", "max"},
		"gpt-6-sol":   {"none", "low", "medium", "high", "xhigh", "max"},
		"gpt-6-luna":  {"none", "low", "medium", "high", "xhigh", "max"},
		"gpt-6-astra": {"low", "medium", "high", "xhigh", "max"},
		"gpt-6.1-sol": {"low", "medium", "high", "xhigh", "max"},
	}
	for model, want := range cases {
		s, ok := Lookup(model)
		if !ok {
			t.Errorf("Lookup(%q): no match", model)
			continue
		}
		if !slices.Equal(s.Reasoning.Efforts, want) {
			t.Errorf("%s efforts = %v, want %v", model, s.Reasoning.Efforts, want)
		}
	}
}

func TestParseAcceptsValidCatalog(t *testing.T) {
	got, err := parse([]byte(`
models:
  - {id: a, match: [a, a2], vendor: x, api: responses, contextWindow: 10, maxOutputTokens: 5,
     reasoning: {style: effort, efforts: [low]}, temperature: true,
     pricing: {input: 1, output: 2, geoPremium: 0.1, longContext: {threshold: 5, input: 3, output: 4}}}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got) != 1 || got[0].ID != "a" || got[0].Pricing.LongContext.Threshold != 5 {
		t.Errorf("parse = %+v", got)
	}
}

func TestMustParsePanicsOnInvalidCatalog(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("mustParse did not panic on an invalid catalog")
		}
	}()
	mustParse([]byte(`models: [{id: a}]`))
}

// TestEveryMatchReachesItsEntry guards the YAML against entries that can never
// be selected: each match prefix must resolve to the entry that lists it, not
// to a longer prefix elsewhere.
func TestEveryMatchReachesItsEntry(t *testing.T) {
	for _, s := range Specs() {
		for _, m := range s.Match {
			got, ok := Lookup(m)
			if !ok || got.ID != s.ID {
				t.Errorf("Lookup(%q) = %q (%v), want entry %q", m, got.ID, ok, s.ID)
			}
		}
	}
}

func TestSpecsReturnsACopy(t *testing.T) {
	all := Specs()
	if len(all) == 0 {
		t.Fatal("Specs() is empty")
	}
	all[0].ID = "changed"
	if Specs()[0].ID == "changed" {
		t.Error("Specs() exposed the catalog's backing slice")
	}
}
