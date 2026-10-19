// Package modelcatalog describes the model families langchaingo can call:
// context window, output limit, wire API, reasoning style, sampling-parameter
// support and price. The data lives in catalog.yaml, so supporting a new model
// is an edit there rather than a new strings.Contains case in each provider.
//
// Lookup normalizes a model id before matching it, so one entry covers every
// platform a model is served on:
//
//	claude-opus-4-8                          (Anthropic API)
//	global.anthropic.claude-opus-4-8         (Bedrock inference profile)
//	us.anthropic.claude-opus-4-6-v1          (Bedrock geo inference profile)
//	arn:aws:bedrock:...:inference-profile/us.anthropic.claude-sonnet-5
//	claude-opus-4-8@20250101                 (Vertex)
//
// Each entry lists id prefixes. The longest matching prefix wins, so
// claude-opus-4-5 resolves to its own entry rather than to claude-opus-4, and a
// prefix only matches at a boundary, so gpt-5 does not claim gpt-5.7.
//
// An id that matches no entry is unknown, not unsupported: providers fall back
// to their own heuristics for it, which keeps Azure deployment names and newly
// released models working as before.
package modelcatalog

import (
	"bytes"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

// API is the wire API a model is served by.
type API string

const (
	APIMessages        API = "messages"
	APIChatCompletions API = "chat_completions"
	APIResponses       API = "responses"
	APIGemini          API = "gemini"
	APIConverse        API = "converse"
)

// ReasoningStyle is how a model is asked to reason.
type ReasoningStyle string

const (
	// ReasoningNone means the model has no controllable reasoning.
	ReasoningNone ReasoningStyle = "none"
	// ReasoningBudgetTokens takes an explicit thinking token budget.
	ReasoningBudgetTokens ReasoningStyle = "budget_tokens"
	// ReasoningAdaptive decides how much to think on its own, steered by an
	// effort level. It rejects an explicit token budget.
	ReasoningAdaptive ReasoningStyle = "adaptive"
	// ReasoningEffort takes a discrete effort level (low, medium, high).
	ReasoningEffort ReasoningStyle = "effort"
)

type Reasoning struct {
	Style ReasoningStyle `yaml:"style"`
	// Efforts lists the effort levels the model accepts, when it takes one.
	Efforts []string `yaml:"efforts"`
}

func (r Reasoning) SupportsEffort(level string) bool {
	return slices.Contains(r.Efforts, level)
}

// Pricing is USD per 1M tokens.
type Pricing struct {
	InputPerMillion  float64 `yaml:"input"`
	OutputPerMillion float64 `yaml:"output"`
	// GeoPremium is the fractional uplift charged on Bedrock geo inference
	// profiles (us., eu., jp., ...). Global profiles pay the base rate.
	GeoPremium float64 `yaml:"geoPremium"`
	// LongContext, when set, replaces the base rates for prompts larger than
	// its threshold.
	LongContext *LongContextPricing `yaml:"longContext"`
}

type LongContextPricing struct {
	Threshold        int     `yaml:"threshold"`
	InputPerMillion  float64 `yaml:"input"`
	OutputPerMillion float64 `yaml:"output"`
}

// Spec is one model family.
type Spec struct {
	ID            string   `yaml:"id"`
	Match         []string `yaml:"match"`
	Vendor        string   `yaml:"vendor"`
	API           API      `yaml:"api"`
	ContextWindow int      `yaml:"contextWindow"`
	// MaxOutputTokens is the model's own output limit; 0 when unknown.
	MaxOutputTokens int       `yaml:"maxOutputTokens"`
	Reasoning       Reasoning `yaml:"reasoning"`
	Temperature     bool      `yaml:"temperature"`
	Pricing         *Pricing  `yaml:"pricing"`
}

//go:embed catalog.yaml
var catalogYAML []byte

var specs = mustParse(catalogYAML)

type catalogFile struct {
	Models []Spec `yaml:"models"`
}

func mustParse(data []byte) []Spec {
	s, err := parse(data)
	if err != nil {
		panic(fmt.Sprintf("modelcatalog: invalid embedded catalog.yaml: %v", err))
	}
	return s
}

func parse(data []byte) ([]Spec, error) {
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	var f catalogFile
	if err := dec.Decode(&f); err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("parse catalog: %w", err)
	}
	seenID := map[string]bool{}
	seenMatch := map[string]string{}
	for _, s := range f.Models {
		if err := s.validate(); err != nil {
			return nil, fmt.Errorf("model %q: %w", s.ID, err)
		}
		if seenID[s.ID] {
			return nil, fmt.Errorf("model %q is listed twice", s.ID)
		}
		seenID[s.ID] = true
		for _, m := range s.Match {
			if prev, ok := seenMatch[m]; ok {
				return nil, fmt.Errorf("match %q is listed by both %q and %q", m, prev, s.ID)
			}
			seenMatch[m] = s.ID
		}
	}
	return f.Models, nil
}

func (s Spec) validate() error {
	switch {
	case s.ID == "":
		return errors.New("missing id")
	case len(s.Match) == 0:
		return errors.New("missing match")
	case s.Vendor == "":
		return errors.New("missing vendor")
	case s.ContextWindow <= 0:
		return errors.New("missing contextWindow")
	case s.MaxOutputTokens < 0:
		return errors.New("negative maxOutputTokens")
	}
	switch s.API {
	case APIMessages, APIChatCompletions, APIResponses, APIGemini, APIConverse:
	default:
		return fmt.Errorf("unknown api %q", s.API)
	}
	switch s.Reasoning.Style {
	case ReasoningEffort, ReasoningAdaptive:
		if len(s.Reasoning.Efforts) == 0 {
			return fmt.Errorf("%s reasoning needs efforts", s.Reasoning.Style)
		}
	case ReasoningNone, ReasoningBudgetTokens:
		if len(s.Reasoning.Efforts) > 0 {
			return fmt.Errorf("efforts do not apply to %s reasoning", s.Reasoning.Style)
		}
	default:
		return fmt.Errorf("unknown reasoning style %q", s.Reasoning.Style)
	}
	for _, m := range s.Match {
		if m == "" || m != strings.ToLower(m) {
			return fmt.Errorf("match %q must be non-empty and lowercase", m)
		}
	}
	if p := s.Pricing; p != nil {
		if p.InputPerMillion < 0 || p.OutputPerMillion < 0 || p.GeoPremium < 0 {
			return errors.New("negative pricing")
		}
		if lc := p.LongContext; lc != nil && lc.Threshold <= 0 {
			return errors.New("longContext needs a positive threshold")
		}
	}
	return nil
}

// Specs returns every model family in the catalog.
func Specs() []Spec {
	out := make([]Spec, len(specs))
	copy(out, specs)
	return out
}

// Lookup returns the spec for a model id on any platform, and false when no
// family matches.
func Lookup(model string) (Spec, bool) {
	id := Normalize(model)
	var (
		best    Spec
		bestLen int
	)
	for _, s := range specs {
		for _, m := range s.Match {
			if len(m) > bestLen && matchesPrefix(id, m) {
				best, bestLen = s, len(m)
			}
		}
	}
	return best, bestLen > 0
}

// matchesPrefix reports whether id starts with prefix and the prefix ends at a
// boundary. Digits, letters and "." continue a version, so gpt-5 matches
// gpt-5-mini but not gpt-5.7 or gpt-50, and gpt-4 does not match gpt-4o.
func matchesPrefix(id, prefix string) bool {
	if !strings.HasPrefix(id, prefix) {
		return false
	}
	if len(id) == len(prefix) {
		return true
	}
	switch id[len(prefix)] {
	case '-', '@', ':', '_':
		return true
	}
	return false
}

// vendorPrefixes are the Bedrock model-id vendor namespaces, which follow any
// inference-profile prefix ("us.", "global.", ...).
var vendorPrefixes = []string{"anthropic.", "amazon."}

// Normalize reduces a platform-specific model id to the bare model name the
// catalog matches on: lowercased, with any ARN or resource path, Bedrock
// inference-profile prefix and vendor namespace removed.
func Normalize(model string) string {
	id := lastPathSegment(model)
	if _, rest, ok := splitVendor(id); ok {
		return rest
	}
	return id
}

func lastPathSegment(model string) string {
	id := strings.ToLower(strings.TrimSpace(model))
	if i := strings.LastIndex(id, "/"); i >= 0 {
		id = id[i+1:]
	}
	return id
}

// splitVendor splits a Bedrock id into the inference-profile prefix before the
// vendor namespace ("us", "global", or "" when there is none) and the model
// name after it.
func splitVendor(id string) (profile, rest string, ok bool) {
	for _, v := range vendorPrefixes {
		if before, after, found := strings.Cut(id, v); found {
			return strings.TrimSuffix(before, "."), after, true
		}
	}
	return "", "", false
}

// IsGeoInferenceProfile reports whether a Bedrock model id names a geo
// inference profile (us., eu., jp., apac., ...), as opposed to a global.
// profile or a bare in-region model id.
func IsGeoInferenceProfile(model string) bool {
	profile, _, ok := splitVendor(lastPathSegment(model))
	return ok && profile != "" && profile != "global"
}

func Price(model string) (inputPerMillion, outputPerMillion float64, ok bool) {
	return PriceForPrompt(model, 0)
}

// PriceForPrompt is Price for a request whose prompt has promptTokens input
// tokens, so models with long-context pricing use the higher rate when the
// prompt crosses the threshold.
func PriceForPrompt(model string, promptTokens int) (inputPerMillion, outputPerMillion float64, ok bool) {
	s, found := Lookup(model)
	if !found || s.Pricing == nil {
		return 0, 0, false
	}
	p := s.Pricing
	in, out := p.InputPerMillion, p.OutputPerMillion
	if lc := p.LongContext; lc != nil && promptTokens > lc.Threshold {
		in, out = lc.InputPerMillion, lc.OutputPerMillion
	}
	if p.GeoPremium > 0 && IsGeoInferenceProfile(model) {
		in = roundMicro(in * (1 + p.GeoPremium))
		out = roundMicro(out * (1 + p.GeoPremium))
	}
	return in, out, true
}

// roundMicro rounds away the float error from applying a percentage premium, so
// $4 * 1.1 prices as $4.4 rather than $4.4000000000000004.
func roundMicro(v float64) float64 { return math.Round(v*1e6) / 1e6 }
