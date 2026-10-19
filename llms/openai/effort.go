package openai

import (
	"strings"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/modelcatalog"
	"github.com/tmc/langchaingo/llms/openai/internal/openaiclient"
)

// reasoningEffort resolves the reasoning effort for a request. An explicit
// WithReasoningEffort wins and is sent as given, so an unsupported level gets
// the API's own error rather than being dropped. Otherwise the provider-neutral
// llms.WithThinkingMode is mapped onto the levels the model accepts. "" leaves
// the server default in place.
func reasoningEffort(opts *llms.CallOptions, model string) (effort string, explicit bool) {
	if e := reasoningEffortOption(opts); e != "" {
		return e, true
	}
	config, ok := opts.Metadata["thinking_config"].(*llms.ThinkingConfig)
	if !ok || config == nil {
		return "", false
	}
	return effortForThinkingMode(model, config.Mode), false
}

// effortForThinkingMode maps a thinking mode onto a model's effort levels.
// ThinkingModeNone turns reasoning as far down as the model allows: "none" on
// most models, "minimal" on gpt-5, and nothing on models with neither (gpt-6-astra
// and gpt-6.1-sol always reason). ThinkingModeAuto leaves the server default.
func effortForThinkingMode(model string, mode llms.ThinkingMode) string {
	var candidates []string
	switch mode {
	case llms.ThinkingModeLow, llms.ThinkingModeMedium, llms.ThinkingModeHigh:
		candidates = []string{string(mode)}
	case llms.ThinkingModeNone:
		candidates = []string{openaiclient.ReasoningEffortNone, "minimal"}
	default:
		return ""
	}

	spec, ok := modelcatalog.Lookup(model)
	if !ok {
		// Every reasoning model accepts low, medium and high. Which of them
		// can switch reasoning off is not knowable from the name, so an
		// unknown model keeps its default rather than risk a 400.
		if mode != llms.ThinkingModeNone && openaiclient.IsReasoningModel(strings.ToLower(model)) {
			return candidates[0]
		}
		return ""
	}
	for _, c := range candidates {
		if spec.Reasoning.SupportsEffort(c) {
			return c
		}
	}
	return ""
}
