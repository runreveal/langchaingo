package bedrock

import (
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/bedrock/internal/bedrockclient"
)

// CompactionConfig configures server-side context compaction for Claude 4.6+
// models on Bedrock. See WithCompaction.
type CompactionConfig = bedrockclient.CompactionConfig

// WithCompaction enables server-side context compaction (the compact_20260112
// strategy) for Anthropic Claude models. Bedrock summarizes older context once
// the prompt crosses the trigger; with PauseAfterCompaction it returns
// stop_reason "compaction" and the summary in GenerationInfo["CompactionSummary"],
// matching the direct Anthropic provider. Other model providers ignore it.
//
// Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-compaction.html
func WithCompaction(config *CompactionConfig) llms.CallOption {
	return func(opts *llms.CallOptions) {
		if opts.Metadata == nil {
			opts.Metadata = make(map[string]any)
		}
		opts.Metadata[bedrockclient.MetadataKeyCompaction] = config
	}
}

// WithAnthropicBeta enables Anthropic beta features on Claude models, sent as the
// anthropic_beta field of the InvokeModel body. Other model providers ignore it.
func WithAnthropicBeta(betas ...string) llms.CallOption {
	return func(opts *llms.CallOptions) {
		if opts.Metadata == nil {
			opts.Metadata = make(map[string]any)
		}
		existing, _ := opts.Metadata[bedrockclient.MetadataKeyAnthropicBeta].([]string)
		opts.Metadata[bedrockclient.MetadataKeyAnthropicBeta] = append(existing, betas...)
	}
}
