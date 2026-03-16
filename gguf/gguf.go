// Package gguf extracts a BPE tokenizer from GGUF file metadata.
//
// It defines a Metadata interface so callers can provide any GGUF parser
// implementation without depending on a specific one.
package gguf

import (
	"fmt"
	"strings"

	"github.com/zerfoo/ztoken"
)

// Metadata provides access to GGUF key-value metadata.
type Metadata interface {
	GetString(key string) (string, bool)
	GetStringArray(key string) ([]string, bool)
	GetUint32(key string) (uint32, bool)
	GetInt32Array(key string) ([]int32, bool)
}

// ExtractTokenizer builds a BPETokenizer from GGUF metadata. GGUF files store
// tokenizer data under the "tokenizer.ggml.*" metadata keys.
func ExtractTokenizer(m Metadata) (*ztoken.BPETokenizer, error) {
	// Extract token vocabulary.
	tokens, ok := m.GetStringArray("tokenizer.ggml.tokens")
	if !ok {
		return nil, fmt.Errorf("missing tokenizer.ggml.tokens metadata")
	}

	vocab := make(map[string]int, len(tokens))
	for i, s := range tokens {
		vocab[s] = i
	}

	// Extract merges (optional -- some models have no merges).
	var merges []ztoken.MergePair
	if mergeStrs, ok := m.GetStringArray("tokenizer.ggml.merges"); ok {
		merges = make([]ztoken.MergePair, 0, len(mergeStrs))
		for i, s := range mergeStrs {
			left, right, found := strings.Cut(s, " ")
			if !found {
				return nil, fmt.Errorf("tokenizer.ggml.merges[%d]: invalid merge %q", i, s)
			}
			merges = append(merges, ztoken.MergePair{Left: left, Right: right})
		}
	}

	// Extract special token IDs.
	special := ztoken.SpecialTokens{}
	if v, ok := m.GetUint32("tokenizer.ggml.bos_token_id"); ok {
		special.BOS = int(v)
	}
	if v, ok := m.GetUint32("tokenizer.ggml.eos_token_id"); ok {
		special.EOS = int(v)
	}
	if v, ok := m.GetUint32("tokenizer.ggml.unknown_token_id"); ok {
		special.UNK = int(v)
	}
	if v, ok := m.GetUint32("tokenizer.ggml.padding_token_id"); ok {
		special.PAD = int(v)
	}

	// GGUF tokenizers are not byte-level BPE (they use raw token strings).
	tok := ztoken.NewBPETokenizer(vocab, merges, special, false)

	// SentencePiece models (tokenizer.ggml.model = "llama") use U+2581
	// as a space marker. Enable SentencePiece pre-tokenization for these.
	if model, ok := m.GetString("tokenizer.ggml.model"); ok && model == "llama" {
		tok.SetSentencePiece(true)
	}

	// Extract control/special tokens (token_type == 3) for exact matching
	// during encoding. Without this, tokens like <start_of_turn> would be
	// split into characters by BPE.
	if types, ok := m.GetInt32Array("tokenizer.ggml.token_type"); ok {
		specialTokens := make(map[string]int)
		for i, tokenType := range types {
			// Type 3 = control/special token.
			if tokenType == 3 && i < len(tokens) && len(tokens[i]) > 0 {
				specialTokens[tokens[i]] = i
			}
		}
		if len(specialTokens) > 0 {
			tok.SetSpecialTokenStrings(specialTokens)
		}
	}

	return tok, nil
}
