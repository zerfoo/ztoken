package gguf

import (
	"testing"
)

// testMetadata implements Metadata for testing.
type testMetadata struct {
	strings      map[string]string
	stringArrays map[string][]string
	uint32s      map[string]uint32
	int32Arrays  map[string][]int32
}

func (m *testMetadata) GetString(key string) (string, bool) {
	v, ok := m.strings[key]
	return v, ok
}

func (m *testMetadata) GetStringArray(key string) ([]string, bool) {
	v, ok := m.stringArrays[key]
	return v, ok
}

func (m *testMetadata) GetUint32(key string) (uint32, bool) {
	v, ok := m.uint32s[key]
	return v, ok
}

func (m *testMetadata) GetInt32Array(key string) ([]int32, bool) {
	v, ok := m.int32Arrays[key]
	return v, ok
}

func TestExtractTokenizer(t *testing.T) {
	m := &testMetadata{
		strings: map[string]string{},
		stringArrays: map[string][]string{
			"tokenizer.ggml.tokens": {"<unk>", "<s>", "</s>", "hello", "world"},
			"tokenizer.ggml.merges": {"hel lo", "wor ld"},
		},
		uint32s: map[string]uint32{
			"tokenizer.ggml.bos_token_id":     1,
			"tokenizer.ggml.eos_token_id":     2,
			"tokenizer.ggml.unknown_token_id": 0,
		},
		int32Arrays: map[string][]int32{},
	}

	tok, err := ExtractTokenizer(m)
	if err != nil {
		t.Fatalf("ExtractTokenizer error: %v", err)
	}

	if got := tok.VocabSize(); got != 5 {
		t.Errorf("VocabSize() = %d, want 5", got)
	}

	special := tok.SpecialTokens()
	if special.BOS != 1 {
		t.Errorf("BOS = %d, want 1", special.BOS)
	}
	if special.EOS != 2 {
		t.Errorf("EOS = %d, want 2", special.EOS)
	}
	if special.UNK != 0 {
		t.Errorf("UNK = %d, want 0", special.UNK)
	}

	// Verify token lookup.
	if tok, ok := tok.GetToken(3); !ok || tok != "hello" {
		t.Errorf("GetToken(3) = (%q, %v), want (\"hello\", true)", tok, ok)
	}
}

func TestExtractTokenizer_MissingTokens(t *testing.T) {
	m := &testMetadata{
		strings:      map[string]string{},
		stringArrays: map[string][]string{},
		uint32s:      map[string]uint32{},
		int32Arrays:  map[string][]int32{},
	}

	_, err := ExtractTokenizer(m)
	if err == nil {
		t.Fatal("expected error for missing tokens")
	}
}

func TestExtractTokenizer_SentencePiece(t *testing.T) {
	m := &testMetadata{
		strings: map[string]string{
			"tokenizer.ggml.model": "llama",
		},
		stringArrays: map[string][]string{
			"tokenizer.ggml.tokens": {"\u2581hello", "\u2581world"},
		},
		uint32s:     map[string]uint32{},
		int32Arrays: map[string][]int32{},
	}

	tok, err := ExtractTokenizer(m)
	if err != nil {
		t.Fatalf("ExtractTokenizer error: %v", err)
	}

	// Decode should strip leading space from SentencePiece tokens.
	decoded, err := tok.Decode([]int{0, 1})
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != "hello world" {
		t.Errorf("Decode = %q, want %q", decoded, "hello world")
	}
}

func TestExtractTokenizer_ControlTokens(t *testing.T) {
	m := &testMetadata{
		strings: map[string]string{},
		stringArrays: map[string][]string{
			"tokenizer.ggml.tokens": {"<unk>", "<s>", "</s>", "<start_of_turn>", "hello"},
		},
		uint32s: map[string]uint32{
			"tokenizer.ggml.bos_token_id": 1,
			"tokenizer.ggml.eos_token_id": 2,
		},
		int32Arrays: map[string][]int32{
			"tokenizer.ggml.token_type": {1, 3, 3, 3, 1},
		},
	}

	tok, err := ExtractTokenizer(m)
	if err != nil {
		t.Fatalf("ExtractTokenizer error: %v", err)
	}

	// <start_of_turn> should encode as a single token (ID 3).
	ids, err := tok.Encode("<start_of_turn>")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 || ids[0] != 3 {
		t.Errorf("Encode(\"<start_of_turn>\") = %v, want [3]", ids)
	}
}

func TestExtractTokenizer_InvalidMerge(t *testing.T) {
	m := &testMetadata{
		strings: map[string]string{},
		stringArrays: map[string][]string{
			"tokenizer.ggml.tokens": {"a", "b"},
			"tokenizer.ggml.merges": {"nospace"},
		},
		uint32s:     map[string]uint32{},
		int32Arrays: map[string][]int32{},
	}

	_, err := ExtractTokenizer(m)
	if err == nil {
		t.Fatal("expected error for invalid merge")
	}
}
