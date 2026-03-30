package ztoken

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestBPETokenizer_RoundTripVaried verifies encode/decode identity on varied inputs.
func TestBPETokenizer_RoundTripVaried(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		name  string
		input string
	}{
		{"single word", "hello"},
		{"single char in vocab", "h"},
		{"repeated word", "hello hello hello"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			decoded, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("Decode(%v) error: %v", ids, err)
			}
			if decoded != tc.input {
				t.Errorf("round-trip failed: %q -> %v -> %q", tc.input, ids, decoded)
			}
		})
	}
}

// TestBPETokenizer_SentencePieceRoundTrip verifies round-trip identity for
// SentencePiece-mode tokenizers on varied inputs including unicode.
func TestBPETokenizer_SentencePieceRoundTrip(t *testing.T) {
	vocab := map[string]int{
		"<unk>":        0,
		"<s>":          1,
		"</s>":         2,
		"\u2581hello":  3,
		"\u2581world":  4,
		"\u2581foo":    5,
		"\u2581bar":    6,
		"\u2581a":      7,
		"\u2581":       8,
		"h":            9,
		"e":            10,
		"l":            11,
		"o":            12,
	}
	special := SpecialTokens{BOS: 1, EOS: 2, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSentencePiece(true)
	tok.SetScores([]float32{
		-100, -100, -100,
		-1, -1, -1, -1, -1, -1,
		-5, -5, -5, -5,
	})

	tests := []struct {
		name  string
		input string
	}{
		{"two words", "hello world"},
		{"three words", "foo bar a"},
		{"single word", "hello"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			decoded, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("Decode(%v) error: %v", ids, err)
			}
			if decoded != tc.input {
				t.Errorf("round-trip failed: %q -> %v -> %q", tc.input, ids, decoded)
			}
		})
	}
}

// TestWhitespaceTokenizer_RoundTripVaried verifies encode/decode identity for
// the whitespace tokenizer on varied inputs.
func TestWhitespaceTokenizer_RoundTripVaried(t *testing.T) {
	tok := NewWhitespaceTokenizer()
	tok.AddToken("hello")
	tok.AddToken("world")
	tok.AddToken("foo")
	tok.AddToken("bar")

	tests := []struct {
		name  string
		input string
	}{
		{"single word", "hello"},
		{"four words", "hello world foo bar"},
		{"repeated", "hello hello hello"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			decoded, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("Decode(%v) error: %v", ids, err)
			}
			if decoded != tc.input {
				t.Errorf("round-trip failed: %q -> %v -> %q", tc.input, ids, decoded)
			}
		})
	}
}

// TestBPETokenizer_UnicodeMultibyte tests encoding of multibyte unicode sequences.
func TestBPETokenizer_UnicodeMultibyte(t *testing.T) {
	// Vocabulary with multibyte unicode tokens.
	vocab := map[string]int{
		"<unk>":       0,
		"<s>":         1,
		"</s>":        2,
		"\u2581":      3, // ▁
		"\u2581café":  4,
		"\u2581naïve": 5,
		"c":           6,
		"a":           7,
		"f":           8,
		"é":           9,
		// Byte fallback tokens for multibyte chars.
		"<0xC3>": 10,
		"<0xA9>": 11, // é in UTF-8
		"<0xAF>": 12, // ï in UTF-8 (second byte)
		"<0xC3>": 10, // shared
	}
	special := SpecialTokens{BOS: 1, EOS: 2, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSentencePiece(true)
	tok.SetScores([]float32{
		-100, -100, -100, -100,
		-1, -1, -5, -5, -5, -5, -5, -5, -5,
	})

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{"accented word", "café", []int{4}},
		{"diaeresis word", "naïve", []int{5}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			if len(ids) != len(tc.wantIDs) {
				t.Fatalf("Encode(%q) = %v (len=%d), want %v", tc.input, ids, len(ids), tc.wantIDs)
			}
			for i, id := range ids {
				if id != tc.wantIDs[i] {
					t.Errorf("Encode(%q)[%d] = %d, want %d", tc.input, i, id, tc.wantIDs[i])
				}
			}
		})
	}
}

// TestBPETokenizer_ControlChars tests encoding of control characters.
func TestBPETokenizer_ControlChars(t *testing.T) {
	vocab := map[string]int{
		"<unk>":  0,
		"<s>":    1,
		"</s>":   2,
		"hello":  3,
		"\t":     4,
		"\n":     5,
		"<0x00>": 6,
		"<0x01>": 7,
		"<0x7F>": 8,
	}
	special := SpecialTokens{BOS: 1, EOS: 2, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{"tab character", "\t", []int{4}},
		{"newline character", "\n", []int{5}},
		{"mixed text and tab", "hello\thello", []int{3, 4, 3}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			if len(ids) != len(tc.wantIDs) {
				t.Fatalf("Encode(%q) = %v (len=%d), want %v", tc.input, ids, len(ids), tc.wantIDs)
			}
			for i, id := range ids {
				if id != tc.wantIDs[i] {
					t.Errorf("Encode(%q)[%d] = %d, want %d", tc.input, i, id, tc.wantIDs[i])
				}
			}
		})
	}
}

// TestBPETokenizer_SentencePieceByteFallback tests byte fallback encoding
// for characters not in the SentencePiece vocabulary.
func TestBPETokenizer_SentencePieceByteFallback(t *testing.T) {
	vocab := map[string]int{
		"<unk>":  0,
		"<s>":    1,
		"</s>":   2,
		"\u2581": 3,
		"a":      4,
		"b":      5,
		"<0xC3>": 6,
		"<0xA9>": 7,  // UTF-8 bytes for 'é' (0xC3, 0xA9)
		"<0xF0>": 8,
		"<0x9F>": 9,
		"<0x98>": 10,
		"<0x80>": 11, // UTF-8 bytes for '😀' (0xF0, 0x9F, 0x98, 0x80)
	}
	special := SpecialTokens{BOS: 1, EOS: 2, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSentencePiece(true)
	tok.SetScores([]float32{
		-100, -100, -100, -100,
		-1, -1, -1, -1, -1, -1, -1, -1,
	})

	t.Run("byte fallback for accented char", func(t *testing.T) {
		// 'é' not in vocab as a single token, so should use byte fallback.
		ids, err := tok.Encode("é")
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}
		// Expect ▁ + <0xC3> + <0xA9>
		if len(ids) < 2 {
			t.Fatalf("expected byte fallback tokens, got %v", ids)
		}
	})

	t.Run("byte fallback decode round-trip", func(t *testing.T) {
		// Manually construct byte fallback tokens and decode them.
		// ▁ + <0xC3> + <0xA9> should decode to "é"
		decoded, err := tok.Decode([]int{3, 6, 7})
		if err != nil {
			t.Fatalf("Decode error: %v", err)
		}
		if decoded != "é" {
			t.Errorf("Decode byte fallback = %q, want %q", decoded, "é")
		}
	})

	t.Run("emoji byte fallback decode", func(t *testing.T) {
		// ▁ + <0xF0> + <0x9F> + <0x98> + <0x80> should decode to "😀"
		decoded, err := tok.Decode([]int{3, 8, 9, 10, 11})
		if err != nil {
			t.Fatalf("Decode error: %v", err)
		}
		if decoded != "😀" {
			t.Errorf("Decode emoji byte fallback = %q, want %q", decoded, "😀")
		}
	})
}

// TestBPETokenizer_EmptyAndWhitespace tests edge cases with empty and
// whitespace-only inputs.
func TestBPETokenizer_EmptyAndWhitespace(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		name    string
		input   string
		wantLen int
	}{
		{"empty string", "", 0},
		{"single space", " ", 0},
		{"multiple spaces", "     ", 0},
		{"tab only", "\t", 0},
		{"newline only", "\n", 0},
		{"mixed whitespace", " \t\n\r ", 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			if len(ids) != tc.wantLen {
				t.Errorf("Encode(%q) len = %d, want %d; got %v", tc.input, len(ids), tc.wantLen, ids)
			}
		})
	}
}

// TestDecodeSentencePieceBytes tests the decodeSentencePieceBytes helper
// with various hex byte patterns.
func TestDecodeSentencePieceBytes(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"no hex bytes", "hello", "hello"},
		{"single hex byte", "<0x41>", "A"},
		{"multiple hex bytes", "<0x48><0x69>", "Hi"},
		{"mixed text and hex", "hello<0x21>", "hello!"},
		{"incomplete pattern", "<0x4", "<0x4"},
		{"invalid hex digit", "<0xGG>", "<0xGG>"},
		{"lowercase hex", "<0xff>", "\xff"},
		{"null byte", "<0x00>", "\x00"},
		{"uppercase hex", "<0xFF>", "\xff"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := decodeSentencePieceBytes(tc.input)
			if got != tc.want {
				t.Errorf("decodeSentencePieceBytes(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

// TestUnhex tests the unhex helper for all valid and invalid inputs.
func TestUnhex(t *testing.T) {
	tests := []struct {
		input byte
		want  int
	}{
		{'0', 0}, {'1', 1}, {'9', 9},
		{'A', 10}, {'F', 15},
		{'a', 10}, {'f', 15},
		{'G', -1}, {'g', -1}, {'/', -1}, {':', -1}, {' ', -1},
	}

	for _, tc := range tests {
		t.Run(string(tc.input), func(t *testing.T) {
			got := unhex(tc.input)
			if got != tc.want {
				t.Errorf("unhex(%q) = %d, want %d", tc.input, got, tc.want)
			}
		})
	}
}

// TestWordPieceTokenizer_UnicodeMultibyte tests WordPiece with multibyte
// unicode tokens.
func TestWordPieceTokenizer_UnicodeMultibyte(t *testing.T) {
	vocab := map[string]int{
		"[PAD]":  0,
		"[UNK]":  1,
		"[CLS]":  2,
		"[SEP]":  3,
		"café":   4,
		"naïve":  5,
		"über":   6,
	}
	special := SpecialTokens{BOS: 2, EOS: 3, PAD: 0, UNK: 1}
	tok := NewWordPieceTokenizer(vocab, special)

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{"accented", "café", []int{4}},
		{"diaeresis", "naïve", []int{5}},
		{"umlaut", "über", []int{6}},
		{"unknown multibyte", "résumé", []int{1}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			if len(ids) != len(tc.wantIDs) {
				t.Fatalf("Encode(%q) = %v (len=%d), want %v", tc.input, ids, len(ids), tc.wantIDs)
			}
			for i, id := range ids {
				if id != tc.wantIDs[i] {
					t.Errorf("Encode(%q)[%d] = %d, want %d", tc.input, i, id, tc.wantIDs[i])
				}
			}
		})
	}
}

// TestWordPieceTokenizer_ControlChars tests encoding of strings containing
// control characters in WordPiece mode.
func TestWordPieceTokenizer_ControlChars(t *testing.T) {
	vocab := map[string]int{
		"[PAD]": 0,
		"[UNK]": 1,
		"[CLS]": 2,
		"[SEP]": 3,
		"hello": 4,
		"world": 5,
	}
	special := SpecialTokens{BOS: 2, EOS: 3, PAD: 0, UNK: 1}
	tok := NewWordPieceTokenizer(vocab, special)

	tests := []struct {
		name    string
		input   string
		wantLen int
	}{
		{"tab separated", "hello\tworld", 2},
		{"newline separated", "hello\nworld", 2},
		{"only whitespace", "\t\n\r", 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			if len(ids) != tc.wantLen {
				t.Errorf("Encode(%q) len = %d, want %d; got %v", tc.input, len(ids), tc.wantLen, ids)
			}
		})
	}
}

// TestLoadFromJSON_MalformedMetadata tests error paths for malformed tokenizer
// JSON metadata.
func TestLoadFromJSON_MalformedMetadata(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		wantErr bool
	}{
		{
			name:    "empty object",
			json:    `{}`,
			wantErr: false, // empty vocab is valid
		},
		{
			name:    "null model",
			json:    `{"model": null}`,
			wantErr: false,
		},
		{
			name:    "missing vocab key",
			json:    `{"model": {"type": "BPE"}}`,
			wantErr: false,
		},
		{
			name:    "invalid merges type",
			json:    `{"model": {"type": "BPE", "vocab": {}, "merges": 42}}`,
			wantErr: true,
		},
		{
			name:    "merges with three-element array",
			json:    `{"model": {"type": "BPE", "vocab": {}, "merges": [["a","b","c"]]}}`,
			wantErr: true,
		},
		{
			name:    "truncated JSON",
			json:    `{"model": {"type": "BPE"`,
			wantErr: true,
		},
		{
			name:    "empty merges string entry",
			json:    `{"model": {"type": "BPE", "vocab": {}, "merges": [""]}}`,
			wantErr: true,
		},
		{
			name:    "non-BPE model type",
			json:    `{"model": {"type": "Unigram"}}`,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "tokenizer.json")
			if err := os.WriteFile(path, []byte(tc.json), 0o600); err != nil {
				t.Fatal(err)
			}

			_, err := LoadFromJSON(path)
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

// TestLoad_MalformedMetadata tests error paths for the Load function with
// various malformed JSON inputs.
func TestLoad_MalformedMetadata(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		wantErr bool
	}{
		{
			name:    "completely empty file",
			json:    ``,
			wantErr: true,
		},
		{
			name:    "binary content",
			json:    "\x00\x01\x02\x03",
			wantErr: true,
		},
		{
			name:    "unsupported model type",
			json:    `{"model": {"type": "Unigram"}, "added_tokens": []}`,
			wantErr: true,
		},
		{
			name:    "WordPiece with missing vocab",
			json:    `{"model": {"type": "WordPiece"}, "added_tokens": []}`,
			wantErr: false, // empty vocab is valid
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "tokenizer.json")
			if err := os.WriteFile(path, []byte(tc.json), 0o600); err != nil {
				t.Fatal(err)
			}

			_, err := Load(path)
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

// TestLoadFromJSON_EmptyFile tests loading from a zero-byte file.
func TestLoadFromJSON_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := LoadFromJSON(path)
	if err == nil {
		t.Error("expected error for empty file, got nil")
	}
}

// TestBPETokenizer_SentencePieceNewlines tests that newlines are handled
// correctly in SentencePiece mode.
func TestBPETokenizer_SentencePieceNewlines(t *testing.T) {
	vocab := map[string]int{
		"<unk>":       0,
		"<s>":         1,
		"</s>":        2,
		"\u2581hello": 3,
		"\u2581world": 4,
		"\n":          5,
	}
	special := SpecialTokens{BOS: 1, EOS: 2, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSentencePiece(true)
	tok.SetScores([]float32{-100, -100, -100, -1, -1, -1})

	ids, err := tok.Encode("hello\nworld")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	// Expect: ▁hello, \n, world (world may not have ▁ after newline)
	if len(ids) < 2 {
		t.Fatalf("expected at least 2 tokens, got %v", ids)
	}
	// First token should be ▁hello
	if ids[0] != 3 {
		t.Errorf("first token = %d, want 3 (▁hello)", ids[0])
	}
}

// TestBPETokenizer_SetAddLeadingSpace tests the addLeadingSpace toggle.
func TestBPETokenizer_SetAddLeadingSpace(t *testing.T) {
	vocab := map[string]int{
		"<unk>":       0,
		"<s>":         1,
		"</s>":        2,
		"\u2581hello": 3,
		"hello":       4,
	}
	special := SpecialTokens{BOS: 1, EOS: 2, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSentencePiece(true)
	tok.SetScores([]float32{-100, -100, -100, -1, -1})

	t.Run("with leading space", func(t *testing.T) {
		tok.SetAddLeadingSpace(true)
		ids, err := tok.Encode("hello")
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}
		if len(ids) != 1 || ids[0] != 3 {
			t.Errorf("with leading space: Encode(\"hello\") = %v, want [3]", ids)
		}
	})

	t.Run("without leading space", func(t *testing.T) {
		tok.SetAddLeadingSpace(false)
		ids, err := tok.Encode("hello")
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}
		if len(ids) != 1 || ids[0] != 4 {
			t.Errorf("without leading space: Encode(\"hello\") = %v, want [4]", ids)
		}
	})
}

// TestBPETokenizer_EncodeWithSpecialsAndSegments tests encoding text that
// contains special token strings mixed with regular text.
func TestBPETokenizer_EncodeWithSpecialsAndSegments(t *testing.T) {
	vocab := map[string]int{
		"<unk>":            0,
		"<s>":              1,
		"</s>":             2,
		"<start_of_turn>":  3,
		"<end_of_turn>":    4,
		"hello":            5,
		"world":            6,
	}
	special := SpecialTokens{BOS: 1, EOS: 2, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSpecialTokenStrings(map[string]int{
		"<start_of_turn>": 3,
		"<end_of_turn>":   4,
	})

	ids, err := tok.Encode("<start_of_turn>hello<end_of_turn>")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	want := []int{3, 5, 4}
	if len(ids) != len(want) {
		t.Fatalf("Encode = %v, want %v", ids, want)
	}
	for i, id := range ids {
		if id != want[i] {
			t.Errorf("[%d] = %d, want %d", i, id, want[i])
		}
	}
}

// TestBPETokenizer_DecodeUnknownID tests that Decode returns an error for
// unknown token IDs.
func TestBPETokenizer_DecodeUnknownID(t *testing.T) {
	tok := makeTestBPE()
	_, err := tok.Decode([]int{9999})
	if err == nil {
		t.Error("expected error for unknown token ID, got nil")
	}
	if !strings.Contains(err.Error(), "unknown token ID") {
		t.Errorf("error = %q, want containing 'unknown token ID'", err.Error())
	}
}

// TestBPETokenizer_NormalizerApplied tests that the normalizer is applied
// before encoding.
func TestBPETokenizer_NormalizerApplied(t *testing.T) {
	tok := makeTestBPE()
	tok.normalizer = strings.ToLower

	// "HELLO" lowercased to "hello" which is token 17.
	ids, err := tok.Encode("HELLO")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 || ids[0] != 17 {
		t.Errorf("Encode(\"HELLO\") with lowercase normalizer = %v, want [17]", ids)
	}
}

// TestWordPieceTokenizer_EmptyAndWhitespace tests edge cases for WordPiece.
func TestWordPieceTokenizer_EmptyAndWhitespace(t *testing.T) {
	tok := testWordPieceTokenizer()

	tests := []struct {
		name    string
		input   string
		wantLen int
	}{
		{"empty", "", 0},
		{"spaces only", "   ", 0},
		{"tabs only", "\t\t", 0},
		{"newlines only", "\n\n\n", 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			if len(ids) != tc.wantLen {
				t.Errorf("Encode(%q) len = %d, want %d", tc.input, len(ids), tc.wantLen)
			}
		})
	}
}

// TestPreTokenize_Unicode tests pre-tokenization with unicode punctuation
// and mixed scripts.
func TestPreTokenize_Unicode(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{"ascii punctuation", "hello,world!", []string{"hello", ",", "world", "!"}},
		{"unicode em-dash", "hello\u2014world", []string{"hello", "\u2014", "world"}},
		{"cjk characters", "hello世界", []string{"hello世界"}},
		{"mixed with spaces", "a , b", []string{"a", ",", "b"}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := preTokenize(tc.input)
			if len(got) != len(tc.want) {
				t.Fatalf("preTokenize(%q) = %v (len=%d), want %v (len=%d)",
					tc.input, got, len(got), tc.want, len(tc.want))
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("preTokenize(%q)[%d] = %q, want %q", tc.input, i, got[i], tc.want[i])
				}
			}
		})
	}
}

// TestIsPrintableGPT2Byte tests the GPT-2 printable byte classification.
func TestIsPrintableGPT2Byte(t *testing.T) {
	tests := []struct {
		name string
		b    byte
		want bool
	}{
		{"exclamation", '!', true},
		{"tilde", '~', true},
		{"space", ' ', false},
		{"null", 0, false},
		{"tab", '\t', false},
		{"newline", '\n', false},
		{"DEL", 0x7F, false},
		{"latin supplement start", 0xA1, true},
		{"0xAC", 0xAC, true},
		{"0xAD", 0xAD, false},
		{"0xAE", 0xAE, true},
		{"0xFF", 0xFF, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := isPrintableGPT2Byte(tc.b)
			if got != tc.want {
				t.Errorf("isPrintableGPT2Byte(0x%02X) = %v, want %v", tc.b, got, tc.want)
			}
		})
	}
}
