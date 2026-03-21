package ztoken

import (
	"os"
	"path/filepath"
	"testing"
)

func testWordPieceVocab() map[string]int {
	return map[string]int{
		"[PAD]":  0,
		"[UNK]":  1,
		"[CLS]":  2,
		"[SEP]":  3,
		"[MASK]": 4,
		"hello":  5,
		"world":  6,
		"un":     7,
		"##aff":  8,
		"##able": 9,
		"the":    10,
		"##s":    11,
		"cat":    12,
		"dog":    13,
		"play":   14,
		"##ing":  15,
		"a":      16,
		",":      17,
		".":      18,
	}
}

func testWordPieceTokenizer() *WordPieceTokenizer {
	vocab := testWordPieceVocab()
	special := SpecialTokens{
		BOS: 2,  // [CLS]
		EOS: 3,  // [SEP]
		PAD: 0,  // [PAD]
		UNK: 1,  // [UNK]
	}
	return NewWordPieceTokenizer(vocab, special)
}

func TestWordPieceTokenizer_Encode(t *testing.T) {
	tok := testWordPieceTokenizer()

	tests := []struct {
		name  string
		input string
		want  []int
	}{
		{"single word", "hello", []int{5}},
		{"two words", "hello world", []int{5, 6}},
		{"subword split", "unaffable", []int{7, 8, 9}},
		{"unknown word", "xyzzy", []int{1}},
		{"empty string", "", []int{}},
		{"punctuation split", "hello,world", []int{5, 17, 6}},
		{"continuation suffix", "cats", []int{12, 11}},
		{"playing", "playing", []int{14, 15}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ids, err := tok.Encode(tt.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tt.input, err)
			}
			if len(ids) != len(tt.want) {
				t.Fatalf("Encode(%q) = %v (len=%d), want %v (len=%d)",
					tt.input, ids, len(ids), tt.want, len(tt.want))
			}
			for i, id := range ids {
				if id != tt.want[i] {
					t.Errorf("Encode(%q)[%d] = %d, want %d", tt.input, i, id, tt.want[i])
				}
			}
		})
	}
}

func TestWordPieceTokenizer_Decode(t *testing.T) {
	tok := testWordPieceTokenizer()

	tests := []struct {
		name  string
		ids   []int
		want  string
	}{
		{"single word", []int{5}, "hello"},
		{"two words", []int{5, 6}, "hello world"},
		{"subword joined", []int{7, 8, 9}, "unaffable"},
		{"with continuation", []int{12, 11}, "cats"},
		{"skip CLS/SEP", []int{2, 5, 3}, "hello"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decoded, err := tok.Decode(tt.ids)
			if err != nil {
				t.Fatalf("Decode(%v) error: %v", tt.ids, err)
			}
			if decoded != tt.want {
				t.Errorf("Decode(%v) = %q, want %q", tt.ids, decoded, tt.want)
			}
		})
	}
}

func TestWordPieceTokenizer_RoundTrip(t *testing.T) {
	tok := testWordPieceTokenizer()

	texts := []string{
		"hello world",
		"the cat",
		"playing",
		"cats",
		"unaffable",
	}

	for _, text := range texts {
		ids, err := tok.Encode(text)
		if err != nil {
			t.Fatalf("Encode(%q) error: %v", text, err)
		}
		decoded, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode(%v) error: %v", ids, err)
		}
		if decoded != text {
			t.Errorf("round-trip: %q -> %v -> %q", text, ids, decoded)
		}
	}
}

func TestWordPieceTokenizer_VocabSize(t *testing.T) {
	tok := testWordPieceTokenizer()
	if got := tok.VocabSize(); got != 19 {
		t.Errorf("VocabSize() = %d, want 19", got)
	}
}

func TestWordPieceTokenizer_GetToken(t *testing.T) {
	tok := testWordPieceTokenizer()
	tok1, ok := tok.GetToken(5)
	if !ok || tok1 != "hello" {
		t.Errorf("GetToken(5) = (%q, %v), want (\"hello\", true)", tok1, ok)
	}
	_, ok = tok.GetToken(999)
	if ok {
		t.Error("GetToken(999) should return false")
	}
}

func TestWordPieceTokenizer_GetID(t *testing.T) {
	tok := testWordPieceTokenizer()
	id, ok := tok.GetID("hello")
	if !ok || id != 5 {
		t.Errorf("GetID(\"hello\") = (%d, %v), want (5, true)", id, ok)
	}
	_, ok = tok.GetID("nonexistent")
	if ok {
		t.Error("GetID(\"nonexistent\") should return false")
	}
}

func TestWordPieceTokenizer_SpecialTokens(t *testing.T) {
	tok := testWordPieceTokenizer()
	sp := tok.SpecialTokens()
	if sp.BOS != 2 {
		t.Errorf("BOS = %d, want 2", sp.BOS)
	}
	if sp.EOS != 3 {
		t.Errorf("EOS = %d, want 3", sp.EOS)
	}
	if sp.PAD != 0 {
		t.Errorf("PAD = %d, want 0", sp.PAD)
	}
	if sp.UNK != 1 {
		t.Errorf("UNK = %d, want 1", sp.UNK)
	}
}

func TestWordPieceTokenizer_EncodeForBERT_Single(t *testing.T) {
	tok := testWordPieceTokenizer()

	enc, err := tok.EncodeForBERT("hello world", "", 0)
	if err != nil {
		t.Fatalf("EncodeForBERT error: %v", err)
	}

	// [CLS]=2, hello=5, world=6, [SEP]=3
	wantIDs := []int{2, 5, 6, 3}
	wantMask := []int{1, 1, 1, 1}
	wantTypes := []int{0, 0, 0, 0}

	assertIntSlice(t, "InputIDs", enc.InputIDs, wantIDs)
	assertIntSlice(t, "AttentionMask", enc.AttentionMask, wantMask)
	assertIntSlice(t, "TokenTypeIDs", enc.TokenTypeIDs, wantTypes)
}

func TestWordPieceTokenizer_EncodeForBERT_Pair(t *testing.T) {
	tok := testWordPieceTokenizer()

	enc, err := tok.EncodeForBERT("hello", "world", 0)
	if err != nil {
		t.Fatalf("EncodeForBERT error: %v", err)
	}

	// [CLS]=2, hello=5, [SEP]=3, world=6, [SEP]=3
	wantIDs := []int{2, 5, 3, 6, 3}
	wantMask := []int{1, 1, 1, 1, 1}
	wantTypes := []int{0, 0, 0, 1, 1}

	assertIntSlice(t, "InputIDs", enc.InputIDs, wantIDs)
	assertIntSlice(t, "AttentionMask", enc.AttentionMask, wantMask)
	assertIntSlice(t, "TokenTypeIDs", enc.TokenTypeIDs, wantTypes)
}

func TestWordPieceTokenizer_EncodeForBERT_Padding(t *testing.T) {
	tok := testWordPieceTokenizer()

	enc, err := tok.EncodeForBERT("hello", "", 8)
	if err != nil {
		t.Fatalf("EncodeForBERT error: %v", err)
	}

	// [CLS]=2, hello=5, [SEP]=3, [PAD]=0 x5
	wantIDs := []int{2, 5, 3, 0, 0, 0, 0, 0}
	wantMask := []int{1, 1, 1, 0, 0, 0, 0, 0}
	wantTypes := []int{0, 0, 0, 0, 0, 0, 0, 0}

	assertIntSlice(t, "InputIDs", enc.InputIDs, wantIDs)
	assertIntSlice(t, "AttentionMask", enc.AttentionMask, wantMask)
	assertIntSlice(t, "TokenTypeIDs", enc.TokenTypeIDs, wantTypes)
}

func TestWordPieceTokenizer_PreTokenize(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"hello world", []string{"hello", "world"}},
		{"hello,world", []string{"hello", ",", "world"}},
		{"hello. world!", []string{"hello", ".", "world", "!"}},
		{"  spaces  ", []string{"spaces"}},
		{"", nil},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := preTokenize(tt.input)
			if len(got) != len(tt.want) {
				t.Fatalf("preTokenize(%q) = %v, want %v", tt.input, got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("preTokenize(%q)[%d] = %q, want %q", tt.input, i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestLoad_WordPiece(t *testing.T) {
	fixture := `{
  "model": {
    "type": "WordPiece",
    "vocab": {
      "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
      "hello": 5, "world": 6, "un": 7, "##aff": 8, "##able": 9
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[PAD]", "special": true},
    {"id": 1, "content": "[UNK]", "special": true},
    {"id": 2, "content": "[CLS]", "special": true},
    {"id": 3, "content": "[SEP]", "special": true},
    {"id": 4, "content": "[MASK]", "special": true}
  ],
  "normalizer": {
    "type": "Lowercase"
  }
}`
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	tok, err := Load(path)
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	wp, ok := tok.(*WordPieceTokenizer)
	if !ok {
		t.Fatalf("expected *WordPieceTokenizer, got %T", tok)
	}

	// Normalizer should lowercase.
	ids, err := wp.Encode("HELLO")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 || ids[0] != 5 {
		t.Errorf("Encode(\"HELLO\") = %v, want [5]", ids)
	}

	// Subword splitting.
	ids, err = wp.Encode("unaffable")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	wantIDs := []int{7, 8, 9}
	assertIntSlice(t, "subword", ids, wantIDs)

	// Special tokens.
	sp := tok.SpecialTokens()
	if sp.BOS != 2 || sp.EOS != 3 || sp.PAD != 0 || sp.UNK != 1 {
		t.Errorf("SpecialTokens = %+v", sp)
	}
}

func TestLoad_BPE(t *testing.T) {
	// Load should still work for BPE models.
	tok, err := Load(filepath.Join("testdata", "tokenizer.json"))
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if _, ok := tok.(*BPETokenizer); !ok {
		t.Fatalf("expected *BPETokenizer, got %T", tok)
	}
}

func TestLoad_UnsupportedType(t *testing.T) {
	fixture := `{
  "model": {"type": "Unigram", "vocab": {}},
  "added_tokens": []
}`
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for Unigram model type")
	}
}

func assertIntSlice(t *testing.T, name string, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len=%d, want len=%d; got %v, want %v", name, len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s[%d] = %d, want %d", name, i, got[i], want[i])
		}
	}
}
