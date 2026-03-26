package ztoken

import (
	"fmt"
	"testing"
)

func makeTestBPE() *BPETokenizer {
	// Small vocabulary for testing.
	vocab := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
		"<pad>": 3,
		"h":     4,
		"e":     5,
		"l":     6,
		"o":     7,
		"w":     8,
		"r":     9,
		"d":     10,
		" ":     11,
		"he":    12,
		"ll":    13,
		"lo":    14,
		"hel":   15,
		"hell":  16,
		"hello": 17,
		"wo":    18,
		"wor":   19,
		"worl":  20,
		"world": 21,
	}
	// Merge order matters: BPE always picks the globally lowest-rank pair.
	// For "hello": h,e,l,l,o -> he,l,l,o -> hel,l,o -> hel,lo -> hello
	// For "world": w,o,r,l,d -> wo,r,l,d -> wor,l,d -> worl,d -> world
	merges := []MergePair{
		{Left: "h", Right: "e"},     // rank 0: h+e -> he
		{Left: "he", Right: "l"},    // rank 1: he+l -> hel (before l+l!)
		{Left: "l", Right: "o"},     // rank 2: l+o -> lo
		{Left: "hel", Right: "lo"},  // rank 3: hel+lo -> hello
		{Left: "w", Right: "o"},     // rank 4: w+o -> wo
		{Left: "wo", Right: "r"},    // rank 5: wo+r -> wor
		{Left: "wor", Right: "l"},   // rank 6: wor+l -> worl
		{Left: "worl", Right: "d"},  // rank 7: worl+d -> world
		{Left: "l", Right: "l"},     // rank 8: l+l -> ll (low priority)
		{Left: "hel", Right: "l"},   // rank 9: hel+l -> hell (alt path)
		{Left: "hell", Right: "o"},  // rank 10: hell+o -> hello (alt path)
	}
	special := SpecialTokens{BOS: 1, EOS: 2, PAD: 3, UNK: 0}
	return NewBPETokenizer(vocab, merges, special, false)
}

func TestBPETokenizer_Encode(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{"single word", "hello", []int{17}},
		{"two words whitespace split", "hello world", []int{17, 21}},
		{"empty string", "", []int{}},
		{"unknown characters", "xyz", []int{0, 0, 0}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			if len(ids) != len(tc.wantIDs) {
				t.Fatalf("Encode(%q) len = %d, want %d; got %v", tc.input, len(ids), len(tc.wantIDs), ids)
			}
			for i, id := range ids {
				if id != tc.wantIDs[i] {
					t.Errorf("Encode(%q)[%d] = %d, want %d", tc.input, i, id, tc.wantIDs[i])
				}
			}
		})
	}
}

func TestBPETokenizer_Decode(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		name     string
		ids      []int
		wantText string
		wantErr  bool
	}{
		{"single token", []int{17}, "hello", false},
		{"multiple tokens", []int{17, 11, 21}, "hello world", false},
		{"empty", []int{}, "", false},
		{"unknown ID", []int{999}, "", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tok.Decode(tc.ids)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("Decode(%v) expected error, got %q", tc.ids, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("Decode(%v) error: %v", tc.ids, err)
			}
			if got != tc.wantText {
				t.Errorf("Decode(%v) = %q, want %q", tc.ids, got, tc.wantText)
			}
		})
	}
}

func TestBPETokenizer_RoundTrip(t *testing.T) {
	tok := makeTestBPE()
	// Add space token for round-trip
	text := "hello"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode(%q) error: %v", text, err)
	}
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode(%v) error: %v", ids, err)
	}
	if decoded != text {
		t.Errorf("round-trip: Encode(%q) = %v, Decode = %q", text, ids, decoded)
	}
}

func TestBPETokenizer_EncodeWithSpecialTokens(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		name    string
		text    string
		addBOS  bool
		addEOS  bool
		wantIDs []int
	}{
		{"no special tokens", "hello", false, false, []int{17}},
		{"with BOS", "hello", true, false, []int{1, 17}},
		{"with EOS", "hello", false, true, []int{17, 2}},
		{"with both", "hello", true, true, []int{1, 17, 2}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.EncodeWithSpecialTokens(tc.text, tc.addBOS, tc.addEOS)
			if err != nil {
				t.Fatalf("EncodeWithSpecialTokens error: %v", err)
			}
			if len(ids) != len(tc.wantIDs) {
				t.Fatalf("got %v (len=%d), want %v (len=%d)", ids, len(ids), tc.wantIDs, len(tc.wantIDs))
			}
			for i, id := range ids {
				if id != tc.wantIDs[i] {
					t.Errorf("[%d] = %d, want %d", i, id, tc.wantIDs[i])
				}
			}
		})
	}
}

func TestBPETokenizer_VocabSize(t *testing.T) {
	tok := makeTestBPE()
	if got := tok.VocabSize(); got != 22 {
		t.Errorf("VocabSize() = %d, want 22", got)
	}
}

func TestBPETokenizer_GetToken(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		id     int
		want   string
		wantOK bool
	}{
		{17, "hello", true},
		{0, "<unk>", true},
		{999, "", false},
	}

	for _, tc := range tests {
		got, ok := tok.GetToken(tc.id)
		if ok != tc.wantOK || got != tc.want {
			t.Errorf("GetToken(%d) = (%q, %v), want (%q, %v)", tc.id, got, ok, tc.want, tc.wantOK)
		}
	}
}

func TestBPETokenizer_GetID(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		token  string
		wantID int
		wantOK bool
	}{
		{"hello", 17, true},
		{"<unk>", 0, true},
		{"missing", 0, false},
	}

	for _, tc := range tests {
		id, ok := tok.GetID(tc.token)
		if ok != tc.wantOK || id != tc.wantID {
			t.Errorf("GetID(%q) = (%d, %v), want (%d, %v)", tc.token, id, ok, tc.wantID, tc.wantOK)
		}
	}
}

func TestBPETokenizer_MergeOrder(t *testing.T) {
	// Verify that BPE merges are applied in priority order.
	// With merges: h+e (0), l+l (1), l+o (2), he+l (3), hel+l (4), hell+o (5)
	// "hello" should merge as: h,e,l,l,o -> he,l,l,o -> hel,l,o -> hell,o -> hello
	tok := makeTestBPE()
	ids, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 || ids[0] != 17 {
		t.Errorf("Encode(\"hello\") = %v, want [17]", ids)
	}
}

func TestByteLevelBPEEncoderDecoder(t *testing.T) {
	enc, dec := buildByteEncoderDecoder()

	// All 256 bytes must be mapped.
	if len(enc) != 256 {
		t.Errorf("encoder size = %d, want 256", len(enc))
	}
	if len(dec) != 256 {
		t.Errorf("decoder size = %d, want 256", len(dec))
	}

	// Verify round-trip for every byte.
	for i := range 256 {
		b := byte(i)
		r := enc[b]
		got, ok := dec[r]
		if !ok {
			t.Errorf("byte %d: encoded to %c (U+%04X), decoder missing", i, r, r)
			continue
		}
		if got != b {
			t.Errorf("byte %d: encoded to %c, decoded to %d", i, r, got)
		}
	}

	// Printable ASCII should map to themselves.
	for b := byte(33); b <= 126; b++ {
		if enc[b] != rune(b) {
			t.Errorf("byte %d (%c): expected self-mapping, got U+%04X", b, b, enc[b])
		}
	}
}

func TestBPETokenizer_SentencePiece(t *testing.T) {
	// Build a SentencePiece-style tokenizer with ▁-prefixed tokens.
	// ▁ (U+2581) is a regular character that gets merged with following chars.
	vocab := map[string]int{
		"<unk>":      0,
		"<s>":       1,
		"</s>":      2,
		"<pad>":     3,
		"\u2581":       4, // ▁ as standalone character
		"W":         5,
		"h":         6,
		"a":         7,
		"t":         8,
		"i":         9,
		"s":         10,
		"2":         11,
		"+":         12,
		"?":         13,
		"\u2581W":      20,
		"\u2581Wh":     21,
		"\u2581Wha":    22,
		"\u2581What":   23,
		"\u2581i":      24,
		"\u2581is":     25,
		"\u25812":      26,
		"\u25812+":     27,
		"\u25812+2":    28,
		"\u25812+2?":   29,
	}
	merges := []MergePair{
		{Left: "\u2581", Right: "W"},         // rank 0: ▁+W -> ▁W
		{Left: "\u2581W", Right: "h"},        // rank 1: ▁W+h -> ▁Wh
		{Left: "\u2581Wh", Right: "a"},       // rank 2: ▁Wh+a -> ▁Wha
		{Left: "\u2581Wha", Right: "t"},      // rank 3: ▁Wha+t -> ▁What
		{Left: "\u2581", Right: "i"},         // rank 4: ▁+i -> ▁i
		{Left: "\u2581i", Right: "s"},        // rank 5: ▁i+s -> ▁is
		{Left: "\u2581", Right: "2"},         // rank 6: ▁+2 -> ▁2
		{Left: "\u25812", Right: "+"},        // rank 7: ▁2++ -> ▁2+
		{Left: "\u25812+", Right: "2"},       // rank 8: ▁2++2 -> ▁2+2
		{Left: "\u25812+2", Right: "?"},      // rank 9: ▁2+2+? -> ▁2+2?
	}

	special := SpecialTokens{BOS: 1, EOS: 2, PAD: 3, UNK: 0}
	tok := NewBPETokenizer(vocab, merges, special, false)
	tok.SetSentencePiece(true)

	t.Run("encode", func(t *testing.T) {
		ids, err := tok.Encode("What is 2+2?")
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}
		// "What is 2+2?" -> ["▁What", "▁is", "▁2+2?"]
		want := []int{23, 25, 29}
		if len(ids) != len(want) {
			t.Fatalf("Encode = %v (len=%d), want %v (len=%d)", ids, len(ids), want, len(want))
		}
		for i, id := range ids {
			if id != want[i] {
				t.Errorf("[%d] = %d, want %d", i, id, want[i])
			}
		}
	})

	t.Run("decode", func(t *testing.T) {
		decoded, err := tok.Decode([]int{23, 25, 29})
		if err != nil {
			t.Fatalf("Decode error: %v", err)
		}
		want := "What is 2+2?"
		if decoded != want {
			t.Errorf("Decode = %q, want %q", decoded, want)
		}
	})

	t.Run("pre-tokenize splits", func(t *testing.T) {
		words := tok.sentencePiecePreTokenize("hello world test", true)
		want := []string{"\u2581hello", "\u2581world", "\u2581test"}
		if len(words) != len(want) {
			t.Fatalf("got %v, want %v", words, want)
		}
		for i, w := range words {
			if w != want[i] {
				t.Errorf("[%d] = %q, want %q", i, w, want[i])
			}
		}
	})
}

func TestBPETokenizer_ByteLevelBPE(t *testing.T) {
	// Build a tiny byte-level BPE tokenizer.
	enc, _ := buildByteEncoderDecoder()

	// Map individual byte characters to IDs.
	vocab := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
	}
	nextID := 3
	for i := range 256 {
		s := string(enc[byte(i)])
		vocab[s] = nextID
		nextID++
	}

	// Add a merge for "h" + "i" = "hi"
	hChar := string(enc['h'])
	iChar := string(enc['i'])
	hiToken := hChar + iChar
	vocab[hiToken] = nextID

	merges := []MergePair{
		{Left: hChar, Right: iChar},
	}
	special := SpecialTokens{BOS: 1, EOS: 2, PAD: 0, UNK: 0}
	tok := NewBPETokenizer(vocab, merges, special, true)

	// Encode "hi" -- should produce the merged token.
	ids, err := tok.Encode("hi")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 || ids[0] != nextID {
		t.Errorf("Encode(\"hi\") = %v, want [%d]", ids, nextID)
	}

	// Decode should round-trip.
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != "hi" {
		t.Errorf("Decode(%v) = %q, want \"hi\"", ids, decoded)
	}
}

// makeTestSentencePieceUnigram creates a SentencePiece unigram tokenizer
// with vocabulary and scores but no merges, simulating Mistral 7B GGUF.
func makeTestSentencePieceUnigram() *BPETokenizer {
	vocab := map[string]int{
		"<unk>":    0,
		"<s>":     1,
		"</s>":    2,
		"\u2581":     3, // ▁
		"\u2581H":    4,
		"\u2581He":   5,
		"\u2581Hel":  6,
		"\u2581Hell": 7,
		"\u2581Hello": 8,
		"\u2581w":    9,
		"\u2581wo":   10,
		"\u2581wor":  11,
		"\u2581worl": 12,
		"\u2581world": 13,
		"H":       14,
		"e":       15,
		"l":       16,
		"o":       17,
		"w":       18,
		"r":       19,
		"d":       20,
		"\u2581the":  21,
		"\u2581is":   22,
		"\u2581a":    23,
		"\u2581test": 24,
		"t":       25,
		"s":       26,
	}

	// Scores: higher (less negative) = more likely. Longer tokens get better scores.
	scores := make([]float32, 27)
	scores[0] = -100   // <unk>
	scores[1] = -100   // <s>
	scores[2] = -100   // </s>
	scores[3] = -5.0   // ▁
	scores[4] = -3.0   // ▁H
	scores[5] = -2.5   // ▁He
	scores[6] = -2.0   // ▁Hel
	scores[7] = -1.5   // ▁Hell
	scores[8] = -1.0   // ▁Hello (best for "Hello")
	scores[9] = -3.0   // ▁w
	scores[10] = -2.5  // ▁wo
	scores[11] = -2.0  // ▁wor
	scores[12] = -1.5  // ▁worl
	scores[13] = -1.0  // ▁world (best for "world")
	scores[14] = -4.0  // H
	scores[15] = -4.0  // e
	scores[16] = -4.0  // l
	scores[17] = -4.0  // o
	scores[18] = -4.0  // w
	scores[19] = -4.0  // r
	scores[20] = -4.0  // d
	scores[21] = -1.0  // ▁the
	scores[22] = -1.0  // ▁is
	scores[23] = -1.5  // ▁a
	scores[24] = -1.0  // ▁test
	scores[25] = -4.0  // t
	scores[26] = -4.0  // s

	special := SpecialTokens{BOS: 1, EOS: 2, PAD: 0, UNK: 0}
	// No merges — this is a unigram model.
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSentencePiece(true)
	tok.SetScores(scores)
	return tok
}

func TestSentencePieceUnigram_Encode(t *testing.T) {
	tok := makeTestSentencePieceUnigram()

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{"single word", "Hello", []int{8}},                         // ▁Hello
		{"two words", "Hello world", []int{8, 13}},                 // ▁Hello ▁world
		{"sentence", "the world is a test", []int{21, 13, 22, 23, 24}}, // ▁the ▁world ▁is ▁a ▁test
		{"empty string", "", []int{}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			if len(ids) != len(tc.wantIDs) {
				t.Fatalf("Encode(%q) = %v (len=%d), want %v (len=%d)", tc.input, ids, len(ids), tc.wantIDs, len(tc.wantIDs))
			}
			for i, id := range ids {
				if id != tc.wantIDs[i] {
					t.Errorf("Encode(%q)[%d] = %d, want %d", tc.input, i, id, tc.wantIDs[i])
				}
			}
		})
	}
}

func TestSentencePieceUnigram_Decode(t *testing.T) {
	tok := makeTestSentencePieceUnigram()

	tests := []struct {
		name     string
		ids      []int
		wantText string
		wantErr  bool
	}{
		{"single token", []int{8}, "Hello", false},
		{"multiple tokens", []int{8, 13}, "Hello world", false},
		{"empty", []int{}, "", false},
		{"unknown ID", []int{999}, "", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tok.Decode(tc.ids)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("Decode(%v) expected error, got %q", tc.ids, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("Decode(%v) error: %v", tc.ids, err)
			}
			if got != tc.wantText {
				t.Errorf("Decode(%v) = %q, want %q", tc.ids, got, tc.wantText)
			}
		})
	}
}

func TestSentencePieceUnigram_RoundTrip(t *testing.T) {
	tok := makeTestSentencePieceUnigram()

	tests := []string{"Hello", "Hello world", "the world is a test"}
	for _, text := range tests {
		ids, err := tok.Encode(text)
		if err != nil {
			t.Fatalf("Encode(%q) error: %v", text, err)
		}
		decoded, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode(%v) error: %v", ids, err)
		}
		if decoded != text {
			t.Errorf("round-trip failed: %q -> %v -> %q", text, ids, decoded)
		}
	}
}

func TestSentencePieceUnigram_UnknownChars(t *testing.T) {
	tok := makeTestSentencePieceUnigram()

	// Characters not in vocab should produce UNK or ▁ tokens via Viterbi.
	// "xyz" -> pre-tokenized as "▁xyz". Since ▁x, ▁y, ▁z are not in vocab,
	// Viterbi will match ▁ first, then x, y, z via byte fallback or UNK.
	ids, err := tok.Encode("xyz")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("expected non-empty token list for 'xyz'")
	}
	for _, id := range ids {
		if id != tok.special.UNK && id != 3 {
			t.Errorf("expected UNK (0) or ▁ (3) token for unknown chars, got id=%d", id)
		}
	}
}

func TestSentencePieceUnigram_ViterbiOptimal(t *testing.T) {
	tok := makeTestSentencePieceUnigram()

	// "Hello" should encode as one token ▁Hello (id=8) via Viterbi,
	// since it has the best score (-1.0) vs splitting into subwords.
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 {
		t.Errorf("expected 1 token for 'Hello', got %d: %v", len(ids), ids)
	}
	if ids[0] != 8 {
		t.Errorf("expected token id 8 (▁Hello), got %d", ids[0])
	}
}

func TestSentencePieceUnigram_WithBPEFallback(t *testing.T) {
	// When merges ARE present, unigram encoding should NOT be used
	// even if scores are also set.
	tok := makeTestBPE()
	tok.SetScores([]float32{0, 0, 0, 0}) // set scores but merges exist
	ids, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	// Should still use BPE merging, producing "hello" (id=17).
	if len(ids) != 1 || ids[0] != 17 {
		t.Errorf("with merges present, expected BPE encoding [17], got %v", ids)
	}
}

func TestSentencePieceUnigram_ViterbiBeatsGreedy(t *testing.T) {
	// This test demonstrates that Viterbi finds a better segmentation than greedy.
	// Vocab has "▁hel" and "lo" with good scores, and "▁h" and "ello" with worse scores.
	// Greedy longest-match would pick "▁hello" if available, or "▁hell" + "o".
	// But here we set up scores so "▁hel" + "lo" is strictly better than "▁hell" + "o".
	vocab := map[string]int{
		"<unk>":     0,
		"<s>":      1,
		"</s>":     2,
		"\u2581":      3,
		"\u2581h":     4,
		"\u2581he":    5,
		"\u2581hel":   6,
		"\u2581hell":  7,
		"o":        8,
		"l":        9,
		"lo":       10,
	}
	scores := make([]float32, 11)
	scores[0] = -100   // <unk>
	scores[1] = -100   // <s>
	scores[2] = -100   // </s>
	scores[3] = -5.0   // ▁
	scores[4] = -4.0   // ▁h
	scores[5] = -3.0   // ▁he
	scores[6] = -1.5   // ▁hel  (good)
	scores[7] = -3.0   // ▁hell (worse than ▁hel)
	scores[8] = -3.0   // o     (bad)
	scores[9] = -4.0   // l     (bad)
	scores[10] = -1.0  // lo    (good)

	special := SpecialTokens{BOS: 1, EOS: 2, PAD: 0, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSentencePiece(true)
	tok.SetScores(scores)

	ids, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	// Viterbi should choose "▁hel" + "lo" (score -1.5 + -1.0 = -2.5)
	// over "▁hell" + "o" (score -3.0 + -3.0 = -6.0).
	want := []int{6, 10} // ▁hel, lo
	if len(ids) != len(want) {
		t.Fatalf("Encode(\"hello\") = %v, want %v", ids, want)
	}
	for i, id := range ids {
		if id != want[i] {
			t.Errorf("Encode(\"hello\")[%d] = %d, want %d", i, id, want[i])
		}
	}
}

// makeTestSentencePieceUnigramWithBytes creates a SentencePiece unigram
// tokenizer that includes <0xNN> byte fallback tokens.
func makeTestSentencePieceUnigramWithBytes() *BPETokenizer {
	vocab := map[string]int{
		"<unk>":     0,
		"<s>":      1,
		"</s>":     2,
		"\u2581":      3,
		"\u2581the":   4,
		"\u2581capital": 5,
		"\u2581of":    6,
		"\u2581France": 7,
		"\u2581is":    8,
		"\u2581Paris":  9,
		"\u2581a":     10,
		"a":        11,
		"t":        12,
		"h":        13,
		"e":        14,
	}
	// Add byte fallback tokens for all 256 bytes.
	nextID := 15
	for b := 0; b < 256; b++ {
		tok := fmt.Sprintf("<0x%02X>", b)
		vocab[tok] = nextID
		nextID++
	}

	scores := make([]float32, nextID)
	scores[0] = -100   // <unk>
	scores[1] = -100   // <s>
	scores[2] = -100   // </s>
	scores[3] = -5.0   // ▁
	scores[4] = -1.0   // ▁the
	scores[5] = -0.5   // ▁capital
	scores[6] = -1.0   // ▁of
	scores[7] = -0.5   // ▁France
	scores[8] = -1.0   // ▁is
	scores[9] = -0.5   // ▁Paris
	scores[10] = -2.0  // ▁a
	scores[11] = -4.0  // a
	scores[12] = -4.0  // t
	scores[13] = -4.0  // h
	scores[14] = -4.0  // e
	// Byte fallback tokens get very negative scores.
	for i := 15; i < nextID; i++ {
		scores[i] = -10.0
	}

	special := SpecialTokens{BOS: 1, EOS: 2, PAD: 0, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSentencePiece(true)
	tok.SetScores(scores)
	return tok
}

func TestSentencePieceUnigram_EncodeDecodeSentence(t *testing.T) {
	tok := makeTestSentencePieceUnigramWithBytes()

	text := "The capital of France is Paris"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode(%q) error: %v", text, err)
	}
	if len(ids) == 0 {
		t.Fatalf("Encode(%q) produced empty result", text)
	}
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != text {
		t.Errorf("round-trip failed: %q -> %v -> %q", text, ids, decoded)
	}
}

func TestSentencePieceUnigram_ByteFallback(t *testing.T) {
	tok := makeTestSentencePieceUnigramWithBytes()

	// Encode a string with characters not directly in vocab.
	// The emoji will require byte fallback via <0xNN> tokens.
	text := "the \xc3\xa9"  // "the é" — é is 0xC3 0xA9 in UTF-8
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode(%q) error: %v", text, err)
	}
	if len(ids) == 0 {
		t.Fatalf("Encode(%q) produced empty result", text)
	}
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != text {
		t.Errorf("byte fallback round-trip: %q -> %v -> %q", text, ids, decoded)
	}
}

func TestSentencePieceUnigram_EmptyAndSingle(t *testing.T) {
	tok := makeTestSentencePieceUnigram()

	// Empty string.
	ids, err := tok.Encode("")
	if err != nil {
		t.Fatalf("Encode empty error: %v", err)
	}
	if len(ids) != 0 {
		t.Errorf("Encode(\"\") = %v, want []", ids)
	}

	// Single character that exists in vocab.
	ids, err = tok.Encode("a")
	if err != nil {
		t.Fatalf("Encode(\"a\") error: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("Encode(\"a\") produced empty result")
	}
}

func TestSentencePieceUnigram_LongText(t *testing.T) {
	tok := makeTestSentencePieceUnigram()

	// Encode and decode a longer text with repeated words.
	text := "the test is a test the test is a test"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != text {
		t.Errorf("round-trip failed: %q -> %v -> %q", text, ids, decoded)
	}
}

func TestSentencePieceUnigram_ByteFallbackNeverBeatsVocab(t *testing.T) {
	// Regression test: byte fallback tokens must never be preferred over
	// multi-character vocab tokens, even when byte token scores are higher.
	// This was the original bug — byte tokens like <0xE2> had scores of 0.0
	// which beat multi-character tokens with negative scores, producing 43
	// byte-level tokens instead of 7 word tokens.
	vocab := map[string]int{
		"<unk>":        0,
		"<s>":         1,
		"</s>":        2,
		"\u2581":         3,
		"\u2581What":     4,
		"\u2581is":       5,
		"\u2581the":      6,
		"\u2581capital":  7,
		"\u2581of":       8,
		"\u2581France":   9,
		"?":           10,
	}
	// Add byte fallback tokens for all 256 bytes.
	nextID := 11
	for b := 0; b < 256; b++ {
		tok := fmt.Sprintf("<0x%02X>", b)
		vocab[tok] = nextID
		nextID++
	}

	scores := make([]float32, nextID)
	scores[0] = -100  // <unk>
	scores[1] = -100  // <s>
	scores[2] = -100  // </s>
	scores[3] = -5.0  // ▁
	scores[4] = -8.0  // ▁What
	scores[5] = -7.0  // ▁is
	scores[6] = -6.0  // ▁the
	scores[7] = -9.0  // ▁capital
	scores[8] = -6.0  // ▁of
	scores[9] = -9.0  // ▁France
	scores[10] = -4.0 // ?
	// Byte fallback tokens get HIGH scores (the bug scenario).
	// Before the fix, these would win over multi-character vocab tokens.
	for i := 11; i < nextID; i++ {
		scores[i] = 0.0
	}

	special := SpecialTokens{BOS: 1, EOS: 2, PAD: 0, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSentencePiece(true)
	tok.SetScores(scores)

	ids, err := tok.Encode("What is the capital of France?")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	// Must produce word-level tokens, not byte-level tokens.
	// "What is the capital of France?" -> [▁What, ▁is, ▁the, ▁capital, ▁of, ▁France, ?]
	want := []int{4, 5, 6, 7, 8, 9, 10}
	if len(ids) != len(want) {
		t.Fatalf("Encode produced %d tokens %v, want %d tokens %v", len(ids), ids, len(want), want)
	}
	for i, id := range ids {
		if id != want[i] {
			t.Errorf("[%d] = %d, want %d", i, id, want[i])
		}
	}

	// Verify round-trip.
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != "What is the capital of France?" {
		t.Errorf("Decode = %q, want %q", decoded, "What is the capital of France?")
	}
}

func TestSentencePieceUnigram_ByteFallbackStillWorksForUnknownChars(t *testing.T) {
	// Byte fallback must still be used for characters that have no
	// vocabulary coverage (e.g., emoji, rare Unicode).
	vocab := map[string]int{
		"<unk>":    0,
		"<s>":     1,
		"</s>":    2,
		"\u2581":     3,
		"\u2581hi":   4,
	}
	nextID := 5
	for b := 0; b < 256; b++ {
		tok := fmt.Sprintf("<0x%02X>", b)
		vocab[tok] = nextID
		nextID++
	}

	scores := make([]float32, nextID)
	scores[0] = -100
	scores[1] = -100
	scores[2] = -100
	scores[3] = -5.0
	scores[4] = -1.0 // ▁hi
	for i := 5; i < nextID; i++ {
		scores[i] = -2.0 // byte scores
	}

	special := SpecialTokens{BOS: 1, EOS: 2, PAD: 0, UNK: 0}
	tok := NewBPETokenizer(vocab, nil, special, false)
	tok.SetSentencePiece(true)
	tok.SetScores(scores)

	// "hi" has a vocab token; should use it.
	ids, err := tok.Encode("hi")
	if err != nil {
		t.Fatalf("Encode(\"hi\") error: %v", err)
	}
	if len(ids) != 1 || ids[0] != 4 {
		t.Errorf("Encode(\"hi\") = %v, want [4] (▁hi)", ids)
	}

	// "hi\xc3\xa9" — é (U+00E9) is not in vocab, must use byte fallback.
	ids, err = tok.Encode("hi\xc3\xa9")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	// Should be: ▁hi + <0xC3> + <0xA9>
	if len(ids) != 3 {
		t.Fatalf("Encode(\"hi\\xc3\\xa9\") = %v (len=%d), want 3 tokens", ids, len(ids))
	}
	if ids[0] != 4 {
		t.Errorf("[0] = %d, want 4 (▁hi)", ids[0])
	}
	// Verify round-trip through decode.
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != "hi\xc3\xa9" {
		t.Errorf("Decode = %q, want %q", decoded, "hi\xc3\xa9")
	}
}

func TestDecodeSentencePieceBytes(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"no byte tokens", "hello", "hello"},
		{"single byte", "<0x41>", "A"},
		{"multiple bytes", "<0xC3><0xA9>", "\xc3\xa9"}, // é
		{"mixed", "hello<0x21>world", "hello!world"},
		{"invalid hex preserved", "<0xZZ>", "<0xZZ>"},
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
