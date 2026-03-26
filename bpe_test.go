package ztoken

import (
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

	// Characters not in vocab should produce UNK tokens.
	ids, err := tok.Encode("xyz")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	// "xyz" -> pre-tokenized as "▁xyz". Since ▁x, ▁y, ▁z are not in vocab,
	// the greedy matcher will match ▁ first, then x, y, z individually → all UNK.
	for _, id := range ids {
		if id != tok.special.UNK {
			// Either ▁ (id=3) or UNK (id=0) are acceptable since ▁ is in vocab.
			if id != 3 {
				t.Errorf("expected UNK or ▁ token for unknown chars, got id=%d", id)
			}
		}
	}
}

func TestSentencePieceUnigram_PrefersLongestMatch(t *testing.T) {
	tok := makeTestSentencePieceUnigram()

	// "Hello" should encode as one token ▁Hello (id=8), not ▁H + e + l + l + o.
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
