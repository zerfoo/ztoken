package ztoken

import (
	"fmt"
	"strings"
	"unicode"
)

// WordPieceTokenizer implements the Tokenizer interface using the WordPiece
// algorithm, as used by BERT-family models. It greedily matches the longest
// subword prefix from the vocabulary, using "##" to denote continuation tokens.
//
// Stable.
type WordPieceTokenizer struct {
	vocab        map[string]int
	reverseVocab map[int]string
	special      SpecialTokens
	normalizer   NormalizerFunc
	// maxTokenLen is the length of the longest token in the vocabulary,
	// used to bound the greedy prefix search.
	maxTokenLen int
	// unkToken is the string representation of the unknown token.
	unkToken string
	// specialTokens maps special token strings to IDs for exact matching.
	specialTokens map[string]int
}

// BERTEncoding holds the input tensors expected by BERT-family models.
type BERTEncoding struct {
	InputIDs     []int // Token IDs: [CLS] + tokens + [SEP] (+ tokens + [SEP] for pairs)
	AttentionMask []int // 1 for real tokens, 0 for padding
	TokenTypeIDs []int // 0 for first sentence, 1 for second sentence
}

// NewWordPieceTokenizer creates a WordPieceTokenizer from a vocabulary and special tokens.
func NewWordPieceTokenizer(vocab map[string]int, special SpecialTokens) *WordPieceTokenizer {
	reverseVocab := make(map[int]string, len(vocab))
	maxLen := 0
	for k, v := range vocab {
		reverseVocab[v] = k
		if len(k) > maxLen {
			maxLen = len(k)
		}
	}

	unkToken := "[UNK]"
	if tok, ok := reverseVocab[special.UNK]; ok {
		unkToken = tok
	}

	return &WordPieceTokenizer{
		vocab:        vocab,
		reverseVocab: reverseVocab,
		special:      special,
		maxTokenLen:  maxLen,
		unkToken:     unkToken,
	}
}

// Encode tokenizes text into a sequence of token IDs using WordPiece.
func (t *WordPieceTokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return []int{}, nil
	}
	if t.normalizer != nil {
		text = t.normalizer(text)
	}
	words := preTokenize(text)
	var ids []int
	for _, word := range words {
		wordIDs := t.tokenizeWord(word)
		ids = append(ids, wordIDs...)
	}
	return ids, nil
}

// Decode converts token IDs back to text. Continuation tokens (##prefix) are
// joined without spaces to reconstruct words.
func (t *WordPieceTokenizer) Decode(ids []int) (string, error) {
	var sb strings.Builder
	for i, id := range ids {
		tok, ok := t.reverseVocab[id]
		if !ok {
			return "", fmt.Errorf("unknown token ID: %d", id)
		}
		// Skip special tokens in decode output.
		if t.isSpecialToken(tok) {
			continue
		}
		if strings.HasPrefix(tok, "##") {
			sb.WriteString(tok[2:])
		} else {
			if i > 0 && sb.Len() > 0 {
				sb.WriteByte(' ')
			}
			sb.WriteString(tok)
		}
	}
	return sb.String(), nil
}

// VocabSize returns the number of tokens in the vocabulary.
func (t *WordPieceTokenizer) VocabSize() int {
	return len(t.vocab)
}

// GetToken returns the string token for a given ID.
func (t *WordPieceTokenizer) GetToken(id int) (string, bool) {
	tok, ok := t.reverseVocab[id]
	return tok, ok
}

// GetID returns the token ID for a given string.
func (t *WordPieceTokenizer) GetID(token string) (int, bool) {
	id, ok := t.vocab[token]
	return id, ok
}

// SpecialTokens returns the special token IDs.
func (t *WordPieceTokenizer) SpecialTokens() SpecialTokens {
	return t.special
}

// EncodeForBERT tokenizes one or two sentences into the BERT input format.
// For a single sentence: [CLS] tokens [SEP]
// For a sentence pair: [CLS] tokens_a [SEP] tokens_b [SEP]
// The result is padded to maxLen if maxLen > 0.
func (t *WordPieceTokenizer) EncodeForBERT(textA string, textB string, maxLen int) (*BERTEncoding, error) {
	idsA, err := t.Encode(textA)
	if err != nil {
		return nil, fmt.Errorf("encode text_a: %w", err)
	}

	clsID, ok := t.vocab["[CLS]"]
	if !ok {
		return nil, fmt.Errorf("vocabulary missing [CLS] token")
	}
	sepID, ok := t.vocab["[SEP]"]
	if !ok {
		return nil, fmt.Errorf("vocabulary missing [SEP] token")
	}

	// Build input_ids: [CLS] + tokens_a + [SEP]
	inputIDs := make([]int, 0, len(idsA)+3)
	inputIDs = append(inputIDs, clsID)
	inputIDs = append(inputIDs, idsA...)
	inputIDs = append(inputIDs, sepID)

	// token_type_ids: 0 for first sentence segment
	tokenTypeIDs := make([]int, len(inputIDs))

	if textB != "" {
		idsB, err := t.Encode(textB)
		if err != nil {
			return nil, fmt.Errorf("encode text_b: %w", err)
		}
		secondStart := len(inputIDs)
		inputIDs = append(inputIDs, idsB...)
		inputIDs = append(inputIDs, sepID)
		// Extend token_type_ids: 1 for second sentence segment
		for range len(idsB) + 1 {
			tokenTypeIDs = append(tokenTypeIDs, 1)
		}
		_ = secondStart
	}

	seqLen := len(inputIDs)

	// Pad if maxLen specified.
	if maxLen > 0 && seqLen < maxLen {
		padCount := maxLen - seqLen
		for range padCount {
			inputIDs = append(inputIDs, t.special.PAD)
			tokenTypeIDs = append(tokenTypeIDs, 0)
		}
	}

	// Build attention_mask: 1 for real tokens, 0 for padding.
	attentionMask := make([]int, len(inputIDs))
	for i := range seqLen {
		attentionMask[i] = 1
	}

	return &BERTEncoding{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
		TokenTypeIDs:  tokenTypeIDs,
	}, nil
}

// tokenizeWord applies the WordPiece algorithm to a single pre-tokenized word.
// It greedily matches the longest prefix in the vocabulary, continuing with
// "##"-prefixed subwords for the remainder.
func (t *WordPieceTokenizer) tokenizeWord(word string) []int {
	if _, ok := t.vocab[word]; ok {
		return []int{t.vocab[word]}
	}

	var ids []int
	start := 0
	runes := []rune(word)

	for start < len(runes) {
		end := len(runes)
		if end-start > t.maxTokenLen {
			end = start + t.maxTokenLen
		}
		matched := false
		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				substr = "##" + substr
			}
			if id, ok := t.vocab[substr]; ok {
				ids = append(ids, id)
				start = end
				matched = true
				break
			}
			end--
		}
		if !matched {
			// No subword match found — entire remaining word is UNK.
			return []int{t.special.UNK}
		}
	}
	return ids
}

// isSpecialToken returns true if the token string is a known special token.
func (t *WordPieceTokenizer) isSpecialToken(tok string) bool {
	switch tok {
	case "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]":
		return true
	}
	if _, ok := t.specialTokens[tok]; ok {
		return true
	}
	return false
}

// preTokenize splits text on whitespace and punctuation boundaries,
// producing individual words and punctuation characters as separate tokens.
func preTokenize(text string) []string {
	var tokens []string
	var current strings.Builder
	for _, r := range text {
		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			continue
		}
		if unicode.IsPunct(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
			continue
		}
		current.WriteRune(r)
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

// Statically assert WordPieceTokenizer implements Tokenizer.
var _ Tokenizer = (*WordPieceTokenizer)(nil)
