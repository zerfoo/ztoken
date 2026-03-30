# ztoken Design Document

**Module:** `github.com/zerfoo/ztoken`
**Version:** v1.0.0
**Status:** Stable

## Overview

ztoken is a pure-Go tokenizer library for ML model inference. It provides BPE (byte-pair encoding), SentencePiece unigram, and WordPiece tokenization with two loading paths: HuggingFace `tokenizer.json` and GGUF metadata extraction. The library has a single external dependency (`golang.org/x/text` for Unicode normalization) and zero CGo.

## Architecture

```
ztoken/
  tokenizer.go      Tokenizer interface + WhitespaceTokenizer
  bpe.go            BPETokenizer (BPE merges, SentencePiece unigram, byte-level BPE)
  wordpiece.go      WordPieceTokenizer (BERT-family models)
  loader.go         HuggingFace tokenizer.json loader
  gguf/gguf.go      GGUF metadata tokenizer extraction
```

### Tokenizer Interface

All tokenizer implementations satisfy a single interface:

```go
type Tokenizer interface {
    Encode(text string) ([]int, error)
    Decode(ids []int) (string, error)
    VocabSize() int
    GetToken(id int) (string, bool)
    GetID(token string) (int, bool)
    SpecialTokens() SpecialTokens
}
```

`SpecialTokens` holds integer IDs for BOS, EOS, PAD, and UNK. All implementations populate these from the loaded model data.

### Encode/Decode Contract

- **Encode** accepts arbitrary UTF-8 text and returns a slice of integer token IDs. An empty string returns an empty slice with no error. Text normalization (NFC, NFD, lowercase, strip) is applied first when configured.
- **Decode** accepts a slice of token IDs and returns the reconstructed UTF-8 string. Unknown IDs return an error. Special tokens are stripped during WordPiece decoding.
- **Round-trip fidelity**: `Decode(Encode(text))` reproduces the original text modulo normalization and leading-space behavior inherent to each algorithm.

## BPE Tokenizer

`BPETokenizer` is the primary production tokenizer. It supports three encoding modes selected by configuration:

### 1. Standard BPE (merge-based)

The classic byte-pair encoding algorithm. Text is pre-tokenized into words, each word is split into characters, and adjacent pairs are iteratively merged in priority order defined by the merge table.

**Encoding steps:**
1. Apply normalizer (if configured)
2. Split around registered special tokens (exact string match, longest wins)
3. Pre-tokenize into words (whitespace split or byte-level or SentencePiece, depending on mode)
4. For each word, split into characters and iteratively merge the highest-rank adjacent pair until no more merges apply
5. Map final subword strings to vocabulary IDs (unmapped subwords become UNK)

### 2. Byte-level BPE (GPT-2 style)

Enabled when `byteLevelBPE` is true. Every byte of the UTF-8 input is mapped to a printable Unicode character using the GPT-2 byte encoder table (printable ASCII maps to itself; other bytes map to codepoints starting at U+0100). This ensures all inputs are representable without an UNK token. Decode reverses the mapping.

**Pre-tokenization:** whitespace becomes a prefix of the following word token, preserving space information in the token stream.

### 3. SentencePiece unigram (score-based)

Activated when `sentencePiece` is true and the merge table is empty but token scores are set. Spaces are replaced with the metaspace character (U+2581). Encoding uses greedy leftmost-longest match: at each position, the longest vocabulary token is selected, with ties broken by score (negative log probability). Unmatched bytes fall back to `<0xNN>` byte tokens.

This mode matches llama.cpp's `llm_tokenizer_spm::tokenize` behavior and is used by Llama, Gemma, and other GGUF models with `tokenizer.ggml.model = "llama"`.

**Leading space:** by default, SentencePiece mode prepends U+2581 to the first word. This can be overridden via `SetAddLeadingSpace(false)`, which GGUF models control through the `tokenizer.ggml.add_space_prefix` metadata key.

### Special Token Handling

Special tokens (e.g., `<start_of_turn>`, `<end_of_turn>`) are registered via `SetSpecialTokenStrings`. During encoding, the input is scanned for these strings before BPE/unigram processing. Each match emits its pre-assigned ID as a single token, preventing BPE from splitting control sequences into characters.

## WordPiece Tokenizer

`WordPieceTokenizer` implements the subword algorithm used by BERT-family models. Text is pre-tokenized by splitting on whitespace and punctuation boundaries, then each word is greedily matched against the vocabulary:

1. Try the full word
2. If not found, find the longest prefix in the vocabulary
3. Continue with the remainder, prefixed by `##` (continuing subword marker)
4. If no subword match is found at any position, the entire word maps to UNK

### BERT Encoding

`EncodeForBERT` produces the standard BERT input format:
- **Single sentence:** `[CLS] tokens [SEP]`
- **Sentence pair:** `[CLS] tokens_a [SEP] tokens_b [SEP]`
- Returns `BERTEncoding` with `InputIDs`, `AttentionMask`, and `TokenTypeIDs`
- Optional padding to `maxLen`

## HuggingFace Compatibility Layer

The `Load` and `LoadFromJSON` functions parse HuggingFace `tokenizer.json` files. This is the standard format exported by the `transformers` library and hosted on the HuggingFace Hub.

### Supported Schema

| JSON field | Purpose |
|------------|---------|
| `model.type` | Selects algorithm: `"BPE"` (or empty) routes to `BPETokenizer`, `"WordPiece"` routes to `WordPieceTokenizer` |
| `model.vocab` | Token-to-ID mapping |
| `model.merges` | Merge rules in either `["a b", ...]` or `[["a","b"], ...]` format |
| `added_tokens` | Special tokens with IDs and `special` flag |
| `pre_tokenizer` | Pre-tokenizer config; `ByteLevel` type enables byte-level BPE |
| `normalizer` | Text normalization chain (NFC, NFD, Lowercase, Strip, Sequence) |
| `decoder` | Decoder config; `Metaspace` or `Replace` with U+2581 enables SentencePiece mode |

### Auto-detection

The loader auto-detects the tokenization mode from the JSON structure:
- **Byte-level BPE:** detected when `pre_tokenizer` contains a `ByteLevel` entry (direct or inside a `Sequence`)
- **SentencePiece:** detected when `decoder` contains a `Metaspace` entry or a `Replace` rule targeting U+2581
- **WordPiece:** detected when `model.type` is `"WordPiece"`

### Merge Format Compatibility

Merges accept both the standard space-separated string format (`"a b"`) used by most models and the two-element array format (`["a", "b"]`) used by Gemma 3 tokenizers.

### Special Token Extraction

`extractSpecialTokens` maps `added_tokens` entries to BOS/EOS/PAD/UNK using both GPT-style names (`<s>`, `</s>`, `<pad>`, `<unk>`) and BERT-style names (`[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`), plus substring matching on `bos`, `eos`, `pad`, `unk`.

## GGUF Tokenizer Loading

The `gguf` sub-package extracts tokenizer data from GGUF file metadata without depending on a specific GGUF parser. It defines a `Metadata` interface:

```go
type Metadata interface {
    GetString(key string) (string, bool)
    GetStringArray(key string) ([]string, bool)
    GetUint32(key string) (uint32, bool)
    GetInt32Array(key string) ([]int32, bool)
    GetFloat32Array(key string) ([]float32, bool)
}
```

`ExtractTokenizer(m Metadata)` reads the following GGUF keys:

| Key | Required | Purpose |
|-----|----------|---------|
| `tokenizer.ggml.tokens` | Yes | Token vocabulary (string array) |
| `tokenizer.ggml.merges` | No | BPE merge rules (space-separated strings) |
| `tokenizer.ggml.scores` | No | SentencePiece unigram scores (float32 array) |
| `tokenizer.ggml.bos_token_id` | No | Beginning-of-sequence token ID |
| `tokenizer.ggml.eos_token_id` | No | End-of-sequence token ID |
| `tokenizer.ggml.unknown_token_id` | No | Unknown token ID |
| `tokenizer.ggml.padding_token_id` | No | Padding token ID |
| `tokenizer.ggml.model` | No | Model type; `"llama"` enables SentencePiece mode |
| `tokenizer.ggml.token_type` | No | Per-token type array; type 3 = control/special token |

### Mode Selection

- If `tokenizer.ggml.model` is `"llama"`, SentencePiece pre-tokenization is enabled
- If scores are present but merges are absent, the tokenizer uses greedy unigram encoding
- If merges are present, standard BPE merge encoding is used
- Control tokens (type 3) are registered for exact-match during encoding

### Interface Decoupling

The `Metadata` interface decouples ztoken from any specific GGUF parser. In the Zerfoo ecosystem, `zerfoo/model` implements this interface over its GGUF reader, but any implementation satisfying the five-method interface works.

## Text Normalization

Normalizers are optional functions applied before tokenization. The HuggingFace loader builds them from JSON config:

| Type | Behavior |
|------|----------|
| `NFC` | Unicode NFC normalization |
| `NFD` | Unicode NFD normalization |
| `Lowercase` | Case folding |
| `Strip` | Trim leading/trailing whitespace |
| `Sequence` | Chain of normalizers applied in order |

Both `BPETokenizer` and `WordPieceTokenizer` accept a `NormalizerFunc` internally. GGUF-loaded tokenizers do not currently carry normalizer configuration (normalization is handled at the model level).
