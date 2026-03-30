# ADR-001: API Stability Contract for ztoken v1.0.0

**Status:** Accepted
**Date:** 2026-03-29

## Context

The `ztoken` package (`github.com/zerfoo/ztoken`) provides BPE and WordPiece tokenization for ML inference in Go. The public API has been stable across multiple releases and is consumed by `zerfoo` and downstream applications. We need to formalize which symbols are part of the stable v1 contract and which are internal implementation details.

## Decision

### Stable v1 Public API

The following exported symbols constitute the stable v1 API. They will not have breaking changes within the v1.x release series, following Go module compatibility guarantees.

#### Root package (`github.com/zerfoo/ztoken`)

**Interface:**

- `Tokenizer` -- the core abstraction; all tokenizer implementations satisfy this interface
  - `Encode(text string) ([]int, error)`
  - `Decode(ids []int) (string, error)`
  - `VocabSize() int`
  - `GetToken(id int) (string, bool)`
  - `GetID(token string) (int, bool)`
  - `SpecialTokens() SpecialTokens`

**Types:**

- `SpecialTokens` -- struct with BOS, EOS, PAD, UNK fields
- `MergePair` -- struct with Left, Right fields
- `NormalizerFunc` -- function type `func(string) string`
- `BERTEncoding` -- struct with InputIDs, AttentionMask, TokenTypeIDs fields

**Concrete tokenizers:**

- `BPETokenizer` (struct, implements `Tokenizer`)
  - `NewBPETokenizer(vocab, merges, special, byteLevelBPE)`
  - `Encode`, `Decode`, `VocabSize`, `GetToken`, `GetID`, `SpecialTokens`
  - `EncodeWithSpecialTokens(text, addBOS, addEOS)`
  - `SetScores(scores)`
  - `SetSentencePiece(enabled)`
  - `SetAddLeadingSpace(enabled)`
  - `SetSpecialTokenStrings(tokens)`

- `WhitespaceTokenizer` (struct, implements `Tokenizer`)
  - `NewWhitespaceTokenizer()`
  - `Encode`, `Decode`, `VocabSize`, `GetToken`, `GetID`, `SpecialTokens`
  - `AddToken(token)`

- `WordPieceTokenizer` (struct, implements `Tokenizer`)
  - `NewWordPieceTokenizer(vocab, special)`
  - `Encode`, `Decode`, `VocabSize`, `GetToken`, `GetID`, `SpecialTokens`
  - `EncodeForBERT(textA, textB, maxLen)`

**Loader functions:**

- `Load(path string) (Tokenizer, error)` -- loads HuggingFace tokenizer.json, returns appropriate implementation
- `LoadFromJSON(path string) (*BPETokenizer, error)` -- loads HuggingFace tokenizer.json as BPE specifically

#### Subpackage `gguf` (`github.com/zerfoo/ztoken/gguf`)

- `Metadata` -- interface for GGUF key-value access
- `ExtractTokenizer(m Metadata) (*ztoken.BPETokenizer, error)` -- builds a BPETokenizer from GGUF metadata

### Not Public API

The following are explicitly **not** part of the v1 stability contract:

- **Unexported fields and methods** on all types
- **Internal implementation details** of BPE merge algorithms, SentencePiece encoding, and WordPiece subword matching
- **Test utilities and test data** under `testdata/`
- **File format parsing internals** within the `gguf` subpackage (only the `Metadata` interface and `ExtractTokenizer` function are stable)

### Compatibility Rules

1. No exported type, function, or method listed above will be removed or have its signature changed in v1.x.
2. New methods may be added to concrete types (`BPETokenizer`, `WhitespaceTokenizer`, `WordPieceTokenizer`).
3. New fields may be added to `SpecialTokens`, `MergePair`, `BERTEncoding`, and other struct types.
4. The `Tokenizer` interface will not gain new methods in v1.x -- that would break external implementations.
5. New subpackages or new exported functions may be added.
6. Bug fixes that change tokenization output to match the reference implementation (HuggingFace, llama.cpp) are permitted.

## Consequences

- Downstream consumers can depend on `github.com/zerfoo/ztoken` v1.x with confidence that upgrades will not break their code.
- The `Tokenizer` interface is frozen for v1 -- any new capabilities must be added via separate interfaces or concrete type methods.
- Internal refactoring (merge algorithm, byte-level BPE internals, GGUF parsing helpers) can proceed freely without versioning concerns.
