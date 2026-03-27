# ztoken

[![CI](https://github.com/zerfoo/ztoken/actions/workflows/ci.yml/badge.svg)](https://github.com/zerfoo/ztoken/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/zerfoo/ztoken.svg)](https://pkg.go.dev/github.com/zerfoo/ztoken)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

BPE tokenizer library for Go with HuggingFace compatibility.

Part of the [Zerfoo](https://github.com/zerfoo) ML ecosystem.

## Features

- **Byte-Pair Encoding (BPE)** tokenizer with full merge-based encoding/decoding
- **HuggingFace `tokenizer.json`** loading — compatible with GPT-2, Llama, Gemma, Mistral, and other models
- **GGUF tokenizer extraction** — extract tokenizer data directly from GGUF model files via `ztoken/gguf`
- **SentencePiece compatibility** — handles U+2581 space markers used by Llama-family models
- **Special token handling** — BOS, EOS, PAD, UNK with exact-match encoding for control tokens
- **Byte-level BPE** — GPT-2 style byte-to-Unicode encoding for full UTF-8 coverage
- **Text normalization** — configurable normalizer pipeline (NFC, NFD, NFKC, lowercase, etc.)
- **Zero external dependencies** — stdlib only, plus `golang.org/x/text` for Unicode normalization

## Installation

```bash
go get github.com/zerfoo/ztoken
```

## Quick Start

### Load from HuggingFace tokenizer.json

```go
package main

import (
    "fmt"

    "github.com/zerfoo/ztoken"
)

func main() {
    // Load a HuggingFace tokenizer.json file
    tok, err := ztoken.LoadFromJSON("tokenizer.json")
    if err != nil {
        panic(err)
    }

    // Encode text to token IDs
    ids, _ := tok.Encode("Hello, world!")
    fmt.Println(ids)

    // Decode token IDs back to text
    text, _ := tok.Decode(ids)
    fmt.Println(text) // Hello, world!

    // Inspect vocabulary
    fmt.Println(tok.VocabSize())

    // Access special tokens
    special := tok.SpecialTokens()
    fmt.Printf("BOS=%d EOS=%d PAD=%d UNK=%d\n",
        special.BOS, special.EOS, special.PAD, special.UNK)
}
```

### Extract Tokenizer from GGUF Model Files

The `ztoken/gguf` sub-package extracts tokenizer data directly from GGUF model files, so you don't need a separate `tokenizer.json`:

```go
package main

import (
    "fmt"

    "github.com/zerfoo/ztoken/gguf"
)

func main() {
    // metadata is any type implementing gguf.Metadata interface:
    //   GetString(key string) (string, bool)
    //   GetStringArray(key string) ([]string, bool)
    //   GetUint32(key string) (uint32, bool)
    //   GetInt32Array(key string) ([]int32, bool)
    tok, err := gguf.ExtractTokenizer(metadata)
    if err != nil {
        panic(err)
    }

    ids, _ := tok.Encode("Hello from GGUF!")
    fmt.Println(ids)
}
```

### Build a Tokenizer Programmatically

```go
package main

import (
    "fmt"

    "github.com/zerfoo/ztoken"
)

func main() {
    vocab := map[string]int{
        "hello": 0, "world": 1, " ": 2,
        "<unk>": 3, "<s>": 4, "</s>": 5, "<pad>": 6,
    }
    merges := []ztoken.MergePair{
        {Left: "hel", Right: "lo"},
        {Left: "wor", Right: "ld"},
    }
    special := ztoken.SpecialTokens{BOS: 4, EOS: 5, PAD: 6, UNK: 3}

    tok := ztoken.NewBPETokenizer(vocab, merges, special, false)
    ids, _ := tok.Encode("hello")
    fmt.Println(ids) // [0]
}
```

## SentencePiece Compatibility

Models using SentencePiece tokenization (Llama, Gemma) encode spaces as the U+2581 character. ztoken handles this automatically when loading from GGUF files with `tokenizer.ggml.model = "llama"`, or you can enable it manually:

```go
tok := ztoken.NewBPETokenizer(vocab, merges, special, false)
tok.SetSentencePiece(true)
```

## Use Cases

- **ML inference preprocessing** — tokenize prompts before feeding them to transformer models via [zerfoo](https://github.com/zerfoo/zerfoo)
- **Text processing pipelines** — encode/decode text with production-grade BPE
- **Model tooling** — extract and inspect tokenizers from GGUF and HuggingFace model files
- **Embedding in Go services** — zero-CGo tokenization that compiles with `go build`

## Package Structure

| Package | Description |
|---------|-------------|
| `ztoken` | Core tokenizer interface, BPE implementation, HuggingFace JSON loader |
| `ztoken/gguf` | GGUF metadata-based tokenizer extraction |

## Dependencies

ztoken has zero external dependencies beyond the Go standard library and `golang.org/x/text` for Unicode normalization.

ztoken is used by:

- [zerfoo](https://github.com/zerfoo/zerfoo) — ML inference, training, and serving framework

## License

Apache 2.0
