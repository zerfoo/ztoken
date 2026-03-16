# ztoken

BPE tokenizer for Go with HuggingFace compatibility. Part of the [Zerfoo](https://github.com/zerfoo) ecosystem.

## Install

```sh
go get github.com/zerfoo/ztoken
```

## Features

- **Byte-Pair Encoding (BPE)** -- production-grade tokenizer with merge-based subword splitting
- **SentencePiece support** -- handles SentencePiece-style pre-tokenization with `▁` boundaries
- **Special tokens** -- automatic detection and handling of BOS, EOS, PAD, and UNK tokens
- **GGUF extraction** -- extract tokenizer vocabulary and merges from GGUF model metadata
- **HuggingFace loader** -- load tokenizers directly from `tokenizer.json` files

## Quick Start

```go
package main

import (
	"fmt"
	"log"

	"github.com/zerfoo/ztoken"
)

func main() {
	tok, err := ztoken.LoadFromJSON("tokenizer.json")
	if err != nil {
		log.Fatal(err)
	}

	ids, err := tok.Encode("Hello, world!")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Token IDs:", ids)

	text, err := tok.Decode(ids)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Decoded:", text)
}
```

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.
