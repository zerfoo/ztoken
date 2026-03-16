# ztoken

BPE tokenizer library for Go with HuggingFace compatibility.

Part of the [Zerfoo](https://github.com/zerfoo) ML ecosystem.

## Install

```sh
go get github.com/zerfoo/ztoken
```

## Features

- Byte-Pair Encoding (BPE) tokenizer with HuggingFace tokenizer.json support
- SentencePiece compatibility mode
- Special token handling (BOS, EOS, PAD, UNK)
- GGUF tokenizer extraction via `ztoken/gguf` sub-package
- Zero external dependencies (stdlib only, plus golang.org/x/text)

## Quick start

```go
package main

import (
	"fmt"

	"github.com/zerfoo/ztoken"
)

func main() {
	tok, err := ztoken.LoadFromJSON("tokenizer.json")
	if err != nil {
		panic(err)
	}
	ids, _ := tok.Encode("Hello, world!")
	fmt.Println(ids)
	text, _ := tok.Decode(ids)
	fmt.Println(text)
}
```

## License

Apache 2.0
