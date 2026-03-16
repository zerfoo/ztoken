# Contributing to ztoken

Thank you for your interest in contributing to ztoken, the BPE tokenizer library with HuggingFace compatibility for the Zerfoo ML ecosystem. This guide will help you get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Building from Source](#building-from-source)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Good First Issues](#good-first-issues)
- [Key Conventions](#key-conventions)

## Development Setup

### Prerequisites

- **Go 1.25+**
- **Git**

No GPU, C compiler, or external libraries are required. ztoken has zero external dependencies beyond the Go standard library and `golang.org/x/text`.

### Clone and Verify

```bash
git clone https://github.com/zerfoo/ztoken.git
cd ztoken
go mod tidy
go test ./...
```

## Building from Source

```bash
go build ./...
```

ztoken compiles on every platform Go supports with no additional setup.

## Running Tests

```bash
# Run all tests
go test ./...

# Run tests with race detector
go test -race ./...

# Run tests with coverage
go test -cover ./...
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
```

All new code must have tests. Aim for at least 80% coverage on new packages.

## Code Style

### Formatting and Linting

- **`gofmt`** — all code must be formatted with `gofmt`
- **`goimports`** — imports must be organized (stdlib, external)
- **`golangci-lint`** — run `golangci-lint run` before submitting

### Go Conventions

- Follow standard Go naming: PascalCase for exported symbols, camelCase for unexported
- Use table-driven tests with `t.Run` subtests
- Write documentation comments for all exported functions, types, and methods

## Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/) for automated versioning with release-please.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `perf` | A performance improvement |
| `docs` | Documentation only changes |
| `test` | Adding or correcting tests |
| `chore` | Maintenance tasks, CI, dependencies |
| `refactor` | Code change that neither fixes a bug nor adds a feature |

### Examples

```
feat(bpe): add support for Llama 3 tokenizer format
fix(decode): handle surrogate pairs in Unicode decoding
perf(encode): optimize merge priority queue for long sequences
docs: update HuggingFace compatibility notes
test: add round-trip encoding/decoding tests for Gemma vocabulary
```

## Pull Request Process

1. **One logical change per PR** — keep PRs focused and reviewable
2. **Branch from `main`** and keep your branch up to date with rebase
3. **All CI checks must pass** — tests, linting, formatting
4. **Rebase and merge** — we do not use squash merges or merge commits
5. **Reference related issues** — use `Fixes #123` or `Closes #123` in the PR description
6. **Respond to review feedback** promptly

### Before Submitting

```bash
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
```

## Issue Reporting

### Bug Reports

Please include:

- **Description**: Clear summary of the bug
- **Steps to reproduce**: Minimal code with the tokenizer model file used
- **Expected behavior**: Expected token IDs or decoded text
- **Actual behavior**: Actual token IDs or decoded text
- **Environment**: Go version, OS
- **Tokenizer model**: Which HuggingFace model's tokenizer was used

### Feature Requests

Please include:

- **Problem statement**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you thought about
- **Use case**: How would you use this feature in practice?

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/zerfoo/ztoken/labels/good%20first%20issue) on GitHub. These are scoped, well-defined tasks suitable for new contributors.

Good areas for first contributions:

- Adding test coverage for edge cases in encoding/decoding
- Documentation improvements
- Supporting additional HuggingFace tokenizer configurations
- Performance optimizations in the BPE merge loop

## Key Conventions

These conventions are critical to maintaining consistency across the codebase:

### HuggingFace compatibility

ztoken must produce identical token IDs to the HuggingFace `tokenizers` library for all supported models. When adding support for a new tokenizer format, verify against the Python reference implementation:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("model-name")
print(tok.encode("test string"))
```

### Zero external dependencies

ztoken depends only on the Go standard library and `golang.org/x/text`. Do not add third-party dependencies. This keeps the library lightweight and easy to embed.

### Stdlib-only testing

Tests use only the `testing` package from the standard library. Do not introduce test frameworks like testify.
