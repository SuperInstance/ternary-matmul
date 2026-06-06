# ternary-matmul

**Ternary matrix multiplication for {-1, 0, +1} matrices — with multiple algorithmic strategies.**

[![crate](https://img.shields.io/badge/crates.io-ternary--matmul-orange)](https://crates.io)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Overview

`ternary-matmul` provides efficient multiplication of matrices whose elements are restricted to the **ternary alphabet** {-1, 0, +1}, sometimes called *trits* (trinary digits). This restriction enables specialized algorithms that are faster than general-purpose matrix multiplication while maintaining exact Z₃ arithmetic semantics.

The crate implements four distinct multiplication strategies:

| Algorithm | Complexity | Best For |
|-----------|-----------|----------|
| **Naive** | O(n³) | Small matrices, correctness reference |
| **Tiled** | O(n³) | Medium matrices, cache-friendly |
| **XNOR+Popcount** | O(n³/64) | Binary {-1,+1} matrices only |
| **Strassen** | O(n^2.807) | Large power-of-2 matrices |

## Why Ternary Matrices?

Ternary arithmetic arises naturally in several domains:

- **Quantized neural networks**: 1-bit and ternary weight quantization (TNNs, TWNs) use {-1, 0, +1} weights to reduce memory and compute
- **Ternary logic circuits**: Beyond-binary logic synthesis and verification
- **Compressed sensing**: Sparse ternary measurement matrices
- **Z₃ algebra**: Mathematical research in modular arithmetic over the field with three elements
- **Hashing & sketching**: Ternary sketch matrices for dimensionality reduction

## Z₃ Arithmetic

All operations use **explicit pattern matching** on trit pairs rather than modular arithmetic tricks like `(a+b+3)%3-1`. This ensures:

1. **Correctness by construction** — every case is enumerated
2. **Optimizable** — the compiler can see all paths
3. **Auditable** — no hidden overflow or sign-extension bugs

```rust
pub fn trit_mul(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) =>  1,  (-1, 0) =>  0,  (-1, 1) => -1,
        ( 0, -1) =>  0,  ( 0, 0) =>  0,  ( 0, 1) =>  0,
        ( 1, -1) => -1,  ( 1, 0) =>  0,  ( 1, 1) =>  1,
        _ => unreachable!(),
    }
}
```

## Quick Start

```rust
use ternary_matmul::*;

// Create matrices
let a = TernaryMatrix::from_vec(2, 3, vec![1, 0, -1, -1, 1, 0]);
let b = TernaryMatrix::from_vec(3, 2, vec![1, -1, 0, 1, -1, 0]);

// Naive multiplication
let c = naive_matmul(&a, &b);

// Cache-friendly tiled multiplication
let c_tiled = tiled_matmul(&a, &b, 4);

// For binary {-1,+1} matrices, use the XNOR fast path
let bin_a = TernaryMatrix::from_vec(4, 4, vec![1,-1,1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,1]);
let bin_b = TernaryMatrix::from_vec(4, 4, vec![-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1]);
if let Some(result) = xnor_matmul(&bin_a, &bin_b) {
    // 64× faster than naive for large matrices
}

// Strassen for large power-of-2 matrices
let big_a = TernaryMatrix::random(128, 128, 42);
let big_b = TernaryMatrix::random(128, 128, 99);
let c_strassen = strassen_matmul(&big_a, &big_b, 16);
```

## Algorithm Details

### Naive Multiplication

Standard triple-loop matrix multiplication with Z₃ trit products. Each element is computed as the sign of the accumulated dot product. Serves as the correctness reference for all other algorithms.

### Tiled Multiplication

Reorganizes the naive algorithm into cache-friendly tiles. Instead of streaming through entire rows/columns, it processes small blocks that fit in L1/L2 cache. The tile size is configurable — typical values are 16–64 depending on your hardware.

### XNOR + Popcount

When both matrices contain only {-1, +1} (no zeros), each element can be encoded as a single bit. The dot product then reduces to:

1. Pack rows/columns into `u64` bitmaps
2. Compute `XNOR` (equivalence) via `!(a ^ b)`
3. Count agreeing bits with `popcount`
4. Compute `sum = 2 × agree - k`

This processes **64 elements per instruction**, making it ~64× faster than the naive approach for large binary-ternary matrices.

### Strassen's Algorithm

Implements the classic divide-and-conquer algorithm using **7 multiplications instead of 8** at each recursion level. Internally operates on integer matrices to preserve exact arithmetic through recursive decomposition, only rounding to trits at the final step. Requires square, power-of-2 dimensions.

For an n×n matrix, Strassen achieves O(n^log₂7) ≈ O(n^2.807) versus O(n³) for the naive approach. The threshold parameter controls when to fall back to naive multiplication — typically 16–64.

## API Reference

### Core Types

- `TernaryMatrix` — Row-major ternary matrix with bounds-checked access
- `trit_mul(a, b)` — Z₃ multiplication
- `trit_add(a, b)` — Z₃ addition
- `trit_neg(a)` — Z₃ negation

### Multiplication Functions

- `naive_matmul(a, b)` — Standard O(n³) multiplication
- `tiled_matmul(a, b, tile_size)` — Cache-tiled multiplication
- `xnor_matmul(a, b)` — Fast binary path, returns `Option`
- `strassen_matmul(a, b, threshold)` — Strassen's algorithm

### Utility Functions

- `add_mat(a, b)` — Element-wise Z₃ addition
- `sub_mat(a, b)` — Element-wise Z₃ subtraction
- `time_fn(label, f)` — Benchmark timing helper

## Performance Tips

1. **Use tiled matmul** with tile sizes matching your L1 cache (typically 32–64)
2. **Use XNOR** whenever your matrices have no zeros — it's dramatically faster
3. **Use Strassen** for matrices ≥ 128×128, with threshold ~16–32
4. **Matrix size matters** — Strassen only works for square power-of-2 matrices

## Testing

```bash
cargo test
```

The test suite includes:
- Small matrix correctness checks
- Identity multiplication invariance
- Non-commutativity verification for non-square matrices
- Tiled vs naive equivalence
- XNOR vs naive equivalence for binary matrices
- Strassen vs naive equivalence for 4×4, 8×8, and 16×16
- Sparse matrix handling
- Zero matrix edge cases

## License

MIT
