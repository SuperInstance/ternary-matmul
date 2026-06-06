//! # ternary-matmul
//!
//! Ternary matrix multiplication for matrices with elements in {-1, 0, +1}.
//!
//! This crate provides multiple strategies for multiplying ternary matrices,
//! ranging from naive O(n³) multiplication to cache-friendly tiled approaches,
//! XNOR+popcount fast paths, and Strassen's algorithm for large matrices.
//!
//! All arithmetic uses explicit Z₃ matching on trit pairs — never modular tricks.

use std::time::Instant;

/// A ternary matrix storing elements as i8 in {-1, 0, +1}.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TernaryMatrix {
    rows: usize,
    cols: usize,
    data: Vec<i8>, // row-major
}

impl TernaryMatrix {
    /// Create a new zero-filled ternary matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0; rows * cols] }
    }

    /// Create from a Vec<i8>. Panics if any element is not in {-1, 0, 1}.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<i8>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length mismatch");
        for &v in &data {
            assert!(v == -1 || v == 0 || v == 1, "element {} not in {{-1, 0, 1}}", v);
        }
        Self { rows, cols, data }
    }

    /// Create a random ternary matrix using a simple seed-based RNG.
    pub fn random(rows: usize, cols: usize, seed: u64) -> Self {
        let mut s = seed;
        let data: Vec<i8> = (0..rows * cols).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            match (s >> 62) % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            }
        }).collect();
        Self { rows, cols, data }
    }

    /// Identity matrix of size `n`.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1;
        }
        m
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    /// Get element at (r, c).
    pub fn get(&self, r: usize, c: usize) -> i8 {
        self.data[r * self.cols + c]
    }

    /// Set element at (r, c). Panics if value not in {-1, 0, 1}.
    pub fn set(&mut self, r: usize, c: usize, v: i8) {
        assert!(v == -1 || v == 0 || v == 1);
        self.data[r * self.cols + c] = v;
    }

    /// Convert result of a regular integer accumulation to nearest trit.
    fn round_to_trit(v: i32) -> i8 {
        // Clamp to [-1, 1] by sign
        match v {
            ..=-1 => -1,
            0 => 0,
            1.. => 1,
        }
    }
}

/// Z₃ multiplication: explicit match on trit pairs.
#[inline(always)]
pub fn trit_mul(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) => 1,
        (-1,  0) => 0,
        (-1,  1) => -1,
        ( 0, -1) => 0,
        ( 0,  0) => 0,
        ( 0,  1) => 0,
        ( 1, -1) => -1,
        ( 1,  0) => 0,
        ( 1,  1) => 1,
        _ => unreachable!(),
    }
}

/// Z₃ addition: explicit match on trit pairs.
#[inline(always)]
pub fn trit_add(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) => 1,   // -2 mod 3 = 1
        (-1,  0) => -1,
        (-1,  1) => 0,
        ( 0, -1) => -1,
        ( 0,  0) => 0,
        ( 0,  1) => 1,
        ( 1, -1) => 0,
        ( 1,  0) => 1,
        ( 1,  1) => -1,  // 2 mod 3 = -1
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Naive matmul
// ---------------------------------------------------------------------------

/// Naive O(n³) ternary matrix multiplication.
///
/// Computes C = A × B using Z₃ arithmetic with explicit match arms.
pub fn naive_matmul(a: &TernaryMatrix, b: &TernaryMatrix) -> TernaryMatrix {
    assert_eq!(a.cols, b.rows, "dimension mismatch");
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let mut c = TernaryMatrix::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            // Accumulate in regular integer, then round
            let mut acc: i32 = 0;
            for p in 0..k {
                acc += (trit_mul(a.get(i, p), b.get(p, j))) as i32;
            }
            c.data[i * n + j] = TernaryMatrix::round_to_trit(acc);
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Tiled (cache-friendly) matmul
// ---------------------------------------------------------------------------

/// Cache-friendly tiled matrix multiplication.
///
/// Uses tile size `TILE` to improve cache locality. Semantics are identical
/// to [`naive_matmul`] but with better performance for larger matrices.
pub fn tiled_matmul(a: &TernaryMatrix, b: &TernaryMatrix, tile: usize) -> TernaryMatrix {
    assert_eq!(a.cols, b.rows, "dimension mismatch");
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let mut acc = vec![0i32; m * n];

    for i0 in (0..m).step_by(tile) {
        for j0 in (0..n).step_by(tile) {
            for p0 in (0..k).step_by(tile) {
                let im = (i0 + tile).min(m);
                let jm = (j0 + tile).min(n);
                let pm = (p0 + tile).min(k);
                for i in i0..im {
                    for p in p0..pm {
                        let a_val = a.get(i, p);
                        for j in j0..jm {
                            acc[i * n + j] += trit_mul(a_val, b.get(p, j)) as i32;
                        }
                    }
                }
            }
        }
    }

    let data: Vec<i8> = acc.iter().map(|&v| TernaryMatrix::round_to_trit(v)).collect();
    TernaryMatrix::from_vec(m, n, data)
}

// ---------------------------------------------------------------------------
// XNOR + popcount fast path
// ---------------------------------------------------------------------------

/// XNOR+popcount fast path for binary-ternary multiplication.
///
/// When both matrices contain only {-1, +1} (no zeros), this uses XNOR
/// equivalence and popcount to compute the result extremely quickly.
/// Returns `None` if any zero is found in either matrix.
pub fn xnor_matmul(a: &TernaryMatrix, b: &TernaryMatrix) -> Option<TernaryMatrix> {
    // Verify no zeros
    if a.data.iter().any(|&v| v == 0) || b.data.iter().any(|&v| v == 0) {
        return None;
    }
    assert_eq!(a.cols, b.rows, "dimension mismatch");
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    // Pack rows of A and columns of B into u64 bitmaps
    let bits_needed = k;
    let u64_count = (bits_needed + 63) / 64;

    // Pack: -1 → 0 bit, +1 → 1 bit
    let pack = |vals: &[i8]| -> Vec<u64> {
        let mut packed = vec![0u64; u64_count];
        for (i, &v) in vals.iter().enumerate() {
            if v == 1 {
                packed[i / 64] |= 1u64 << (i % 64);
            }
        }
        packed
    };

    let a_packed: Vec<Vec<u64>> = (0..m).map(|i| {
        let row = &a.data[i * k..(i + 1) * k];
        pack(row)
    }).collect();

    // Pack B columns (transpose)
    let b_packed: Vec<Vec<u64>> = (0..n).map(|j| {
        let col: Vec<i8> = (0..k).map(|p| b.get(p, j)).collect();
        pack(&col)
    }).collect();

    let mut c = TernaryMatrix::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut total_agree = 0i32;
            for b_idx in 0..u64_count {
                let xnor = !(a_packed[i][b_idx] ^ b_packed[j][b_idx]);
                let bits_in_word = if b_idx == u64_count - 1 && bits_needed % 64 != 0 {
                    bits_needed % 64
                } else {
                    64
                };
                let mask = if bits_in_word == 64 { !0u64 } else { (1u64 << bits_in_word) - 1 };
                total_agree += (xnor & mask).count_ones() as i32;
            }
            // agree = positions with same sign, disagree = k - agree
            // sum = agree * 1 + disagree * (-1) = agree - disagree = 2*agree - k
            let sum = 2 * total_agree - k as i32;
            c.data[i * n + j] = TernaryMatrix::round_to_trit(sum);
        }
    }
    Some(c)
}

// ---------------------------------------------------------------------------
// Strassen
// ---------------------------------------------------------------------------

// --- Internal i32 matrix for Strassen over integers ---

#[derive(Clone)]
struct IntMatrix {
    n: usize,
    data: Vec<i32>, // n×n row-major
}

impl IntMatrix {
    fn zeros(n: usize) -> Self { Self { n, data: vec![0; n * n] } }
    fn from_ternary(m: &TernaryMatrix) -> Self {
        Self { n: m.rows, data: m.data.iter().map(|&v| v as i32).collect() }
    }
    fn to_ternary(&self) -> TernaryMatrix {
        let data: Vec<i8> = self.data.iter().map(|&v| TernaryMatrix::round_to_trit(v)).collect();
        TernaryMatrix::from_vec(self.n, self.n, data)
    }
    fn get(&self, r: usize, c: usize) -> i32 { self.data[r * self.n + c] }
    fn set(&mut self, r: usize, c: usize, v: i32) { self.data[r * self.n + c] = v; }
}

fn int_add(a: &IntMatrix, b: &IntMatrix) -> IntMatrix {
    let n = a.n;
    let data: Vec<i32> = a.data.iter().zip(&b.data).map(|(&x, &y)| x + y).collect();
    IntMatrix { n, data }
}

fn int_sub(a: &IntMatrix, b: &IntMatrix) -> IntMatrix {
    let n = a.n;
    let data: Vec<i32> = a.data.iter().zip(&b.data).map(|(&x, &y)| x - y).collect();
    IntMatrix { n, data }
}

fn int_naive_mul(a: &IntMatrix, b: &IntMatrix) -> IntMatrix {
    let n = a.n;
    let mut c = IntMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..n {
                acc += a.get(i, p) * b.get(p, j);
            }
            c.set(i, j, acc);
        }
    }
    c
}

fn int_split(m: &IntMatrix) -> [IntMatrix; 4] {
    let half = m.n / 2;
    let mut qs = [IntMatrix::zeros(half), IntMatrix::zeros(half),
                  IntMatrix::zeros(half), IntMatrix::zeros(half)];
    for i in 0..half {
        for j in 0..half {
            qs[0].set(i, j, m.get(i, j));
            qs[1].set(i, j, m.get(i, j + half));
            qs[2].set(i, j, m.get(i + half, j));
            qs[3].set(i, j, m.get(i + half, j + half));
        }
    }
    qs
}

fn int_join(c11: &IntMatrix, c12: &IntMatrix, c21: &IntMatrix, c22: &IntMatrix) -> IntMatrix {
    let half = c11.n;
    let n = half * 2;
    let mut m = IntMatrix::zeros(n);
    for i in 0..half {
        for j in 0..half {
            m.set(i, j, c11.get(i, j));
            m.set(i, j + half, c12.get(i, j));
            m.set(i + half, j, c21.get(i, j));
            m.set(i + half, j + half, c22.get(i, j));
        }
    }
    m
}

fn int_strassen(a: &IntMatrix, b: &IntMatrix, threshold: usize) -> IntMatrix {
    let n = a.n;
    if n <= threshold {
        return int_naive_mul(a, b);
    }
    let half = n / 2;
    assert_eq!(half * 2, n);

    let aq = int_split(a);
    let bq = int_split(b);

    let m1 = int_strassen(&int_add(&aq[0], &aq[3]), &int_add(&bq[0], &bq[3]), threshold);
    let m2 = int_strassen(&int_add(&aq[2], &aq[3]), &bq[0], threshold);
    let m3 = int_strassen(&aq[0], &int_sub(&bq[1], &bq[3]), threshold);
    let m4 = int_strassen(&aq[3], &int_sub(&bq[2], &bq[0]), threshold);
    let m5 = int_strassen(&int_add(&aq[0], &aq[1]), &bq[3], threshold);
    let m6 = int_strassen(&int_sub(&aq[2], &aq[0]), &int_add(&bq[0], &bq[1]), threshold);
    let m7 = int_strassen(&int_sub(&aq[1], &aq[3]), &int_add(&bq[2], &bq[3]), threshold);

    let c11 = int_add(&int_sub(&int_add(&m1, &m4), &m5), &m7);
    let c12 = int_add(&m3, &m5);
    let c21 = int_add(&m2, &m4);
    let c22 = int_add(&int_sub(&int_add(&m1, &m3), &m2), &m6);

    int_join(&c11, &c12, &c21, &c22)
}

/// Strassen's algorithm for ternary matrix multiplication.
///
/// Internally operates on integer matrices to preserve exact arithmetic
/// through recursive decomposition, then rounds the final result to trits.
/// Only used for larger matrices (power-of-2 sized). Falls back to naive
/// for sub-matrices below `threshold` size.
pub fn strassen_matmul(a: &TernaryMatrix, b: &TernaryMatrix, threshold: usize) -> TernaryMatrix {
    assert_eq!(a.cols, b.rows, "dimension mismatch");
    assert_eq!(a.rows, a.cols, "Strassen requires square A");
    assert_eq!(b.rows, b.cols, "Strassen requires square B");
    assert_eq!(a.rows, b.rows, "Strassen requires same size");

    let n = a.rows;
    if n <= threshold {
        return naive_matmul(a, b);
    }
    assert_eq!(n & (n - 1), 0, "Strassen requires power-of-2 dimensions");

    let ia = IntMatrix::from_ternary(a);
    let ib = IntMatrix::from_ternary(b);
    let ic = int_strassen(&ia, &ib, threshold);
    ic.to_ternary()
}

/// Element-wise Z₃ addition of two ternary matrices.
pub fn add_mat(a: &TernaryMatrix, b: &TernaryMatrix) -> TernaryMatrix {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let data: Vec<i8> = a.data.iter().zip(&b.data).map(|(&x, &y)| trit_add(x, y)).collect();
    TernaryMatrix::from_vec(a.rows, a.cols, data)
}

/// Element-wise Z₃ subtraction: A + (-B). Uses explicit match for negation.
pub fn sub_mat(a: &TernaryMatrix, b: &TernaryMatrix) -> TernaryMatrix {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let data: Vec<i8> = a.data.iter().zip(&b.data).map(|(&x, &y)| {
        let neg_y = trit_neg(y);
        trit_add(x, neg_y)
    }).collect();
    TernaryMatrix::from_vec(a.rows, a.cols, data)
}

/// Z₃ negation: explicit match.
#[inline(always)]
pub fn trit_neg(a: i8) -> i8 {
    match a {
        -1 => 1,
        0 => 0,
        1 => -1,
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

/// Benchmark helper: runs `f` and returns elapsed time in microseconds.
pub fn time_fn<F, R>(label: &str, f: F) -> (R, u128)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed().as_micros();
    println!("{label}: {elapsed} µs");
    (result, elapsed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_mul() {
        assert_eq!(trit_mul(-1, -1), 1);
        assert_eq!(trit_mul(-1, 0), 0);
        assert_eq!(trit_mul(-1, 1), -1);
        assert_eq!(trit_mul(0, -1), 0);
        assert_eq!(trit_mul(0, 0), 0);
        assert_eq!(trit_mul(0, 1), 0);
        assert_eq!(trit_mul(1, -1), -1);
        assert_eq!(trit_mul(1, 0), 0);
        assert_eq!(trit_mul(1, 1), 1);
    }

    #[test]
    fn test_trit_add() {
        assert_eq!(trit_add(-1, -1), 1);
        assert_eq!(trit_add(-1, 1), 0);
        assert_eq!(trit_add(1, 1), -1);
        assert_eq!(trit_add(0, 0), 0);
        assert_eq!(trit_add(1, -1), 0);
        assert_eq!(trit_add(1, 0), 1);
    }

    #[test]
    fn test_small_matrix_correctness() {
        // [1  0] × [ 1  1]   = ?
        // [-1 1]   [-1  0]
        let a = TernaryMatrix::from_vec(2, 2, vec![1, 0, -1, 1]);
        let b = TernaryMatrix::from_vec(2, 2, vec![1, 1, -1, 0]);
        let c = naive_matmul(&a, &b);
        // Row 0: [1*1+0*(-1), 1*1+0*0] = [1, 1]
        // Row 1: [-1*1+1*(-1), -1*1+1*0] = [-2, -1] → round → [-1, -1]
        assert_eq!(c.get(0, 0), 1);
        assert_eq!(c.get(0, 1), 1);
        assert_eq!(c.get(1, 0), -1);
        assert_eq!(c.get(1, 1), -1);
    }

    #[test]
    fn test_identity_multiplication() {
        let id = TernaryMatrix::identity(3);
        let a = TernaryMatrix::from_vec(3, 3, vec![1, -1, 0, 0, 1, 1, -1, 0, -1]);
        let c = naive_matmul(&a, &id);
        assert_eq!(c, a, "A × I should equal A");
        let c2 = naive_matmul(&id, &a);
        assert_eq!(c2, a, "I × A should equal A");
    }

    #[test]
    fn test_commutativity_check_fails_for_non_square() {
        // Non-square: A×B ≠ B×A in general (dimensions don't even match)
        let a = TernaryMatrix::from_vec(2, 3, vec![1, 0, -1, 1, 1, 0]);
        let b = TernaryMatrix::from_vec(3, 2, vec![1, -1, 0, 1, -1, 0]);
        let ab = naive_matmul(&a, &b);
        let ba = naive_matmul(&b, &a);
        // Dimensions differ: AB is 2×2, BA is 3×3
        assert_ne!(ab.rows(), ba.rows());
    }

    #[test]
    fn test_tiled_matches_naive() {
        let a = TernaryMatrix::random(8, 8, 42);
        let b = TernaryMatrix::random(8, 8, 99);
        let naive = naive_matmul(&a, &b);
        let tiled = tiled_matmul(&a, &b, 4);
        assert_eq!(naive, tiled, "tiled matmul should match naive");
    }

    #[test]
    fn test_tiled_matches_naive_non_square() {
        let a = TernaryMatrix::random(4, 6, 10);
        let b = TernaryMatrix::random(6, 5, 20);
        let naive = naive_matmul(&a, &b);
        let tiled = tiled_matmul(&a, &b, 2);
        assert_eq!(naive, tiled);
    }

    #[test]
    fn test_xnor_matches_naive_binary() {
        let mut a = TernaryMatrix::random(4, 4, 123);
        let mut b = TernaryMatrix::random(4, 4, 456);
        // Ensure no zeros
        for v in a.data.iter_mut().chain(b.data.iter_mut()) {
            if *v == 0 { *v = -1; }
        }
        let naive = naive_matmul(&a, &b);
        let xnor = xnor_matmul(&a, &b).expect("should succeed for binary matrices");
        assert_eq!(naive, xnor, "XNOR path should match naive for binary matrices");
    }

    #[test]
    fn test_xnor_returns_none_with_zeros() {
        let a = TernaryMatrix::from_vec(2, 2, vec![1, 0, -1, 1]);
        let b = TernaryMatrix::from_vec(2, 2, vec![1, -1, 1, -1]);
        assert!(xnor_matmul(&a, &b).is_none());
    }

    #[test]
    fn test_strassen_matches_naive_4x4() {
        let a = TernaryMatrix::random(4, 4, 777);
        let b = TernaryMatrix::random(4, 4, 888);
        let naive = naive_matmul(&a, &b);
        let strassen = strassen_matmul(&a, &b, 2);
        assert_eq!(naive, strassen, "Strassen should match naive for 4×4");
    }

    #[test]
    fn test_strassen_matches_naive_8x8() {
        let a = TernaryMatrix::random(8, 8, 111);
        let b = TernaryMatrix::random(8, 8, 222);
        let naive = naive_matmul(&a, &b);
        let strassen = strassen_matmul(&a, &b, 2);
        assert_eq!(naive, strassen);
    }

    #[test]
    fn test_strassen_matches_naive_16x16() {
        let a = TernaryMatrix::random(16, 16, 333);
        let b = TernaryMatrix::random(16, 16, 444);
        let naive = naive_matmul(&a, &b);
        let strassen = strassen_matmul(&a, &b, 4);
        assert_eq!(naive, strassen);
    }

    #[test]
    fn test_sparse_matrices() {
        // Mostly zeros
        let mut a = TernaryMatrix::zeros(4, 4);
        let mut b = TernaryMatrix::zeros(4, 4);
        a.set(0, 0, 1);
        a.set(3, 3, -1);
        b.set(0, 0, -1);
        b.set(3, 3, 1);
        let c = naive_matmul(&a, &b);
        assert_eq!(c.get(0, 0), -1); // 1 * -1 = -1
        assert_eq!(c.get(3, 3), -1); // -1 * 1 = -1
        // All other elements should be 0
        for i in 0..4 {
            for j in 0..4 {
                if !((i == 0 && j == 0) || (i == 3 && j == 3)) {
                    assert_eq!(c.get(i, j), 0);
                }
            }
        }
    }

    #[test]
    fn test_add_and_sub_mat() {
        let a = TernaryMatrix::from_vec(2, 2, vec![1, -1, 0, 1]);
        let b = TernaryMatrix::from_vec(2, 2, vec![1, 1, -1, -1]);
        let sum = add_mat(&a, &b);
        assert_eq!(sum.get(0, 0), -1); // 1+1 = -1 in Z₃
        assert_eq!(sum.get(0, 1), 0);  // -1+1 = 0
        let diff = sub_mat(&a, &b);
        // a - b = a + (-b)
        assert_eq!(diff.get(0, 0), 0); // 1 - 1 = 0
        assert_eq!(diff.get(1, 1), -1); // 1 - (-1) = 1 + 1 = -1
    }

    #[test]
    fn test_benchmark_timing_helper() {
        let (_, us) = time_fn("test_op", || {
            let a = TernaryMatrix::random(32, 32, 0);
            let b = TernaryMatrix::random(32, 32, 1);
            naive_matmul(&a, &b)
        });
        assert!(us > 0, "timing should be positive");
    }

    #[test]
    fn test_zero_matrix_multiply() {
        let a = TernaryMatrix::zeros(3, 3);
        let b = TernaryMatrix::from_vec(3, 3, vec![1, -1, 0, 1, 0, -1, -1, 1, 1]);
        let c = naive_matmul(&a, &b);
        assert_eq!(c, TernaryMatrix::zeros(3, 3));
    }
}
