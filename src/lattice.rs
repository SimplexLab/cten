//! Inverting an affine map on a box without scanning the box: finding every
//! index `x` with `basis·x = c` and `0 <= x_j < extents[j]`.
//!
//! Integer column operations of determinant ±1 bring the basis to
//! `basis·[particular | kernel] = [h | 0]`, where `h` is a staircase of full
//! column rank. Every integer solution is then `x = particular·z + kernel·t`,
//! exactly: `z` is unique and found by substitution down the staircase of `h`,
//! and `t` ranges over the integer kernel. The kernel's basis is brought to a
//! staircase too, so that each coordinate of `t` in turn gets an exact range
//! from the extents.

/// The indices of a box that `x ↦ basis·x` sends to a given point.
pub(crate) struct Preimage {
    h: Vec<Vec<i64>>,
    h_pivots: Vec<usize>,
    particular: Vec<Vec<i64>>,
    kernel: Vec<Vec<i64>>,
    kernel_pivots: Vec<usize>,
    extents: Vec<i64>,
}

impl Preimage {
    pub(crate) fn new(basis: &[Vec<i64>], extents: &[usize]) -> Self {
        let n = extents.len();
        let (echelon, operations, h_pivots) = column_echelon(basis.to_vec(), n);
        let r = h_pivots.len();
        let h = echelon.iter().map(|row| row[..r].to_vec()).collect();
        let particular = operations.iter().map(|row| row[..r].to_vec()).collect();
        let free = operations.iter().map(|row| row[r..].to_vec()).collect();
        let (kernel, _, kernel_pivots) = column_echelon(free, n - r);
        Self {
            h,
            h_pivots,
            particular,
            kernel,
            kernel_pivots,
            extents: extents.iter().map(|&extent| extent as i64).collect(),
        }
    }

    /// Every index `x` of the box with `basis·x = c`.
    pub(crate) fn solve(&self, c: &[i64]) -> Vec<Vec<usize>> {
        let mut z = vec![0; self.h_pivots.len()];
        for (i, &row) in self.h_pivots.iter().enumerate() {
            let rest = c[row] - dot(&self.h[row][..i], &z[..i]);
            let step = self.h[row][i];
            if rest % step != 0 {
                return Vec::new();
            }
            z[i] = rest / step;
        }
        // The rows between pivots must hold too.
        if self
            .h
            .iter()
            .zip(c)
            .any(|(row, &target)| dot(row, &z) != target)
        {
            return Vec::new();
        }
        let mut x: Vec<i64> = self.particular.iter().map(|row| dot(row, &z)).collect();
        // Coordinates above the kernel's first pivot are already final.
        let fixed = self.kernel_pivots.first().copied().unwrap_or(x.len());
        if !(0..fixed).all(|j| (0..self.extents[j]).contains(&x[j])) {
            return Vec::new();
        }
        let mut found = Vec::new();
        self.enumerate(0, &mut x, &mut found);
        found
    }

    /// Tries every value of `t[level]` that keeps the coordinates it settles in
    /// range, these being the rows from its pivot to the next one.
    fn enumerate(&self, level: usize, x: &mut [i64], found: &mut Vec<Vec<usize>>) {
        if level == self.kernel_pivots.len() {
            found.push(x.iter().map(|&coord| coord as usize).collect());
            return;
        }
        let end = self
            .kernel_pivots
            .get(level + 1)
            .copied()
            .unwrap_or(x.len());
        let rows = self.kernel_pivots[level]..end;
        let settled = self.kernel[rows.clone()]
            .iter()
            .zip(&x[rows.clone()])
            .zip(&self.extents[rows]);
        let (mut low, mut high) = (i64::MIN, i64::MAX);
        for ((row, &coord), &extent) in settled {
            let (step, last) = (row[level], extent - 1);
            if step > 0 {
                low = low.max(ceil_div(-coord, step));
                high = high.min(floor_div(last - coord, step));
            } else if step < 0 {
                low = low.max(ceil_div(last - coord, step));
                high = high.min(floor_div(-coord, step));
            } else if !(0..=last).contains(&coord) {
                return;
            }
        }
        for t in low..=high {
            for (coord, row) in x.iter_mut().zip(&self.kernel) {
                *coord += row[level] * t;
            }
            self.enumerate(level + 1, x, found);
            for (coord, row) in x.iter_mut().zip(&self.kernel) {
                *coord -= row[level] * t;
            }
        }
    }
}

/// Brings `matrix` to column echelon form by integer column operations of
/// determinant ±1. Returns the echelon form `e`, the operations `v` with
/// `matrix·v = e`, and the pivot row of each column of `e` that is not zero.
/// Those columns come first: each is zero above its pivot, and every column is
/// zero to the right of a pivot.
fn column_echelon(
    mut matrix: Vec<Vec<i64>>,
    cols: usize,
) -> (Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<usize>) {
    let mut operations: Vec<Vec<i64>> = (0..cols)
        .map(|i| (0..cols).map(|j| i64::from(i == j)).collect())
        .collect();
    let mut pivots = Vec::new();
    for row in 0..matrix.len() {
        let c = pivots.len();
        if c == cols {
            break;
        }
        // Folds every entry right of column `c` into it, leaving their gcd there.
        for j in c + 1..cols {
            let (a, b) = (matrix[row][c], matrix[row][j]);
            if b != 0 {
                let (g, s, t) = ext_gcd(a, b);
                let operation = (s, t, -b / g, a / g);
                column_op(&mut matrix, c, j, operation);
                column_op(&mut operations, c, j, operation);
            }
        }
        if matrix[row][c] != 0 {
            pivots.push(row);
        }
    }
    (matrix, operations, pivots)
}

/// Turns columns `c` and `j` into `s·c + t·j` and `u·c + w·j`.
fn column_op(matrix: &mut [Vec<i64>], c: usize, j: usize, (s, t, u, w): (i64, i64, i64, i64)) {
    for row in matrix {
        let (x, y) = (row[c], row[j]);
        row[c] = s * x + t * y;
        row[j] = u * x + w * y;
    }
}

/// `(g, s, t)` with `s·a + t·b = g`, where `g` is the gcd of `a` and `b`, and
/// is not negative.
fn ext_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        if a < 0 { (-a, -1, 0) } else { (a, 1, 0) }
    } else {
        let (g, s, t) = ext_gcd(b, a % b);
        (g, t, s - (a / b) * t)
    }
}

fn dot(a: &[i64], b: &[i64]) -> i64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn floor_div(n: i64, d: i64) -> i64 {
    if d < 0 {
        (-n).div_euclid(-d)
    } else {
        n.div_euclid(d)
    }
}

fn ceil_div(n: i64, d: i64) -> i64 {
    -floor_div(-n, d)
}
