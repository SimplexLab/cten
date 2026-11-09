use std::cell::Cell;
use std::rc::Rc;

use cten::{
    COO, HRect, Physical, ReductionTensor, Stride, Sum, Tensor, ValueFunction, compose,
    diagonalize, rewrite, scale, sparse_coo_tensor, square, sum,
};

/// The sparse matrix of torch's `sparse_coo_tensor` documentation, which stores
/// three values: `[[0, 0, 3, 0], [4, 0, 5, 0]]`.
fn sparse() -> ReductionTensor<Sum, Physical, COO> {
    sparse_coo_tensor(
        vec![vec![0, 1, 1], vec![2, 0, 2]],
        Physical::new(vec![3.0, 4.0, 5.0]),
        vec![2, 4],
    )
}

fn collect_2d(tensor: &dyn Tensor, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|i| (0..cols).map(|j| tensor.apply(&[i, j])).collect())
        .collect()
}

/// `x ↦ x²`, counting how many times it is applied.
fn counting_square(calls: &Rc<Cell<usize>>) -> ValueFunction<f32> {
    let calls = calls.clone();
    ValueFunction::new(move |x: f32| {
        calls.set(calls.get() + 1);
        x * x
    })
}

#[test]
fn theorem_1_rewrites_a_function_after_a_sparse_tensor_into_a_sparse_tensor() {
    let Ok(rewritten) = rewrite(square(sparse())) else {
        panic!("squaring keeps 0, so theorem 1 applies");
    };

    assert_eq!(
        collect_2d(&rewritten, 2, 4),
        vec![vec![0.0, 0.0, 9.0, 0.0], vec![16.0, 0.0, 25.0, 0.0]]
    );
    assert_eq!(
        collect_2d(&rewritten, 2, 4),
        collect_2d(&square(sparse()), 2, 4)
    );
}

// The point of theorem 1: evaluated as written, the function is applied at every
// index of the matrix; rewritten, only at the three stored values.
#[test]
fn a_rewritten_tensor_applies_the_function_to_the_stored_values_only() {
    let calls = Rc::new(Cell::new(0));
    collect_2d(&compose(counting_square(&calls), sparse()), 2, 4);
    assert_eq!(calls.get(), 8);

    let calls = Rc::new(Cell::new(0));
    let Ok(rewritten) = rewrite(compose(counting_square(&calls), sparse())) else {
        panic!("squaring keeps 0, so theorem 1 applies");
    };
    // Checking that the function keeps 0 applied it once.
    calls.set(0);
    collect_2d(&rewritten, 2, 4);
    assert_eq!(calls.get(), 3);
}

#[test]
fn a_rewritten_tensor_prints_like_the_original() {
    let Ok(rewritten) = rewrite(square(sparse())) else {
        panic!("squaring keeps 0, so theorem 1 applies");
    };

    assert_eq!(rewritten.to_string(), square(sparse()).to_string());
}

#[test]
fn theorem_1_applies_to_other_functions_and_sparse_formats() {
    let relu = || ValueFunction::new(|x: f32| x.max(0.0));
    let diagonal = || diagonalize(Physical::new(vec![-1.0, 2.0, -3.0]), 2);
    let Ok(rewritten) = rewrite(compose(relu(), diagonal())) else {
        panic!("relu keeps 0, so theorem 1 applies");
    };

    assert_eq!(
        collect_2d(&rewritten, 3, 3),
        vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 0.0]
        ]
    );
    assert_eq!(
        collect_2d(&rewritten, 3, 3),
        collect_2d(&compose(relu(), diagonal()), 3, 3)
    );
}

// The result is a sparse tensor again, so a second function rewrites the same way.
#[test]
fn a_rewritten_tensor_can_be_rewritten_again() {
    let Ok(squared) = rewrite(square(sparse())) else {
        panic!("squaring keeps 0, so theorem 1 applies");
    };
    let Ok(rewritten) = rewrite(scale(squared, 2.0)) else {
        panic!("scaling keeps 0, so theorem 1 applies");
    };

    assert_eq!(
        collect_2d(&rewritten, 2, 4),
        vec![vec![0.0, 0.0, 18.0, 0.0], vec![32.0, 0.0, 50.0, 0.0]]
    );
}

// Adding 1 does not keep 0: off the stored values, the tensor is 1 and not 0. The
// tensor comes back unchanged.
#[test]
fn theorem_1_needs_a_function_that_keeps_zero() {
    let plus_one = ValueFunction::new(|x: f32| x + 1.0);
    let Err(unchanged) = rewrite(compose(plus_one, sparse())) else {
        panic!("adding 1 does not keep 0, so theorem 1 does not apply");
    };

    assert_eq!(
        collect_2d(&unchanged, 2, 4),
        vec![vec![1.0, 1.0, 4.0, 1.0], vec![5.0, 1.0, 6.0, 1.0]]
    );
}

// With several values at one index, squaring their sum is not summing their
// squares, so theorem 1 does not apply.
#[test]
fn theorem_1_needs_at_most_one_stored_value_per_index() {
    let repeated = sparse_coo_tensor(vec![vec![1, 1]], Physical::new(vec![2.0, 3.0]), vec![3]);
    let matrix = compose(
        Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        Stride::new(HRect::new(vec![2, 3]), vec![3, 1]),
    );
    let row_sums = sum(matrix, 1);

    assert!(rewrite(square(repeated)).is_err());
    assert!(rewrite(square(row_sums)).is_err());
}
