use std::cell::Cell;
use std::rc::Rc;

use cten::{
    COO, Diagonal, HRect, Physical, Product, ReductionTensor, Stride, Sum, Sup, Tensor,
    ValueFunction, compose, multiset_lift, rewrite, sparse_coo_tensor, square, sum, transpose,
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

/// `f`, counting how many times it is applied.
fn counting(calls: &Rc<Cell<usize>>, f: impl Fn(f32) -> f32 + 'static) -> ValueFunction<f32> {
    let calls = calls.clone();
    ValueFunction::new(move |x: f32| {
        calls.set(calls.get() + 1);
        f(x)
    })
}

#[test]
fn theorem_1_keeps_the_values() {
    let rewritten = rewrite(square(sparse()));

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
    collect_2d(&compose(counting(&calls, |x| x * x), sparse()), 2, 4);
    assert_eq!(calls.get(), 8);

    let calls = Rc::new(Cell::new(0));
    let rewritten = rewrite(compose(counting(&calls, |x| x * x), sparse()));
    // Checking that the function keeps 0 applied it once.
    calls.set(0);
    collect_2d(&rewritten, 2, 4);
    assert_eq!(calls.get(), 3);
}

#[test]
fn a_rewritten_tensor_prints_like_the_original() {
    assert_eq!(
        rewrite(square(sparse())).to_string(),
        square(sparse()).to_string()
    );
}

// Adding 1 does not keep 0, so theorem 1 does not apply, and the tensor is left
// as it is: the function is still applied at every index.
#[test]
fn rewrite_leaves_a_function_that_does_not_keep_zero_alone() {
    let calls = Rc::new(Cell::new(0));
    let unchanged = rewrite(compose(counting(&calls, |x| x + 1.0), sparse()));
    calls.set(0);

    assert_eq!(
        collect_2d(&unchanged, 2, 4),
        vec![vec![1.0, 1.0, 4.0, 1.0], vec![5.0, 1.0, 6.0, 1.0]]
    );
    assert_eq!(calls.get(), 8);
}

// With several values at one index, squaring their sum is not summing their
// squares, so theorem 1 does not apply there and rewriting changes nothing.
#[test]
fn rewrite_leaves_several_values_at_one_index_alone() {
    let repeated = || sparse_coo_tensor(vec![vec![1, 1]], Physical::new(vec![2.0, 3.0]), vec![3]);
    let matrix = || {
        compose(
            Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Stride::new(HRect::new(vec![2, 3]), vec![3, 1]),
        )
    };

    let squared_sums = rewrite(square(repeated()));
    let squared_row_sums = rewrite(square(sum(matrix(), 1)));

    let read =
        |tensor: &dyn Tensor, len| (0..len).map(|i| tensor.apply(&[i])).collect::<Vec<f32>>();
    assert_eq!(read(&squared_sums, 3), vec![0.0, 25.0, 0.0]);
    assert_eq!(read(&squared_row_sums, 2), vec![36.0, 225.0]);
}

#[test]
fn rewrite_takes_any_tensor() {
    let physical = rewrite(Physical::new(vec![1.0, 2.0, 3.0]));

    assert_eq!(
        (0..3).map(|i| physical.apply(&[i])).collect::<Vec<f32>>(),
        vec![1.0, 2.0, 3.0]
    );
}

// The result of a rewrite is a sparse tensor again, so a second function after it
// is rewritten the same way, and neither touches more than the stored values.
#[test]
fn a_rewritten_tensor_can_be_rewritten_again() {
    let squares = Rc::new(Cell::new(0));
    let doubles = Rc::new(Cell::new(0));
    let once = rewrite(compose(counting(&squares, |x| x * x), sparse()));
    let twice = rewrite(compose(counting(&doubles, |x| 2.0 * x), once));
    squares.set(0);
    doubles.set(0);

    assert_eq!(
        collect_2d(&twice, 2, 4),
        vec![vec![0.0, 0.0, 18.0, 0.0], vec![32.0, 0.0, 50.0, 0.0]]
    );
    assert_eq!(squares.get(), 3);
    assert_eq!(doubles.get(), 3);
}

// Rewriting reaches inside a tensor: here the square under a transpose.
#[test]
fn rewrite_reaches_inside_a_tensor() {
    let calls = Rc::new(Cell::new(0));
    let rewritten = rewrite(transpose(compose(counting(&calls, |x| x * x), sparse())));
    calls.set(0);

    assert_eq!(
        collect_2d(&rewritten, 4, 2),
        collect_2d(&transpose(square(sparse())), 4, 2)
    );
    assert_eq!(calls.get(), 3);
}

// Lemma 6 extends theorem 1 to any reduction: a function that keeps the neutral
// element of the reduction only has to be applied to the stored values.
#[test]
fn a_function_keeping_the_neutral_element_of_any_reduction_is_rewritten() {
    let products = || {
        compose(
            Product::new(),
            compose(
                multiset_lift(Physical::new(vec![2.0, 3.0, 4.0])),
                Diagonal::new(HRect::new(vec![3, 3])),
            ),
        )
    };
    let calls = Rc::new(Cell::new(0));
    // Squaring keeps 1, the neutral element of a product.
    let rewritten = rewrite(compose(counting(&calls, |x| x * x), products()));
    calls.set(0);

    assert_eq!(
        collect_2d(&rewritten, 3, 3),
        collect_2d(&square(products()), 3, 3)
    );
    assert_eq!(calls.get(), 3);
}

#[test]
fn a_function_changing_the_neutral_element_is_left_alone() {
    let sups = || {
        compose(
            Sup::new(),
            compose(
                multiset_lift(Physical::new(vec![-2.0, 3.0, -4.0])),
                Diagonal::new(HRect::new(vec![3, 3])),
            ),
        )
    };
    let calls = Rc::new(Cell::new(0));
    // Relu sends -∞, the neutral element of a sup, to 0.
    let unchanged = rewrite(compose(counting(&calls, |x| x.max(0.0)), sups()));
    calls.set(0);

    assert_eq!(
        collect_2d(&unchanged, 3, 3),
        vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 0.0]
        ]
    );
    assert_eq!(calls.get(), 9);
}
