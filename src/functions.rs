//! Named constructors for the compositions the library can already express.
//! Everything here is assembled from `function`, `mapping` and `tensor`; none
//! of it adds machinery of its own.

use std::assert_eq;
use std::fmt::Display;
use std::ops::{Mul, Neg};
use std::rc::Rc;

use crate::domain::{Domain, HRect};
use crate::function::{Compose, ValueFunction, compose, multiset_lift};
use crate::mapping::{
    Affine, COO, Concat, Diagonal, EmptyMapping, InvAffine, InvProjection, Permutation, Projection,
    Squeeze, Unsqueeze, Where,
};
use crate::tensor::{
    BoxedTensor, JaggedTensor, Physical, Product, ProductReduction, Reduce, Reduction,
    ReductionTensor, Sum, SumReduction, Tensor,
};

pub fn square<A: Mul<Output = A> + Copy + 'static, U: Tensor<A>>(
    u: U,
) -> Compose<ValueFunction<A>, U> {
    compose(ValueFunction::new(|x: A| x * x), u)
}

pub fn negate<A: Neg<Output = A> + Clone + 'static, U: Tensor<A>>(
    u: U,
) -> Compose<ValueFunction<A>, U> {
    compose(ValueFunction::new(|x: A| -x), u)
}

pub fn scale<A: Mul<Output = A> + Copy + 'static, U: Tensor<A>>(
    u: U,
    factor: A,
) -> Compose<ValueFunction<A>, U> {
    compose(ValueFunction::new(move |x: A| x * factor), u)
}

/// Holds values between `low` and `high`. Checking the bounds here reports the
/// mistake where it is made, not at the first read, and rules out NaN bounds,
/// which nothing would be below or above.
pub fn clamp<A: PartialOrd + Copy + Display + 'static, U: Tensor<A>>(
    u: U,
    low: A,
    high: A,
) -> Compose<ValueFunction<A>, U> {
    assert!(low <= high, "clamp needs low <= high, got {low} and {high}");
    let clamped = ValueFunction::new(move |x: A| {
        if x < low {
            low
        } else if x > high {
            high
        } else {
            x
        }
    });
    compose(clamped, u)
}

/// Reorders the axes of a tensor: `axes[i]` is the axis of `u` that becomes
/// axis `i` of the result.
pub fn permute<A, U: BoxedTensor<A>>(u: U, axes: Vec<usize>) -> Compose<U, Permutation> {
    let codomain = u.hrect().clone();
    compose(u, Permutation::new(codomain, axes))
}

/// Swaps the last two axes and leaves any leading ones alone, so a stack of
/// matrices transposes matrix by matrix. At rank two this is the plain
/// transpose.
pub fn transpose<A, U: BoxedTensor<A>>(u: U) -> Compose<U, Permutation> {
    let ndim = u.hrect().ndim();
    assert!(ndim >= 2, "transpose needs at least two dimensions");
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.swap(ndim - 2, ndim - 1);
    permute(u, axes)
}

/// Moves the axis at `source` to `destination`; the axes left over keep their
/// relative order.
pub fn movedim<A, U: BoxedTensor<A>>(
    u: U,
    source: usize,
    destination: usize,
) -> Compose<U, Permutation> {
    let ndim = u.hrect().ndim();
    assert!(source < ndim, "source axis {source} is out of range");
    assert!(
        destination < ndim,
        "destination axis {destination} is out of range"
    );
    let mut axes: Vec<usize> = (0..ndim).collect();
    let axis = axes.remove(source);
    axes.insert(destination, axis);
    permute(u, axes)
}

/// Inserts an axis of extent 1 at `dim`, which can be anywhere from before the
/// first axis to after the last.
pub fn unsqueeze<A, U: BoxedTensor<A>>(u: U, dim: usize) -> Compose<U, Unsqueeze> {
    let codomain = u.hrect().clone();
    compose(u, Unsqueeze::new(codomain, dim))
}

/// The axes a function is told to act on, in place of torch's optional `dim`
/// argument: `()` names every axis, a `usize` one axis, and an array, vector or
/// slice several.
pub trait Dims {
    fn axes(self, ndim: usize) -> Vec<usize>;
}

impl Dims for () {
    fn axes(self, ndim: usize) -> Vec<usize> {
        (0..ndim).collect()
    }
}

impl Dims for usize {
    fn axes(self, _ndim: usize) -> Vec<usize> {
        vec![self]
    }
}

impl<const N: usize> Dims for [usize; N] {
    fn axes(self, _ndim: usize) -> Vec<usize> {
        self.to_vec()
    }
}

impl Dims for Vec<usize> {
    fn axes(self, _ndim: usize) -> Vec<usize> {
        self
    }
}

impl Dims for &[usize] {
    fn axes(self, _ndim: usize) -> Vec<usize> {
        self.to_vec()
    }
}

/// Drops axes of extent 1, as torch's `squeeze` does: `squeeze(u, ())` drops all
/// of them, `squeeze(u, 1)` only axis 1, and `squeeze(u, [0, 2])` axes 0 and 2.
/// A named axis whose extent isn't 1 is left alone.
pub fn squeeze<A, U: BoxedTensor<A>>(u: U, dims: impl Dims) -> Compose<U, Squeeze> {
    let codomain = u.hrect().clone();
    let ndim = codomain.shape().len();
    let named = dims.axes(ndim);
    for (i, &axis) in named.iter().enumerate() {
        assert!(axis < ndim, "dim {axis} is out of range");
        assert!(
            !named[..i].contains(&axis),
            "dim {axis} is named more than once"
        );
    }
    let removed = named
        .into_iter()
        .filter(|&axis| codomain.shape()[axis] == 1)
        .collect();
    compose(u, Squeeze::new(codomain, removed))
}

/// Joins tensors end to end along `dim`; their other extents must agree.
pub fn concatenate<A>(
    tensors: Vec<Rc<dyn BoxedTensor<A>>>,
    dim: usize,
) -> Compose<JaggedTensor<A>, Concat> {
    assert!(!tensors.is_empty(), "concatenate needs at least one tensor");
    let underlying = JaggedTensor::new(tensors);
    let codomain = underlying.jagged_domain().clone();
    compose(underlying, Concat::new(codomain, dim))
}

/// Joins tensors of one shape along a new axis at `dim`, by concatenating them
/// once each has an axis of extent 1 there.
pub fn stack<A: 'static>(
    tensors: Vec<Rc<dyn BoxedTensor<A>>>,
    dim: usize,
) -> Compose<JaggedTensor<A>, Concat> {
    assert!(!tensors.is_empty(), "stack needs at least one tensor");
    let shape = tensors[0].hrect().clone();
    assert!(
        tensors.iter().all(|tensor| *tensor.hrect() == shape),
        "stack needs tensors of one shape"
    );
    let unsqueezed = tensors
        .into_iter()
        .map(|tensor| {
            let unsqueeze = Unsqueeze::new(tensor.hrect().clone(), dim);
            Rc::new(compose(tensor, unsqueeze)) as Rc<dyn BoxedTensor<A>>
        })
        .collect();
    concatenate(unsqueezed, dim)
}

/// Reads `input` where `condition` holds and `other` where it does not, as
/// torch's `where` does. `where` is a Rust keyword, hence the underscore.
pub fn where_<A>(
    condition: Rc<dyn BoxedTensor<bool>>,
    input: Rc<dyn BoxedTensor<A>>,
    other: Rc<dyn BoxedTensor<A>>,
) -> Compose<JaggedTensor<A>, Where> {
    let underlying = JaggedTensor::new(vec![input, other]);
    let codomain = underlying.jagged_domain().clone();
    compose(underlying, Where::new(codomain, condition))
}

/// Spreads a vector over the diagonal of a tensor of the given rank: rank two
/// gives a diagonal matrix, rank three a tensor whose only non-zero indices
/// are of the form `[i, i, i]`.
pub fn diagonalize<A: Clone, U: BoxedTensor<A>>(
    u: U,
    rank: usize,
) -> ReductionTensor<Sum<A>, U, Diagonal>
where
    SumReduction: Reduction<A>,
{
    assert_eq!(u.hrect().ndim(), 1, "diagonalize takes a vector");
    assert!(rank > 0, "a diagonal tensor has at least one dimension");
    let extent = u.hrect().shape()[0];
    let diagonal = Diagonal::new(HRect::new(vec![extent; rank]));
    compose(Sum::new(), compose(multiset_lift(u), diagonal))
}

/// Reducing over the empty set gives the reduction's neutral element at every
/// index, so a constant tensor needs no underlying data at all.
fn neutral<R: Reduction<f32>>(
    shape: Vec<usize>,
) -> ReductionTensor<Reduce<R>, Physical, EmptyMapping> {
    let mapping = EmptyMapping::new(HRect::new(shape), HRect::new(vec![0]));
    compose(
        Reduce::new(),
        compose(multiset_lift(Physical::new(vec![])), mapping),
    )
}

pub fn zeros(shape: Vec<usize>) -> ReductionTensor<Sum, Physical, EmptyMapping> {
    neutral(shape)
}

pub fn ones(shape: Vec<usize>) -> ReductionTensor<Product, Physical, EmptyMapping> {
    neutral(shape)
}

/// Ones down the diagonal and zeros elsewhere, over no stored data: the ones
/// are the neutral element of a product, the zeros that of a sum.
pub fn eye_rect(
    rows: usize,
    cols: usize,
) -> ReductionTensor<Sum, ReductionTensor<Product, Physical, EmptyMapping>, Diagonal> {
    let extent = rows.min(cols);
    let diagonal = Diagonal::new(HRect::new(vec![rows, cols]));
    compose(
        Sum::new(),
        compose(multiset_lift(ones(vec![extent])), diagonal),
    )
}

pub fn eye(
    n: usize,
) -> ReductionTensor<Sum, ReductionTensor<Product, Physical, EmptyMapping>, Diagonal> {
    eye_rect(n, n)
}

/// A sparse tensor of shape `size` holding `values` at the coordinates in
/// `indices`, and zero everywhere else. As in torch, `indices` has one row per
/// dimension, so column `k` is the coordinate of `values[k]`, and values at a
/// repeated coordinate add up.
pub fn sparse_coo_tensor<A: Clone, U: BoxedTensor<A>>(
    indices: Vec<Vec<usize>>,
    values: U,
    size: Vec<usize>,
) -> ReductionTensor<Sum<A>, U, COO>
where
    SumReduction: Reduction<A>,
{
    assert_eq!(
        indices.len(),
        size.len(),
        "sparse_coo_tensor needs one row of indices per dimension of size"
    );
    assert_eq!(
        values.hrect().ndim(),
        1,
        "sparse_coo_tensor takes a vector of values"
    );
    let count = values.hrect().shape()[0];
    for row in &indices {
        assert_eq!(
            row.len(),
            count,
            "sparse_coo_tensor needs a coordinate for each of its {count} values"
        );
    }
    let coordinates: Vec<Vec<usize>> = (0..count)
        .map(|k| indices.iter().map(|row| row[k]).collect())
        .collect();
    for coordinate in &coordinates {
        assert!(
            coordinate
                .iter()
                .zip(&size)
                .all(|(coord, extent)| coord < extent),
            "coordinate {coordinate:?} lies outside size {size:?}"
        );
    }
    let coo = COO::new(HRect::new(size), coordinates);
    compose(Sum::new(), compose(multiset_lift(values), coo))
}

/// A sparse latticed tensor of shape `shape`: the value at index `x` of
/// `physical` sits at `basis·x + offset`, and the tensor is zero off that
/// lattice. `basis` has a row per dimension of `shape` and a column per
/// dimension of `physical`. Values the lattice sends to one index add up.
pub fn sparse_latticed_tensor<A: Clone, U: BoxedTensor<A>>(
    physical: U,
    basis: Vec<Vec<i64>>,
    offset: Vec<i64>,
    shape: Vec<usize>,
) -> ReductionTensor<Sum<A>, U, InvAffine>
where
    SumReduction: Reduction<A>,
{
    let affine = Affine::new(physical.hrect().clone(), basis, offset, HRect::new(shape));
    compose(
        Sum::new(),
        compose(multiset_lift(physical), InvAffine::new(affine)),
    )
}

pub fn sum<A: Clone, U: BoxedTensor<A>>(
    input: U,
    dim: usize,
) -> ReductionTensor<Sum<A>, U, InvProjection>
where
    SumReduction: Reduction<A>,
{
    let rows = InvProjection::new(input.hrect().clone(), dim);
    compose(Sum::new(), compose(multiset_lift(input), rows))
}

pub fn prod<A: Clone, U: BoxedTensor<A>>(
    input: U,
    dim: usize,
) -> ReductionTensor<Product<A>, U, InvProjection>
where
    ProductReduction: Reduction<A>,
{
    let rows = InvProjection::new(input.hrect().clone(), dim);
    compose(Product::new(), compose(multiset_lift(input), rows))
}

/// The tensors `mm` multiplies and then sums, stacked: the product of the two
/// matrices at each `[m, n, p]`.
type Products<A> = ReductionTensor<Product<A>, Compose<JaggedTensor<A>, Concat>, InvProjection>;

pub fn mm<A: Clone + 'static, U: BoxedTensor<A> + 'static, V: BoxedTensor<A> + 'static>(
    mat1: U,
    mat2: V,
) -> ReductionTensor<Sum<A>, Products<A>, InvProjection>
where
    SumReduction: Reduction<A>,
    ProductReduction: Reduction<A>,
{
    assert_eq!(mat1.hrect().shape().len(), 2, "mat1 must be a matrix");
    assert_eq!(mat2.hrect().shape().len(), 2, "mat2 must be a matrix");
    assert_eq!(
        mat1.hrect().shape()[1],
        mat2.hrect().shape()[0],
        "mat1 must have as many columns as mat2 has rows"
    );

    let m = mat1.hrect().shape()[0];
    let n = mat1.hrect().shape()[1];
    let p = mat2.hrect().shape()[1];
    let latent_domain = HRect::new(vec![m, n, p]);
    let stacked_latent_domain = HRect::new(vec![2, m, n, p]);

    let phi1 = Projection::new(latent_domain.clone(), 2); // phi1(m, n, p) = [m, n]^T
    let phi2 = Projection::new(latent_domain.clone(), 0); // phi2(m, n, p) = [n, p]^T
    let latent_inv_proj = InvProjection::new(stacked_latent_domain, 0); // l(2, m, n, p) = [m, n, p]^T
    let psi_inv = InvProjection::new(latent_domain.clone(), 1); // psi(m, n, p) = [m, p]^T

    let latent_mats: Vec<Rc<dyn BoxedTensor<A>>> = vec![
        Rc::new(compose(mat1, phi1)), // shape [m, n, p]
        Rc::new(compose(mat2, phi2)), // shape [m, n, p]
    ];
    let stacked = stack(latent_mats, 0); // shape [2, m, n, p]

    let prod_tensor = compose(
        Product::new(),
        compose(multiset_lift(stacked), latent_inv_proj),
    ); // shape [m, n, p]

    compose(Sum::new(), compose(multiset_lift(prod_tensor), psi_inv)) // shape [m, p]
}
