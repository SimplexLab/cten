//! Named constructors for the compositions the library can already express.
//! Everything here is assembled from `mapping` and `tensor`; none of it adds
//! machinery of its own.

use std::rc::Rc;

use crate::display::BoxedTensor;
use crate::domain::BoxDomain;
use crate::mapping::{Diagonal, EmptyMapping, Permutation};
use crate::tensor::{
    LeftComposition, Physical, Product, Reduction, ReductionTensor, RightComposition, Sum, Tensor,
};

pub fn square<T: Tensor + ?Sized>(u: Rc<T>) -> LeftComposition<T> {
    LeftComposition::new(u, |x| x * x)
}

pub fn negate<T: Tensor + ?Sized>(u: Rc<T>) -> LeftComposition<T> {
    LeftComposition::new(u, |x| -x)
}

pub fn scale<T: Tensor + ?Sized>(u: Rc<T>, factor: f32) -> LeftComposition<T> {
    LeftComposition::new(u, move |x| x * factor)
}

pub fn pow<T: Tensor + ?Sized>(u: Rc<T>, exponent: f32) -> LeftComposition<T> {
    LeftComposition::new(u, move |x| x.powf(exponent))
}

pub fn clamp<T: Tensor + ?Sized>(u: Rc<T>, low: f32, high: f32) -> LeftComposition<T> {
    assert!(low <= high, "the low bound must not exceed the high one");
    LeftComposition::new(u, move |x| x.clamp(low, high))
}

/// Reorders the axes of a tensor: `axes[i]` is the axis of `u` that becomes
/// axis `i` of the result.
pub fn permute<T: BoxedTensor + 'static>(
    u: Rc<T>,
    axes: Vec<usize>,
) -> RightComposition<Permutation> {
    let codomain = u.box_domain().clone();
    RightComposition::new(u, Rc::new(Permutation::new(codomain, axes)))
}

/// Swaps the last two axes and leaves any leading ones alone, so a stack of
/// matrices transposes matrix by matrix. At rank two this is the plain
/// transpose.
pub fn transpose<T: BoxedTensor + 'static>(u: Rc<T>) -> RightComposition<Permutation> {
    let ndim = u.domain().ndim();
    assert!(ndim >= 2, "transpose needs at least two dimensions");
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.swap(ndim - 2, ndim - 1);
    permute(u, axes)
}

/// Moves the axis at `source` to `destination`; the axes left over keep their
/// relative order.
pub fn movedim<T: BoxedTensor + 'static>(
    u: Rc<T>,
    source: usize,
    destination: usize,
) -> RightComposition<Permutation> {
    let ndim = u.domain().ndim();
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

/// Spreads a vector over the diagonal of a tensor of the given rank: rank two
/// gives a diagonal matrix, rank three a tensor whose only non-zero indices
/// are of the form `[i, i, i]`.
pub fn diagonalize<T: BoxedTensor + 'static>(u: Rc<T>, rank: usize) -> Sum<Diagonal> {
    assert_eq!(u.domain().ndim(), 1, "diagonalize takes a vector");
    assert!(rank > 0, "a diagonal tensor has at least one dimension");
    let extent = u.box_domain().shape()[0];
    Sum::new(
        u,
        Rc::new(Diagonal::new(BoxDomain::new(vec![extent; rank]))),
    )
}

/// Reducing over the empty set gives the reduction's neutral element at every
/// index, so a constant tensor needs no underlying data at all.
fn neutral<R: Reduction>(shape: Vec<usize>) -> ReductionTensor<R, EmptyMapping> {
    let mapping = EmptyMapping::new(BoxDomain::new(shape), BoxDomain::new(vec![0]));
    ReductionTensor::new(Rc::new(Physical::new(vec![])), Rc::new(mapping))
}

pub fn zeros(shape: Vec<usize>) -> Sum<EmptyMapping> {
    neutral(shape)
}

pub fn ones(shape: Vec<usize>) -> Product<EmptyMapping> {
    neutral(shape)
}

/// Ones down the diagonal and zeros elsewhere, over no stored data: the ones
/// are the neutral element of a product, the zeros that of a sum.
pub fn eye_rect(rows: usize, cols: usize) -> Sum<Diagonal> {
    let extent = rows.min(cols);
    Sum::new(
        Rc::new(ones(vec![extent])),
        Rc::new(Diagonal::new(BoxDomain::new(vec![rows, cols]))),
    )
}

pub fn eye(n: usize) -> Sum<Diagonal> {
    eye_rect(n, n)
}
