//! Named constructors for the compositions the library can already express.
//! Everything here is assembled from `mapping` and `tensor`; none of it adds
//! machinery of its own.

use std::assert_eq;
use std::rc::Rc;

use crate::display::BoxedTensor;
use crate::domain::HRect;
use crate::mapping::{
    Affine, COO, Concat, Diagonal, EmptyMapping, InvAffine, InvProjection, Permutation, Projection,
    Squeeze, Unsqueeze, Where,
};
use crate::tensor::{
    JaggedTensor, LeftComposition, Physical, Product, Reduction, ReductionTensor, RightComposition,
    Sum, Tensor,
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

/// Checking the bounds here rather than leaving it to `f32::clamp` reports the
/// mistake where it is made, not at the first read, and rules out the NaN
/// bounds that would panic there too.
pub fn clamp<T: Tensor + ?Sized>(u: Rc<T>, low: f32, high: f32) -> LeftComposition<T> {
    assert!(low <= high, "clamp needs low <= high, got {low} and {high}");
    LeftComposition::new(u, move |x| x.clamp(low, high))
}

/// Reorders the axes of a tensor: `axes[i]` is the axis of `u` that becomes
/// axis `i` of the result.
pub fn permute<T: BoxedTensor + 'static>(
    u: Rc<T>,
    axes: Vec<usize>,
) -> RightComposition<Permutation> {
    let codomain = u.hrect().clone();
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

/// Inserts an axis of extent 1 at `dim`, which can be anywhere from before the
/// first axis to after the last.
pub fn unsqueeze<T: BoxedTensor + 'static>(u: Rc<T>, dim: usize) -> RightComposition<Unsqueeze> {
    let codomain = u.hrect().clone();
    RightComposition::new(u, Rc::new(Unsqueeze::new(codomain, dim)))
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
pub fn squeeze<T: BoxedTensor + 'static>(u: Rc<T>, dims: impl Dims) -> RightComposition<Squeeze> {
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
    RightComposition::new(u, Rc::new(Squeeze::new(codomain, removed)))
}

/// Joins tensors end to end along `dim`; their other extents must agree.
pub fn concatenate(tensors: Vec<Rc<dyn BoxedTensor>>, dim: usize) -> RightComposition<Concat> {
    assert!(!tensors.is_empty(), "concatenate needs at least one tensor");
    let underlying = JaggedTensor::new(tensors);
    let codomain = underlying.jagged_domain().clone();
    RightComposition::new(Rc::new(underlying), Rc::new(Concat::new(codomain, dim)))
}

/// Joins tensors of one shape along a new axis at `dim`, by concatenating them
/// once each has an axis of extent 1 there.
pub fn stack(tensors: Vec<Rc<dyn BoxedTensor>>, dim: usize) -> RightComposition<Concat> {
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
            Rc::new(RightComposition::new(tensor, Rc::new(unsqueeze))) as Rc<dyn BoxedTensor>
        })
        .collect();
    concatenate(unsqueezed, dim)
}

/// Reads `input` where `condition` is not zero and `other` where it is. torch's
/// `where` takes a boolean condition; bten's tensors hold floats, so zero stands
/// for false. `where` is a Rust keyword, hence the underscore.
pub fn where_(
    condition: Rc<dyn BoxedTensor>,
    input: Rc<dyn BoxedTensor>,
    other: Rc<dyn BoxedTensor>,
) -> RightComposition<Where> {
    let underlying = JaggedTensor::new(vec![input, other]);
    let codomain = underlying.jagged_domain().clone();
    RightComposition::new(
        Rc::new(underlying),
        Rc::new(Where::new(codomain, condition)),
    )
}

/// Spreads a vector over the diagonal of a tensor of the given rank: rank two
/// gives a diagonal matrix, rank three a tensor whose only non-zero indices
/// are of the form `[i, i, i]`.
pub fn diagonalize<T: BoxedTensor + 'static>(u: Rc<T>, rank: usize) -> Sum<Diagonal> {
    assert_eq!(u.domain().ndim(), 1, "diagonalize takes a vector");
    assert!(rank > 0, "a diagonal tensor has at least one dimension");
    let extent = u.hrect().shape()[0];
    Sum::new(u, Rc::new(Diagonal::new(HRect::new(vec![extent; rank]))))
}

/// Reducing over the empty set gives the reduction's neutral element at every
/// index, so a constant tensor needs no underlying data at all.
fn neutral<R: Reduction>(shape: Vec<usize>) -> ReductionTensor<R, EmptyMapping> {
    let mapping = EmptyMapping::new(HRect::new(shape), HRect::new(vec![0]));
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
        Rc::new(Diagonal::new(HRect::new(vec![rows, cols]))),
    )
}

pub fn eye(n: usize) -> Sum<Diagonal> {
    eye_rect(n, n)
}

/// A sparse tensor of shape `size` holding `values` at the coordinates in
/// `indices`, and zero everywhere else. As in torch, `indices` has one row per
/// dimension, so column `k` is the coordinate of `values[k]`, and values at a
/// repeated coordinate add up.
pub fn sparse_coo_tensor(
    indices: Vec<Vec<usize>>,
    values: Rc<dyn BoxedTensor>,
    size: Vec<usize>,
) -> Sum<COO> {
    assert_eq!(
        indices.len(),
        size.len(),
        "sparse_coo_tensor needs one row of indices per dimension of size"
    );
    assert_eq!(
        values.domain().ndim(),
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
    Sum::new(values, Rc::new(COO::new(HRect::new(size), coordinates)))
}

/// A sparse latticed tensor of shape `shape`: the value at index `x` of
/// `physical` sits at `basis·x + offset`, and the tensor is zero off that
/// lattice. `basis` has a row per dimension of `shape` and a column per
/// dimension of `physical`. Values the lattice sends to one index add up.
pub fn sparse_latticed_tensor<T: BoxedTensor + 'static>(
    physical: Rc<T>,
    basis: Vec<Vec<i64>>,
    offset: Vec<i64>,
    shape: Vec<usize>,
) -> Sum<InvAffine> {
    let affine = Affine::new(physical.hrect().clone(), basis, offset, HRect::new(shape));
    Sum::new(physical, Rc::new(InvAffine::new(affine)))
}

pub fn sum<T: BoxedTensor + 'static>(input: Rc<T>, dim: usize) -> Sum<InvProjection> {
    let codomain = input.hrect().clone();
    Sum::new(input, Rc::new(InvProjection::new(codomain, dim)))
}

pub fn prod<T: BoxedTensor + 'static>(input: Rc<T>, dim: usize) -> Product<InvProjection> {
    let codomain = input.hrect().clone();
    Product::new(input, Rc::new(InvProjection::new(codomain, dim)))
}

pub fn mm<T1: BoxedTensor + 'static, T2: BoxedTensor + 'static>(
    mat1: Rc<T1>,
    mat2: Rc<T2>,
) -> Sum<InvProjection> {
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

    let phi1 = Rc::new(Projection::new(latent_domain.clone(), 2)); // phi1(m, n, p) = [m, n]^T
    let phi2 = Rc::new(Projection::new(latent_domain.clone(), 0)); // phi2(m, n, p) = [n, p]^T
    let latent_inv_proj = Rc::new(InvProjection::new(stacked_latent_domain, 0)); // l(2, m, n, p) = [m, n, p]^T
    let psi_inv = Rc::new(InvProjection::new(latent_domain.clone(), 1)); // psi(m, n, p) = [m, p]^T

    let latent_mat1 = Rc::new(RightComposition::new(mat1, phi1)); // shape [m, n, p]
    let latent_mat2 = Rc::new(RightComposition::new(mat2, phi2)); // shape [m, n, p]
    let stacked = Rc::new(stack(vec![latent_mat1, latent_mat2], 0)); // shape [2, m, n, p]

    let prod_tensor = Rc::new(Product::new(stacked, latent_inv_proj)); // shape [m, n, p]

    Sum::new(prod_tensor, psi_inv) // shape [m, p]
}
