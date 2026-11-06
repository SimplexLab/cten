use std::rc::Rc;

use crate::function::Function;
use crate::lattice::Preimage;
use crate::multiset::Multiset;
use crate::sets::{HRect, IndexSet, JaggedDomain, Multisets, Set};
use crate::signature::{HRectSignature, Signature};
use crate::tensor::Tensor;

/// The signature of a multi-index function: from a hrect to the multisets of
/// indices of another.
pub type MultiIndexFunctionSignature = Signature<HRect, Multisets<HRect>>;

/// A function on the indices of a hrect, sending each index to an index of its
/// codomain, or, for a multi-index function, to a multiset of them.
/// Implementing this gets `Function` for free, with the domain and codomain of
/// the signature.
pub trait IndexFunction {
    type Codomain: Set;

    fn map(&self, index: &[usize]) -> <Self::Codomain as Set>::Owned;

    fn signature(&self) -> &Signature<HRect, Self::Codomain>;
}

impl<M: IndexFunction> Function for M {
    type Domain = HRect;
    type Codomain = M::Codomain;

    fn apply(&self, index: &[usize]) -> <M::Codomain as Set>::Owned {
        self.map(index)
    }

    fn domain(&self) -> HRect {
        self.signature().domain().clone()
    }

    fn codomain(&self) -> M::Codomain {
        self.signature().codomain().clone()
    }
}

pub struct Stride {
    signature: HRectSignature,
    strides: Vec<usize>,
}

impl Stride {
    pub fn new(domain: HRect, strides: Vec<usize>) -> Self {
        assert_eq!(
            domain.ndim(),
            strides.len(),
            "a stride per dimension of the domain is required"
        );
        let codomain = HRect::new(vec![flat_extent(domain.shape(), &strides)]);
        Self {
            signature: HRectSignature::new(domain, codomain),
            strides,
        }
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

/// Number of flat positions a strided hrect reaches: one past its largest index.
fn flat_extent(shape: &[usize], strides: &[usize]) -> usize {
    if shape.contains(&0) {
        return 0;
    }
    1 + shape
        .iter()
        .zip(strides)
        .map(|(extent, stride)| (extent - 1) * stride)
        .sum::<usize>()
}

impl IndexFunction for Stride {
    type Codomain = HRect;

    fn map(&self, index: &[usize]) -> Vec<usize> {
        let mut result = 0;
        for (coord, stride) in index.iter().zip(&self.strides) {
            result += coord * stride;
        }
        Vec::from([result])
    }

    fn signature(&self) -> &HRectSignature {
        &self.signature
    }
}

pub struct Permutation {
    signature: HRectSignature,
    axes: Vec<usize>,
}

impl Permutation {
    /// Takes the codomain and derives the domain, the other way round from the
    /// index functions above: the shape being permuted is what is known, and the
    /// permuted shape follows from it. `axes[i]` is the axis of the codomain
    /// that becomes axis `i` of the domain.
    pub fn new(codomain: HRect, axes: Vec<usize>) -> Self {
        assert_eq!(
            axes.len(),
            codomain.ndim(),
            "an axis per dimension of the codomain is required"
        );
        let mut seen = vec![false; axes.len()];
        for &axis in &axes {
            assert!(axis < axes.len(), "axis {axis} is out of range");
            assert!(!seen[axis], "axis {axis} is repeated");
            seen[axis] = true;
        }
        let shape = axes.iter().map(|&axis| codomain.shape()[axis]).collect();
        Self {
            signature: HRectSignature::new(HRect::new(shape), codomain),
            axes,
        }
    }

    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
}

impl IndexFunction for Permutation {
    type Codomain = HRect;

    fn map(&self, index: &[usize]) -> Vec<usize> {
        let mut result = vec![0; self.axes.len()];
        for (i, &axis) in self.axes.iter().enumerate() {
            result[axis] = index[i];
        }
        result
    }

    fn signature(&self) -> &HRectSignature {
        &self.signature
    }
}

pub struct Unsqueeze {
    signature: HRectSignature,
    dim: usize,
}

impl Unsqueeze {
    /// Takes the codomain, as `Permutation` does: the domain is the same shape
    /// with an axis of extent 1 inserted at `dim`.
    pub fn new(codomain: HRect, dim: usize) -> Self {
        assert!(
            dim <= codomain.ndim(),
            "dim {dim} is out of range for a new axis of a {}-dimensional hrect",
            codomain.ndim()
        );
        let mut shape = codomain.shape().to_vec();
        shape.insert(dim, 1);
        Self {
            signature: HRectSignature::new(HRect::new(shape), codomain),
            dim,
        }
    }
}

impl IndexFunction for Unsqueeze {
    type Codomain = HRect;

    fn map(&self, index: &[usize]) -> Vec<usize> {
        let mut result = index.to_vec();
        result.remove(self.dim);
        result
    }

    fn signature(&self) -> &HRectSignature {
        &self.signature
    }
}

pub struct Squeeze {
    signature: HRectSignature,
    removed: Vec<usize>,
}

impl Squeeze {
    /// Takes the codomain: the domain is the same shape without the axes in
    /// `removed`, each of which must have extent 1.
    pub fn new(codomain: HRect, mut removed: Vec<usize>) -> Self {
        // Ascending, so that re-inserting the coordinates in order puts each one
        // back where it came from.
        removed.sort_unstable();
        removed.dedup();
        for &axis in &removed {
            assert!(axis < codomain.ndim(), "axis {axis} is out of range");
            assert_eq!(
                codomain.shape()[axis],
                1,
                "only an axis of extent 1 can be squeezed, axis {axis} has extent {}",
                codomain.shape()[axis]
            );
        }
        let shape = (0..codomain.ndim())
            .filter(|axis| !removed.contains(axis))
            .map(|axis| codomain.shape()[axis])
            .collect();
        Self {
            signature: HRectSignature::new(HRect::new(shape), codomain),
            removed,
        }
    }
}

impl IndexFunction for Squeeze {
    type Codomain = HRect;

    fn map(&self, index: &[usize]) -> Vec<usize> {
        let mut result = index.to_vec();
        for &axis in &self.removed {
            result.insert(axis, 0);
        }
        result
    }

    fn signature(&self) -> &HRectSignature {
        &self.signature
    }
}

/// Reads tensors joined end to end along `dim` out of the jagged tensor holding
/// them. Every other coordinate passes through unchanged.
pub struct Concat {
    signature: Signature<HRect, JaggedDomain>,
    dim: usize,
    starts: Vec<usize>,
}

impl Concat {
    /// Takes the codomain: the domain is the parts' common shape, with their
    /// extents along `dim` added up.
    pub fn new(codomain: JaggedDomain, dim: usize) -> Self {
        let parts = codomain.parts();
        let first = parts[0].shape();
        assert!(dim < first.len(), "dim {dim} is out of range");
        for part in parts {
            let shape = part.shape();
            assert!(
                (0..first.len()).all(|axis| axis == dim || shape[axis] == first[axis]),
                "tensors can only be joined along dim {dim} when their other extents agree, got {first:?} and {shape:?}"
            );
        }
        let mut starts = Vec::with_capacity(parts.len());
        let mut extent = 0;
        for part in parts {
            starts.push(extent);
            extent += part.shape()[dim];
        }
        let mut shape = first.to_vec();
        shape[dim] = extent;
        Self {
            signature: Signature::new(HRect::new(shape), codomain),
            dim,
            starts,
        }
    }
}

impl IndexFunction for Concat {
    type Codomain = JaggedDomain;

    fn map(&self, index: &[usize]) -> Vec<usize> {
        let position = index[self.dim];
        // The last part starting at or before `position`. A part of extent 0
        // shares its start with the next one, so it is never picked.
        let part = self.starts.partition_point(|&start| start <= position) - 1;
        let mut result = Vec::with_capacity(index.len() + 1);
        result.push(part);
        result.extend_from_slice(index);
        result[self.dim + 1] = position - self.starts[part];
        result
    }

    fn signature(&self) -> &Signature<HRect, JaggedDomain> {
        &self.signature
    }
}

/// Reads the first of two tensors where `condition` holds and the second where
/// it does not. Unlike the index functions above, where an index is sent depends on a
/// tensor's values and not only on the index itself.
pub struct Where {
    signature: Signature<HRect, JaggedDomain>,
    condition: Rc<dyn Tensor<bool>>,
}

impl Where {
    /// Takes the codomain, the two tensors picked from. The domain is their
    /// common shape, which `condition` must share.
    pub fn new(codomain: JaggedDomain, condition: Rc<dyn Tensor<bool>>) -> Self {
        let parts = codomain.parts();
        assert_eq!(parts.len(), 2, "where picks from exactly two tensors");
        let shape = parts[0].clone();
        assert!(
            parts[1] == shape && condition.domain() == shape,
            "where needs its condition and both tensors to have one shape, got {:?}, {:?} and {:?}",
            condition.domain().shape(),
            parts[0].shape(),
            parts[1].shape()
        );
        Self {
            signature: Signature::new(shape, codomain),
            condition,
        }
    }
}

impl IndexFunction for Where {
    type Codomain = JaggedDomain;

    fn map(&self, index: &[usize]) -> Vec<usize> {
        let part = if self.condition.apply(index) { 0 } else { 1 };
        let mut result = Vec::with_capacity(index.len() + 1);
        result.push(part);
        result.extend_from_slice(index);
        result
    }

    fn signature(&self) -> &Signature<HRect, JaggedDomain> {
        &self.signature
    }
}

pub struct EmptyMapping {
    signature: MultiIndexFunctionSignature,
}

impl EmptyMapping {
    pub fn new(domain: HRect, codomain: HRect) -> Self {
        Self {
            signature: MultiIndexFunctionSignature::new(domain, Multisets::new(codomain)),
        }
    }
}

impl IndexFunction for EmptyMapping {
    type Codomain = Multisets<HRect>;

    fn map(&self, _index: &[usize]) -> Multiset<Vec<usize>> {
        Multiset::default()
    }

    fn signature(&self) -> &MultiIndexFunctionSignature {
        &self.signature
    }
}

pub struct COO {
    signature: MultiIndexFunctionSignature,
    inverse_coords: Vec<Vec<usize>>,
}

impl COO {
    pub fn new(domain: HRect, inverse_coords: Vec<Vec<usize>>) -> Self {
        let codomain = HRect::new(vec![inverse_coords.len()]);
        Self {
            signature: MultiIndexFunctionSignature::new(domain, Multisets::new(codomain)),
            inverse_coords,
        }
    }

    pub fn inverse_coords(&self) -> &[Vec<usize>] {
        &self.inverse_coords
    }
}

impl IndexFunction for COO {
    type Codomain = Multisets<HRect>;

    fn map(&self, index: &[usize]) -> Multiset<Vec<usize>> {
        let mut result = Multiset::default();
        for (j, coords) in self.inverse_coords.iter().enumerate() {
            let matches = coords
                .iter()
                .enumerate()
                .all(|(i, coord)| *coord == index[i]);
            if matches {
                result.push(vec![j]);
            }
        }
        result
    }

    fn signature(&self) -> &MultiIndexFunctionSignature {
        &self.signature
    }
}

pub struct Diagonal {
    signature: MultiIndexFunctionSignature,
}

impl Diagonal {
    pub fn new(domain: HRect) -> Self {
        // Only indices repeated along every dimension are hit, so the diagonal
        // is as long as the shortest dimension.
        let extent = domain.shape().iter().copied().min().unwrap_or(0);
        Self {
            signature: MultiIndexFunctionSignature::new(
                domain,
                Multisets::new(HRect::new(vec![extent])),
            ),
        }
    }
}

impl IndexFunction for Diagonal {
    type Codomain = Multisets<HRect>;

    fn map(&self, index: &[usize]) -> Multiset<Vec<usize>> {
        let mut result = Multiset::default();
        let first = index[0];
        for coord in index {
            if *coord != first {
                return result;
            }
        }
        result.push(vec![first]);
        result
    }

    fn signature(&self) -> &MultiIndexFunctionSignature {
        &self.signature
    }
}

/// Projection dropping axis `dim`.
/// e.g. with dim=1, f(m, n, p) = [m, p]^T
pub struct Projection {
    signature: HRectSignature,
    dim: usize,
}

impl Projection {
    pub fn new(domain: HRect, dim: usize) -> Self {
        assert!(
            dim < domain.ndim(),
            "dim {dim} is out of range for a {}-dimensional hrect",
            domain.ndim()
        );
        let mut shape = domain.shape().to_vec();
        shape.remove(dim);
        Self {
            signature: HRectSignature::new(domain, HRect::new(shape)),
            dim,
        }
    }
}

impl IndexFunction for Projection {
    type Codomain = HRect;

    fn map(&self, index: &[usize]) -> Vec<usize> {
        let mut result = index.to_vec();
        result.remove(self.dim);
        result
    }

    fn signature(&self) -> &HRectSignature {
        &self.signature
    }
}

/// The inverse of the projection dropping axis `dim`.
/// e.g. with dim=1, f(m, p) = {[m n p]^T | n in \[N\]}
pub struct InvProjection {
    signature: MultiIndexFunctionSignature,
    dim: usize,
}

impl InvProjection {
    /// Takes the codomain: the domain is the same shape without `dim`.
    pub fn new(codomain: HRect, dim: usize) -> Self {
        assert!(
            dim < codomain.ndim(),
            "dim {dim} is out of range for a {}-dimensional hrect",
            codomain.ndim()
        );
        let mut shape = codomain.shape().to_vec();
        shape.remove(dim);
        Self {
            signature: MultiIndexFunctionSignature::new(
                HRect::new(shape),
                Multisets::new(codomain),
            ),
            dim,
        }
    }
}

impl IndexFunction for InvProjection {
    type Codomain = Multisets<HRect>;

    fn map(&self, index: &[usize]) -> Multiset<Vec<usize>> {
        (0..self.signature.codomain().set().shape()[self.dim])
            .map(|coord| {
                let mut result = index.to_vec();
                result.insert(self.dim, coord);
                result
            })
            .collect()
    }

    fn signature(&self) -> &MultiIndexFunctionSignature {
        &self.signature
    }
}

/// The affine map `x ↦ basis·x + offset`. `basis` has a row per dimension of the
/// codomain and a column per dimension of the domain. Its entries and `offset`
/// may be negative, as long as every index of the domain lands in the codomain.
/// Read through this, a tensor gives its values on a lattice, e.g. its diagonal
/// with `basis = [[1], [1]]`.
pub struct Affine {
    signature: HRectSignature,
    basis: Vec<Vec<i64>>,
    offset: Vec<i64>,
}

impl Affine {
    pub fn new(domain: HRect, basis: Vec<Vec<i64>>, offset: Vec<i64>, codomain: HRect) -> Self {
        assert_eq!(
            basis.len(),
            codomain.ndim(),
            "a basis row per dimension of the codomain is required"
        );
        assert!(
            basis.iter().all(|row| row.len() == domain.ndim()),
            "a basis column per dimension of the domain is required"
        );
        assert_eq!(
            offset.len(),
            codomain.ndim(),
            "an offset per dimension of the codomain is required"
        );
        // An empty domain has no index to land anywhere.
        if !domain.shape().contains(&0) {
            for (i, (row, &start)) in basis.iter().zip(&offset).enumerate() {
                // Each coordinate varies on its own, so each term is at its
                // smallest and largest at one end or the other of its range.
                let (lowest, highest) = row.iter().zip(domain.shape()).fold(
                    (start, start),
                    |(low, high), (&step, &extent)| {
                        let end = step * (extent as i64 - 1);
                        (low + end.min(0), high + end.max(0))
                    },
                );
                assert!(
                    0 <= lowest && highest < codomain.shape()[i] as i64,
                    "the lattice reaches {lowest} to {highest} along dimension {i}, which has extent {}",
                    codomain.shape()[i]
                );
            }
        }
        Self {
            signature: HRectSignature::new(domain, codomain),
            basis,
            offset,
        }
    }
}

impl IndexFunction for Affine {
    type Codomain = HRect;

    fn map(&self, index: &[usize]) -> Vec<usize> {
        // Every coordinate is non-negative, as checked when constructing.
        self.basis
            .iter()
            .zip(&self.offset)
            .map(|(row, &start)| {
                let coord = start
                    + row
                        .iter()
                        .zip(index)
                        .map(|(&step, &x)| step * x as i64)
                        .sum::<i64>();
                coord as usize
            })
            .collect()
    }

    fn signature(&self) -> &HRectSignature {
        &self.signature
    }
}

/// The inverse of an `Affine`: sends each index of its codomain back to the
/// indices of its domain that land on it. There are none off the lattice, so a
/// sum over this lays its underlying tensor out on the lattice and is zero
/// elsewhere: a sparse latticed tensor. They are found by solving
/// `basis·x = index - offset`, so a read costs about as much as the basis is
/// large, however large the tensors.
pub struct InvAffine {
    signature: MultiIndexFunctionSignature,
    offset: Vec<i64>,
    preimage: Preimage,
}

impl InvAffine {
    pub fn new(affine: Affine) -> Self {
        let Affine {
            signature,
            basis,
            offset,
        } = affine;
        let preimage = Preimage::new(&basis, signature.domain().shape());
        Self {
            signature: MultiIndexFunctionSignature::new(
                signature.codomain().clone(),
                Multisets::new(signature.domain().clone()),
            ),
            offset,
            preimage,
        }
    }
}

impl IndexFunction for InvAffine {
    type Codomain = Multisets<HRect>;

    fn map(&self, index: &[usize]) -> Multiset<Vec<usize>> {
        let target: Vec<i64> = index
            .iter()
            .zip(&self.offset)
            .map(|(&coord, &start)| coord as i64 - start)
            .collect();
        self.preimage.solve(&target).into_iter().collect()
    }

    fn signature(&self) -> &MultiIndexFunctionSignature {
        &self.signature
    }
}
