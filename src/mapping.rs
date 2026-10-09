use std::collections::HashSet;
use std::rc::Rc;

use crate::display::BoxedTensor;
use crate::domain::{Domain, HRect, JaggedDomain};
use crate::signature::{HRectMapping, HRectPowersetMapping, HRectSignature, Signature};

/// A set of indices, as returned by a `PowersetMapping`.
pub type Powerset = HashSet<Vec<usize>>;

/// A mapping between two index sets, from its domain to its codomain.
pub trait IndexMapping {
    fn map(&self, index: &[usize]) -> Vec<usize>;

    fn domain(&self) -> &dyn Domain;

    fn codomain(&self) -> &dyn Domain;
}

/// A mapping whose domain is a hrect. Composing a tensor over one of these gives
/// a boxed tensor, hence a printable one.
pub trait BoxedMapping: IndexMapping {
    fn hrect(&self) -> &HRect;
}

/// A mapping sending each index of its domain to a set of indices of its
/// codomain. Note that the codomain is the set the returned indices are drawn
/// from, not its powerset.
pub trait PowersetMapping {
    fn map(&self, index: &[usize]) -> Powerset;

    fn domain(&self) -> &dyn Domain;

    fn codomain(&self) -> &dyn Domain;
}

/// A powerset mapping whose domain is a hrect. Reducing over one of these gives a
/// boxed tensor, hence a printable one.
pub trait BoxedPowersetMapping: PowersetMapping {
    fn hrect(&self) -> &HRect;
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

impl HRectMapping for Stride {
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
    /// mappings above: the shape being permuted is what is known, and the
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

impl HRectMapping for Permutation {
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

impl HRectMapping for Unsqueeze {
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

impl HRectMapping for Squeeze {
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

impl HRectMapping for Concat {
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

/// Reads the first of two tensors where `condition` is not zero and the second
/// where it is. Unlike the mappings above, where an index is sent depends on a
/// tensor's values and not only on the index itself.
pub struct Where {
    signature: Signature<HRect, JaggedDomain>,
    condition: Rc<dyn BoxedTensor>,
}

impl Where {
    /// Takes the codomain, the two tensors picked from. The domain is their
    /// common shape, which `condition` must share.
    pub fn new(codomain: JaggedDomain, condition: Rc<dyn BoxedTensor>) -> Self {
        let parts = codomain.parts();
        assert_eq!(parts.len(), 2, "where picks from exactly two tensors");
        let shape = parts[0].clone();
        assert!(
            parts[1] == shape && *condition.hrect() == shape,
            "where needs its condition and both tensors to have one shape, got {:?}, {:?} and {:?}",
            condition.hrect().shape(),
            parts[0].shape(),
            parts[1].shape()
        );
        Self {
            signature: Signature::new(shape, codomain),
            condition,
        }
    }
}

impl HRectMapping for Where {
    type Codomain = JaggedDomain;

    fn map(&self, index: &[usize]) -> Vec<usize> {
        // Zero reads as false and anything else, NaN included, as true.
        let part = if self.condition.get(index) != 0.0 {
            0
        } else {
            1
        };
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
    signature: HRectSignature,
}

impl EmptyMapping {
    pub fn new(domain: HRect, codomain: HRect) -> Self {
        Self {
            signature: HRectSignature::new(domain, codomain),
        }
    }
}

impl HRectPowersetMapping for EmptyMapping {
    type Codomain = HRect;

    fn map(&self, _index: &[usize]) -> Powerset {
        Powerset::new()
    }

    fn signature(&self) -> &HRectSignature {
        &self.signature
    }
}

pub struct COO {
    signature: HRectSignature,
    inverse_coords: Vec<Vec<usize>>,
}

impl COO {
    pub fn new(domain: HRect, inverse_coords: Vec<Vec<usize>>) -> Self {
        let codomain = HRect::new(vec![inverse_coords.len()]);
        Self {
            signature: HRectSignature::new(domain, codomain),
            inverse_coords,
        }
    }

    pub fn inverse_coords(&self) -> &[Vec<usize>] {
        &self.inverse_coords
    }
}

impl HRectPowersetMapping for COO {
    type Codomain = HRect;

    fn map(&self, index: &[usize]) -> Powerset {
        let mut result = Powerset::new();
        for (j, coords) in self.inverse_coords.iter().enumerate() {
            let matches = coords
                .iter()
                .enumerate()
                .all(|(i, coord)| *coord == index[i]);
            if matches {
                result.insert(vec![j]);
            }
        }
        result
    }

    fn signature(&self) -> &HRectSignature {
        &self.signature
    }
}

pub struct Diagonal {
    signature: HRectSignature,
}

impl Diagonal {
    pub fn new(domain: HRect) -> Self {
        // Only indices repeated along every dimension are hit, so the diagonal
        // is as long as the shortest dimension.
        let extent = domain.shape().iter().copied().min().unwrap_or(0);
        Self {
            signature: HRectSignature::new(domain, HRect::new(vec![extent])),
        }
    }
}

impl HRectPowersetMapping for Diagonal {
    type Codomain = HRect;

    fn map(&self, index: &[usize]) -> Powerset {
        let mut result = Powerset::new();
        let first = index[0];
        for coord in index {
            if *coord != first {
                return result;
            }
        }
        result.insert(vec![first]);
        result
    }

    fn signature(&self) -> &HRectSignature {
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

impl HRectMapping for Projection {
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
    signature: HRectSignature,
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
            signature: HRectSignature::new(HRect::new(shape), codomain),
            dim,
        }
    }
}

impl HRectPowersetMapping for InvProjection {
    type Codomain = HRect;

    fn map(&self, index: &[usize]) -> Powerset {
        (0..self.signature.codomain().shape()[self.dim])
            .map(|coord| {
                let mut result = index.to_vec();
                result.insert(self.dim, coord);
                result
            })
            .collect()
    }

    fn signature(&self) -> &HRectSignature {
        &self.signature
    }
}
