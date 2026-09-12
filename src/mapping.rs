use std::collections::HashSet;

use crate::domain::{Domain, HRect};

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
    fn hrect_domain(&self) -> &HRect;
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
    fn hrect_domain(&self) -> &HRect;
}

pub struct Stride {
    domain: HRect,
    codomain: HRect,
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
            domain,
            codomain,
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

impl IndexMapping for Stride {
    fn map(&self, index: &[usize]) -> Vec<usize> {
        let mut result = 0;
        for (coord, stride) in index.iter().zip(&self.strides) {
            result += coord * stride;
        }
        vec![result]
    }

    fn domain(&self) -> &dyn Domain {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain {
        &self.codomain
    }
}

impl BoxedMapping for Stride {
    fn hrect_domain(&self) -> &HRect {
        &self.domain
    }
}

pub struct Permutation {
    domain: HRect,
    codomain: HRect,
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
            domain: HRect::new(shape),
            codomain,
            axes,
        }
    }

    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
}

impl IndexMapping for Permutation {
    fn map(&self, index: &[usize]) -> Vec<usize> {
        let mut result = vec![0; self.axes.len()];
        for (i, &axis) in self.axes.iter().enumerate() {
            result[axis] = index[i];
        }
        result
    }

    fn domain(&self) -> &dyn Domain {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain {
        &self.codomain
    }
}

impl BoxedMapping for Permutation {
    fn hrect_domain(&self) -> &HRect {
        &self.domain
    }
}

pub struct EmptyMapping {
    domain: HRect,
    codomain: HRect,
}

impl EmptyMapping {
    pub fn new(domain: HRect, codomain: HRect) -> Self {
        Self { domain, codomain }
    }
}

impl PowersetMapping for EmptyMapping {
    fn map(&self, _index: &[usize]) -> Powerset {
        Powerset::new()
    }

    fn domain(&self) -> &dyn Domain {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain {
        &self.codomain
    }
}

impl BoxedPowersetMapping for EmptyMapping {
    fn hrect_domain(&self) -> &HRect {
        &self.domain
    }
}

pub struct COO {
    domain: HRect,
    codomain: HRect,
    inverse_coords: Vec<Vec<usize>>,
}

impl COO {
    pub fn new(domain: HRect, inverse_coords: Vec<Vec<usize>>) -> Self {
        let codomain = HRect::new(vec![inverse_coords.len()]);
        Self {
            domain,
            codomain,
            inverse_coords,
        }
    }

    pub fn inverse_coords(&self) -> &[Vec<usize>] {
        &self.inverse_coords
    }
}

impl PowersetMapping for COO {
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

    fn domain(&self) -> &dyn Domain {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain {
        &self.codomain
    }
}

impl BoxedPowersetMapping for COO {
    fn hrect_domain(&self) -> &HRect {
        &self.domain
    }
}

pub struct Diagonal {
    domain: HRect,
    codomain: HRect,
}

impl Diagonal {
    pub fn new(domain: HRect) -> Self {
        // Only indices repeated along every dimension are hit, so the diagonal
        // is as long as the shortest dimension.
        let extent = domain.shape().iter().copied().min().unwrap_or(0);
        Self {
            domain,
            codomain: HRect::new(vec![extent]),
        }
    }
}

impl PowersetMapping for Diagonal {
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

    fn domain(&self) -> &dyn Domain {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain {
        &self.codomain
    }
}

impl BoxedPowersetMapping for Diagonal {
    fn hrect_domain(&self) -> &HRect {
        &self.domain
    }
}
