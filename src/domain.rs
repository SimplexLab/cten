use std::marker::PhantomData;

use crate::index::Coords;

/// A set of indices. Tensors are defined on a domain, and mappings go from a
/// domain to a codomain.
pub trait Domain<U: Coords> {
    fn contains(&self, index: &U) -> bool;
}

/// A hyperrectangle of indices: coordinate `i` runs over `0..shape[i]`. This is
/// the only concrete domain for now, and the only one a tensor can be printed
/// over.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BoxDomain<U: Coords> {
    shape: Vec<usize>,
    _marker: PhantomData<fn() -> U>,
}

impl<U: Coords> BoxDomain<U> {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            _marker: PhantomData,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<U: Coords> Domain<U> for BoxDomain<U> {
    fn contains(&self, index: &U) -> bool {
        index.ndim() == self.shape.len()
            && (0..self.shape.len()).all(|i| index.at(i) < self.shape[i])
    }
}
