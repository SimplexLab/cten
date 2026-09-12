/// A set of indices, an index being a vector of coordinates. Tensors are
/// defined on a domain, and mappings go from a domain to a codomain.
pub trait Domain {
    /// Number of coordinates the indices of this domain have.
    fn ndim(&self) -> usize;

    fn contains(&self, index: &[usize]) -> bool;
}

/// A hyperrectangle of indices: coordinate `i` runs over `0..shape[i]`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct HRect {
    shape: Vec<usize>,
}

impl HRect {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Domain for HRect {
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn contains(&self, index: &[usize]) -> bool {
        index.len() == self.shape.len()
            && index
                .iter()
                .zip(&self.shape)
                .all(|(coord, extent)| coord < extent)
    }
}
