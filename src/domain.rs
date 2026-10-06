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

/// The disjoint union of several hrects: `[t, rest..]` belongs to it when `rest`
/// is in hrect `t`. This is the index set of a jagged tensor, whose parts may
/// have different shapes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct JaggedDomain {
    parts: Vec<HRect>,
}

impl JaggedDomain {
    pub fn new(parts: Vec<HRect>) -> Self {
        assert!(!parts.is_empty(), "a jagged domain needs at least one part");
        let ndim = parts[0].ndim();
        assert!(
            parts.iter().all(|part| part.ndim() == ndim),
            "the parts of a jagged domain must all have the same number of coordinates"
        );
        Self { parts }
    }

    pub fn parts(&self) -> &[HRect] {
        &self.parts
    }
}

impl Domain for JaggedDomain {
    fn ndim(&self) -> usize {
        1 + self.parts[0].ndim()
    }

    fn contains(&self, index: &[usize]) -> bool {
        match index.split_first() {
            Some((&part, rest)) => part < self.parts.len() && self.parts[part].contains(rest),
            None => false,
        }
    }
}
