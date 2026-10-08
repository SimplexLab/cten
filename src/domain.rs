use std::rc::Rc;

/// A set of indices, an index being a vector of coordinates. Tensors are
/// defined on a domain, and mappings go from a domain to a codomain.
pub trait Domain {
    /// Number of coordinates the indices of this domain have, or `None` when
    /// they don't all have the same number, as in a multi-domain whose parts
    /// differ in rank.
    fn ndim(&self) -> Option<usize>;

    fn contains(&self, index: &[usize]) -> bool;

    /// A shared copy of this domain. A tensor only lends out its domain, so this
    /// is how a domain built over other tensors' domains holds on to them.
    fn to_rc(&self) -> Rc<dyn Domain>;
}

impl Domain for Rc<dyn Domain> {
    fn ndim(&self) -> Option<usize> {
        (**self).ndim()
    }

    fn contains(&self, index: &[usize]) -> bool {
        (**self).contains(index)
    }

    fn to_rc(&self) -> Rc<dyn Domain> {
        self.clone()
    }
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
    fn ndim(&self) -> Option<usize> {
        Some(self.shape.len())
    }

    fn contains(&self, index: &[usize]) -> bool {
        index.len() == self.shape.len()
            && index
                .iter()
                .zip(&self.shape)
                .all(|(coord, extent)| coord < extent)
    }

    fn to_rc(&self) -> Rc<dyn Domain> {
        Rc::new(self.clone())
    }
}

/// The disjoint union of several domains: `[t, rest..]` belongs to it when
/// `rest` is in part `t`. This is the index set of a multi-tensor. The parts may
/// differ in shape and even in rank, in which case its indices differ in length
/// too.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MultiDomain<D> {
    parts: Vec<D>,
}

impl<D: Domain> MultiDomain<D> {
    pub fn new(parts: Vec<D>) -> Self {
        assert!(!parts.is_empty(), "a multi-domain needs at least one part");
        Self { parts }
    }

    pub fn parts(&self) -> &[D] {
        &self.parts
    }
}

impl<D: Domain + Clone + 'static> Domain for MultiDomain<D> {
    fn ndim(&self) -> Option<usize> {
        let rank = self.parts[0].ndim()?;
        self.parts
            .iter()
            .all(|part| part.ndim() == Some(rank))
            .then_some(1 + rank)
    }

    fn contains(&self, index: &[usize]) -> bool {
        match index.split_first() {
            Some((&part, rest)) => part < self.parts.len() && self.parts[part].contains(rest),
            None => false,
        }
    }

    fn to_rc(&self) -> Rc<dyn Domain> {
        Rc::new(self.clone())
    }
}

/// A multi-domain of hrects, as tensors joined end to end are read from.
pub type JaggedDomain = MultiDomain<HRect>;
