//! The domain and codomain a mapping runs between, and the traits through which
//! a mapping from a hrect gets its accessors and printability from them.

use crate::domain::{Domain, HRect};
use crate::mapping::{BoxedMapping, BoxedPowersetMapping, IndexMapping, Powerset, PowersetMapping};

/// The domain and codomain a mapping runs between.
pub struct Signature<D: Domain, C: Domain> {
    domain: D,
    codomain: C,
}

impl<D: Domain, C: Domain> Signature<D, C> {
    pub fn new(domain: D, codomain: C) -> Self {
        Self { domain, codomain }
    }

    pub fn domain(&self) -> &D {
        &self.domain
    }

    pub fn codomain(&self) -> &C {
        &self.codomain
    }
}

/// The signature of a mapping between two hyperrectangles.
pub type HRectSignature = Signature<HRect, HRect>;

/// What a mapping from a hyperrectangle has to provide. Implementing this rather
/// than `IndexMapping` gets the domain accessors and printability below for
/// free, whatever the codomain; a mapping over a domain that is not a hrect
/// implements `IndexMapping` directly instead, and is not printable.
pub trait HRectMapping {
    type Codomain: Domain;

    fn map(&self, index: &[usize]) -> Vec<usize>;

    fn signature(&self) -> &Signature<HRect, Self::Codomain>;
}

impl<M: HRectMapping> IndexMapping for M {
    fn map(&self, index: &[usize]) -> Vec<usize> {
        HRectMapping::map(self, index)
    }

    fn domain(&self) -> &dyn Domain {
        self.signature().domain()
    }

    fn codomain(&self) -> &dyn Domain {
        self.signature().codomain()
    }
}

impl<M: HRectMapping> BoxedMapping for M {
    fn hrect(&self) -> &HRect {
        self.signature().domain()
    }
}

/// The powerset counterpart of `HRectMapping`.
pub trait HRectPowersetMapping {
    type Codomain: Domain;

    fn map(&self, index: &[usize]) -> Powerset;

    fn signature(&self) -> &Signature<HRect, Self::Codomain>;
}

impl<M: HRectPowersetMapping> PowersetMapping for M {
    fn map(&self, index: &[usize]) -> Powerset {
        HRectPowersetMapping::map(self, index)
    }

    fn domain(&self) -> &dyn Domain {
        self.signature().domain()
    }

    fn codomain(&self) -> &dyn Domain {
        self.signature().codomain()
    }
}

impl<M: HRectPowersetMapping> BoxedPowersetMapping for M {
    fn hrect(&self) -> &HRect {
        self.signature().domain()
    }
}
