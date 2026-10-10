//! The domain and codomain a mapping runs between.

use crate::domain::{Domain, HRect};

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
