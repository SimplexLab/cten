//! The domain and codomain an index function runs between.

use crate::sets::HRect;
use crate::sets::Set;

/// The domain and codomain an index function runs between.
pub struct Signature<D: Set, C: Set> {
    domain: D,
    codomain: C,
}

impl<D: Set, C: Set> Signature<D, C> {
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

/// The signature of an index function between two hyperrectangles.
pub type HRectSignature = Signature<HRect, HRect>;
