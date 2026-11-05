//! The building bricks: functions, their composition, and their extension to
//! multisets. Tensors, index maps, index functions and reductions are all
//! functions, and every operation on them is built from these.

use std::borrow::Borrow;
use std::rc::Rc;

use crate::set::{Multisets, Set, Values};

/// A function from its domain to its codomain, which may be sets of indices,
/// of values, or of multisets of either. A tensor goes from indices to values,
/// an index map from indices to indices, an index function from indices to
/// multisets of indices, and a reduction from multisets of values to values.
pub trait Function {
    type Domain: Set;
    type Codomain: Set;

    fn apply(&self, x: &<Self::Domain as Set>::Element) -> <Self::Codomain as Set>::Owned;

    fn domain(&self) -> Self::Domain;

    fn codomain(&self) -> Self::Codomain;
}

// A shared function is the same function, so that one can be used in several
// compositions, or have its type erased as an `Rc<dyn Tensor<A>>`.
impl<T: Function + ?Sized> Function for Rc<T> {
    type Domain = T::Domain;
    type Codomain = T::Codomain;

    fn apply(&self, x: &<Self::Domain as Set>::Element) -> <Self::Codomain as Set>::Owned {
        (**self).apply(x)
    }

    fn domain(&self) -> Self::Domain {
        (**self).domain()
    }

    fn codomain(&self) -> Self::Codomain {
        (**self).codomain()
    }
}

/// `f ∘ g`: applies `g`, then `f` to the result.
pub struct Compose<F, G> {
    f: F,
    g: G,
}

/// Composes `f` after `g`. Composing a tensor after an index map re-indexes it,
/// and a function on values after a tensor applies it element-wise. The
/// codomain of `g` must lie in the domain of `f`.
pub fn compose<F, G>(f: F, g: G) -> Compose<F, G>
where
    F: Function<Domain = G::Codomain>,
    G: Function,
{
    let (codomain, domain) = (g.codomain(), f.domain());
    assert!(
        codomain.lies_in(&domain),
        "the inner function lands in {codomain:?}, which does not lie in the domain {domain:?} of the outer one"
    );
    Compose { f, g }
}

impl<F, G> Function for Compose<F, G>
where
    F: Function<Domain = G::Codomain>,
    G: Function,
{
    type Domain = G::Domain;
    type Codomain = F::Codomain;

    fn apply(&self, x: &<Self::Domain as Set>::Element) -> <Self::Codomain as Set>::Owned {
        self.f.apply(self.g.apply(x).borrow())
    }

    fn domain(&self) -> Self::Domain {
        self.g.domain()
    }

    fn codomain(&self) -> Self::Codomain {
        self.f.codomain()
    }
}

/// A function from multisets to multisets: the extension of `f` to multisets,
/// written `f̄` in the paper. It maps a multiset to the multiset of the images
/// of its elements, repeats included, so it goes from the multisets of the
/// domain of `f` to those of its codomain.
pub struct MultiFunction<F> {
    f: F,
}

pub fn multiset_lift<F: Function>(f: F) -> MultiFunction<F> {
    MultiFunction { f }
}

impl<F: Function> Function for MultiFunction<F> {
    type Domain = Multisets<F::Domain>;
    type Codomain = Multisets<F::Codomain>;

    fn apply(&self, x: &<Self::Domain as Set>::Element) -> <Self::Codomain as Set>::Owned {
        x.iter()
            .map(|element| self.f.apply(element.borrow()))
            .collect()
    }

    fn domain(&self) -> Self::Domain {
        Multisets::new(self.f.domain())
    }

    fn codomain(&self) -> Self::Codomain {
        Multisets::new(self.f.codomain())
    }
}

/// A function on values, given as a closure, from `A` to `B`. Composed after a
/// tensor, it applies element-wise.
pub struct ValueFunction<A, B = A> {
    // Held as a closure rather than a plain `fn` pointer so that a function can
    // carry values of its own, as `scale` and `clamp` do.
    f: Rc<dyn Fn(A) -> B>,
}

impl<A, B> ValueFunction<A, B> {
    pub fn new(f: impl Fn(A) -> B + 'static) -> Self {
        Self { f: Rc::new(f) }
    }
}

impl<A: Clone, B> Function for ValueFunction<A, B> {
    type Domain = Values<A>;
    type Codomain = Values<B>;

    fn apply(&self, x: &A) -> B {
        (self.f)(x.clone())
    }

    fn domain(&self) -> Values<A> {
        Values::new()
    }

    fn codomain(&self) -> Values<B> {
        Values::new()
    }
}
