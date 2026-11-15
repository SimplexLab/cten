//! The building bricks: functions, their composition, and their extension to
//! multisets. Tensors, index maps, index functions and reductions are all
//! functions, and every operation on them is built from these.

use std::any::Any;
use std::borrow::Borrow;
use std::rc::Rc;

use crate::sets::{Multisets, Set, Values};

/// A function from its domain to its codomain, which may be sets of indices,
/// of values, or of multisets of either. A tensor goes from indices to values,
/// an index map from indices to indices, an index function from indices to
/// multisets of indices, and a reduction from multisets of values to values.
///
/// Besides `apply`, a function can describe itself so that `rewrite` can find
/// the identities of the paper that apply to it. Every such method has a
/// default that finds nothing, which is always correct.
pub trait Function {
    type Domain: Set;
    type Codomain: Set;

    fn apply(&self, x: &<Self::Domain as Set>::Element) -> <Self::Codomain as Set>::Owned;

    fn domain(&self) -> Self::Domain;

    fn codomain(&self) -> Self::Codomain;

    /// An equivalent function that is cheaper to evaluate, or `None` when no
    /// identity applies anywhere inside this one.
    fn rewritten(
        &self,
    ) -> Option<Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Codomain>>> {
        None
    }

    /// When this is a reduction `Ω` after a function returning multisets of at
    /// most one element, and `f` keeps the neutral element of `Ω`: `f` after
    /// this, rewritten as `Ω ∘ f̄` after that function. `f` is given as an
    /// `Rc<dyn Function>` from the codomain of this to itself, behind `Any`.
    fn absorbed(
        &self,
        _f: &dyn Any,
    ) -> Option<Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Codomain>>> {
        None
    }

    /// When this is a reduction and `f` keeps its neutral element: `f̄`, the
    /// extension of `f` to the multisets this reduces. `f` is given as for
    /// `absorbed`.
    fn lifted_keeping_neutral(
        &self,
        _f: &dyn Any,
    ) -> Option<Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Domain>>> {
        None
    }

    /// Whether every multiset this returns holds at most one element.
    fn returns_at_most_one(&self) -> bool {
        false
    }

    /// Whether this maps each multiset to one of the same size, as the
    /// extension of a function to multisets does.
    fn keeps_sizes(&self) -> bool {
        false
    }
}

// A shared function is the same function, so that one can be used in several
// compositions, or have its type erased as an `Rc<dyn Tensor<A>>`. Rewriting
// sees through it.
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

    fn rewritten(
        &self,
    ) -> Option<Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Codomain>>> {
        (**self).rewritten()
    }

    fn absorbed(
        &self,
        f: &dyn Any,
    ) -> Option<Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Codomain>>> {
        (**self).absorbed(f)
    }

    fn lifted_keeping_neutral(
        &self,
        f: &dyn Any,
    ) -> Option<Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Domain>>> {
        (**self).lifted_keeping_neutral(f)
    }

    fn returns_at_most_one(&self) -> bool {
        (**self).returns_at_most_one()
    }

    fn keeps_sizes(&self) -> bool {
        (**self).keeps_sizes()
    }
}

/// `f ∘ g`: applies `g`, then `f` to the result.
pub struct Composition<F, G> {
    // Shared, so that rewriting can reuse them without taking this apart.
    pub(crate) f: Rc<F>,
    pub(crate) g: Rc<G>,
}

/// Composes `f` after `g`. Composing a tensor after an index map re-indexes it,
/// and a function on values after a tensor applies it element-wise. The
/// codomain of `g` must lie in the domain of `f`.
pub fn compose<F, G>(f: F, g: G) -> Composition<F, G>
where
    F: Function<Domain = G::Codomain>,
    G: Function,
{
    let (codomain, domain) = (g.codomain(), f.domain());
    assert!(
        codomain.lies_in(&domain),
        "the inner function lands in {codomain:?}, which does not lie in the domain {domain:?} of the outer one"
    );
    Composition {
        f: Rc::new(f),
        g: Rc::new(g),
    }
}

impl<F, G> Function for Composition<F, G>
where
    F: Function<Domain = G::Codomain> + 'static,
    G: Function + 'static,
    F::Codomain: 'static,
    G::Domain: 'static,
    G::Codomain: 'static,
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

    // Rewrites both sides, then `f` after the rewritten `g` with Theorem 1, which
    // `absorbed` checks the conditions of.
    fn rewritten(
        &self,
    ) -> Option<Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Codomain>>> {
        let new_f = self.f.rewritten();
        let new_g = self.g.rewritten();
        let changed = new_f.is_some() || new_g.is_some();
        let f: Rc<dyn Function<Domain = F::Domain, Codomain = F::Codomain>> =
            new_f.unwrap_or_else(|| self.f.clone());
        let g: Rc<dyn Function<Domain = G::Domain, Codomain = G::Codomain>> =
            new_g.unwrap_or_else(|| self.g.clone());
        if let Some(absorbed) = g.absorbed(&f) {
            // `f` was taken for a function from the codomain of `g` to itself, so
            // the codomain of `g` is that of `f`, as the downcast confirms.
            let absorbed: Box<dyn Any> = Box::new(absorbed);
            if let Ok(absorbed) = absorbed
                .downcast::<Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Codomain>>>()
            {
                return Some(*absorbed);
            }
        }
        changed.then(|| {
            Rc::new(compose(f, g))
                as Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Codomain>>
        })
    }

    // Here this is `Ω ∘ H`, with `Ω` in `f` and `H` in `g`.
    fn absorbed(
        &self,
        f: &dyn Any,
    ) -> Option<Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Codomain>>> {
        if !self.g.returns_at_most_one() {
            return None;
        }
        let lifted = self.f.lifted_keeping_neutral(f)?;
        // Erased, so that the result has the same type whatever this one is, and
        // rewriting it again does not build ever larger types.
        let omega: Rc<dyn Function<Domain = F::Domain, Codomain = F::Codomain>> = self.f.clone();
        let inner: Rc<dyn Function<Domain = G::Domain, Codomain = G::Codomain>> = self.g.clone();
        Some(Rc::new(compose(omega, compose(lifted, inner))))
    }

    fn returns_at_most_one(&self) -> bool {
        self.g.returns_at_most_one() && self.f.keeps_sizes()
    }
}

/// A function from multisets to multisets: the extension of `f` to multisets,
/// written `f̄` in the paper. It maps a multiset to the multiset of the images
/// of its elements, repeats included, so it goes from the multisets of the
/// domain of `f` to those of its codomain.
pub struct MultiFunction<F> {
    pub(crate) f: F,
}

pub fn multiset_lift<F: Function>(f: F) -> MultiFunction<F> {
    MultiFunction { f }
}

impl<F: Function> Function for MultiFunction<F>
where
    F::Domain: 'static,
    F::Codomain: 'static,
{
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

    fn rewritten(
        &self,
    ) -> Option<Rc<dyn Function<Domain = Self::Domain, Codomain = Self::Codomain>>> {
        let rewritten = self.f.rewritten()?;
        Some(Rc::new(multiset_lift(rewritten)))
    }

    fn keeps_sizes(&self) -> bool {
        true
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
