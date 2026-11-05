//! The building bricks: functions, their composition, and their extension to
//! multisets. Tensors, index maps, index functions and reductions are all
//! functions, and every operation on them is built from these.

use std::borrow::Borrow;
use std::rc::Rc;

use crate::domain::{Domain, HRect};
use crate::multiset::Multiset;

/// A function from `Input` to `Output`. A tensor is a function from indices to
/// values, an index map one from indices to indices, an index function one from
/// indices to multisets of indices, and a reduction one from multisets of values
/// to values.
pub trait Function {
    type Input: ?Sized;
    type Output;

    fn apply(&self, x: &Self::Input) -> Self::Output;

    /// For a function on indices, or on multisets of indices, the set those
    /// indices are drawn from. A function on values has none.
    fn domain(&self) -> Option<&dyn Domain> {
        None
    }

    /// For a function to indices, or to multisets of indices, the set those
    /// indices are drawn from. A function to values has none.
    fn codomain(&self) -> Option<&dyn Domain> {
        None
    }
}

/// A function whose domain is a hrect. Only these can be printed, since
/// formatting walks a shape and visits every index of it.
pub trait Boxed: Function {
    fn hrect(&self) -> &HRect;
}

// A shared function is the same function, so that one can be used in several
// compositions, or have its type erased as an `Rc<dyn Tensor<A>>`.
impl<T: Function + ?Sized> Function for Rc<T> {
    type Input = T::Input;
    type Output = T::Output;

    fn apply(&self, x: &Self::Input) -> Self::Output {
        (**self).apply(x)
    }

    fn domain(&self) -> Option<&dyn Domain> {
        (**self).domain()
    }

    fn codomain(&self) -> Option<&dyn Domain> {
        (**self).codomain()
    }
}

impl<T: Boxed + ?Sized> Boxed for Rc<T> {
    fn hrect(&self) -> &HRect {
        (**self).hrect()
    }
}

/// `f ∘ g`: applies `g`, then `f` to the result.
pub struct Compose<F, G> {
    f: F,
    g: G,
}

/// Composes `f` after `g`. Composing a tensor after an index map re-indexes it,
/// and a function on values after a tensor applies it element-wise. Where both
/// sides are on indices, the indices `g` lands on must be those `f` takes, which
/// is checked here as far as it can be without enumerating a domain.
pub fn compose<F: Function, G: Function>(f: F, g: G) -> Compose<F, G>
where
    G::Output: Borrow<F::Input>,
{
    if let (Some(codomain), Some(domain)) = (g.codomain(), f.domain()) {
        assert_eq!(
            codomain.ndim(),
            domain.ndim(),
            "the inner function lands on {}-dimensional indices, but the outer one is indexed by {} coordinates",
            codomain.ndim(),
            domain.ndim()
        );
    }
    Compose { f, g }
}

impl<F: Function, G: Function> Function for Compose<F, G>
where
    G::Output: Borrow<F::Input>,
{
    type Input = G::Input;
    type Output = F::Output;

    fn apply(&self, x: &Self::Input) -> Self::Output {
        self.f.apply(self.g.apply(x).borrow())
    }

    fn domain(&self) -> Option<&dyn Domain> {
        self.g.domain()
    }

    fn codomain(&self) -> Option<&dyn Domain> {
        self.f.codomain()
    }
}

impl<F: Function, G: Boxed> Boxed for Compose<F, G>
where
    G::Output: Borrow<F::Input>,
{
    fn hrect(&self) -> &HRect {
        self.g.hrect()
    }
}

/// The extension of `f` to multisets, written `f̄` in the paper: it maps a
/// multiset to the multiset of the images of its elements, repeats included.
pub struct MultisetLift<F> {
    f: F,
}

pub fn multiset_lift<F: Function>(f: F) -> MultisetLift<F>
where
    F::Input: ToOwned,
{
    MultisetLift { f }
}

// The elements are taken owned, so that a tensor, which takes `[usize]`, lifts to
// take the `Multiset<Vec<usize>>` an index function returns.
impl<F: Function> Function for MultisetLift<F>
where
    F::Input: ToOwned,
{
    type Input = Multiset<<F::Input as ToOwned>::Owned>;
    type Output = Multiset<F::Output>;

    fn apply(&self, x: &Self::Input) -> Self::Output {
        x.iter()
            .map(|element| self.f.apply(element.borrow()))
            .collect()
    }

    fn domain(&self) -> Option<&dyn Domain> {
        self.f.domain()
    }

    fn codomain(&self) -> Option<&dyn Domain> {
        self.f.codomain()
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
    type Input = A;
    type Output = B;

    fn apply(&self, x: &A) -> B {
        (self.f)(x.clone())
    }
}
