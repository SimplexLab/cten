//! The sets functions go between: indices, values, and multisets of either.

use std::borrow::Borrow;
use std::fmt;
use std::marker::PhantomData;

use crate::domain::{Domain, HRect, JaggedDomain};
use crate::multiset::Multiset;

/// A set that functions go between. A function takes an element of its domain
/// as an `Element`, and returns an element of its codomain as an `Owned`.
pub trait Set: Clone + fmt::Debug {
    type Element: ?Sized;
    type Owned: Borrow<Self::Element>;

    /// Whether every element of `self` lies in `other`. Composing `f` after `g`
    /// needs the codomain of `g` to lie in the domain of `f`.
    fn lies_in(&self, other: &Self) -> bool;
}

impl Set for HRect {
    type Element = [usize];
    type Owned = Vec<usize>;

    fn lies_in(&self, other: &Self) -> bool {
        self.ndim() == other.ndim()
            && self
                .shape()
                .iter()
                .zip(other.shape())
                .all(|(extent, other)| extent <= other)
    }
}

impl Set for JaggedDomain {
    type Element = [usize];
    type Owned = Vec<usize>;

    fn lies_in(&self, other: &Self) -> bool {
        self.parts().len() <= other.parts().len()
            && self
                .parts()
                .iter()
                .zip(other.parts())
                .all(|(part, other)| part.lies_in(other))
    }
}

/// All the values of type `A`, such as ℝ for `f32`. This is the domain of a
/// function on values, and the codomain of a tensor.
pub struct Values<A>(PhantomData<fn() -> A>);

impl<A> Values<A> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

// Written out rather than derived, which would require the same of `A`.
impl<A> Default for Values<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> Clone for Values<A> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<A> fmt::Debug for Values<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Values<{}>", std::any::type_name::<A>())
    }
}

impl<A> Set for Values<A> {
    type Element = A;
    type Owned = A;

    // The type says it all: the compiler has already checked it.
    fn lies_in(&self, _other: &Self) -> bool {
        true
    }
}

/// The multisets of elements of `S`, written `M(S)` in the paper.
#[derive(Clone, Debug)]
pub struct Multisets<S> {
    set: S,
}

impl<S> Multisets<S> {
    pub fn new(set: S) -> Self {
        Self { set }
    }

    /// The set the elements of the multisets are drawn from.
    pub fn set(&self) -> &S {
        &self.set
    }
}

impl<S: Set> Set for Multisets<S> {
    type Element = Multiset<S::Owned>;
    type Owned = Multiset<S::Owned>;

    fn lies_in(&self, other: &Self) -> bool {
        self.set.lies_in(&other.set)
    }
}
