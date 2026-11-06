//! The sets functions go between: sets of indices, sets of values, and sets of
//! multisets of either.

use std::borrow::Borrow;
use std::fmt;
use std::marker::PhantomData;

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

/// A set of indices, an index being a vector of coordinates. Tensors go from
/// one, and index functions from one to another or to its multisets.
pub trait IndexSet {
    /// Number of coordinates the indices of this set have.
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

impl IndexSet for HRect {
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

/// The disjoint union of several hrects: `[t, rest..]` belongs to it when `rest`
/// is in hrect `t`. This is the domain of a jagged tensor, whose parts may have
/// different shapes.
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

impl IndexSet for JaggedDomain {
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
