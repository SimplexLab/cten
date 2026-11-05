//! Multisets: collections in which an element may appear more than once, and
//! in which the multiplicity of each element counts but the order does not.

use std::fmt;
use std::fmt::Display;

/// An unordered collection of elements, repeats included. It is stored as a
/// vector, whose order carries no meaning.
#[derive(Clone, Debug)]
pub struct Multiset<A> {
    elements: Vec<A>,
}

impl<A> Multiset<A> {
    pub fn new(elements: Vec<A>) -> Self {
        Self { elements }
    }

    pub fn push(&mut self, element: A) {
        self.elements.push(element);
    }

    pub fn iter(&self) -> std::slice::Iter<'_, A> {
        self.elements.iter()
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

// Written out rather than derived, which would require `A: Default`.
impl<A> Default for Multiset<A> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl<A> FromIterator<A> for Multiset<A> {
    fn from_iter<I: IntoIterator<Item = A>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl<A> IntoIterator for Multiset<A> {
    type Item = A;
    type IntoIter = std::vec::IntoIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<'a, A> IntoIterator for &'a Multiset<A> {
    type Item = &'a A;
    type IntoIter = std::slice::Iter<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter()
    }
}

/// Written as in the paper, with double braces: `{{1, 3, 3}}`. Padded as a whole,
/// so that multisets line up in a printed tensor as numbers do.
impl<A: Display> Display for Multiset<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elements: Vec<String> = self.elements.iter().map(A::to_string).collect();
        f.pad(&format!("{{{{{}}}}}", elements.join(", ")))
    }
}
