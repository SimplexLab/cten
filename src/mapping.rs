use crate::domain::{BoxDomain, Domain};
use crate::index::{Coords, Intdex, Multintdex, Powerset};

/// A mapping between two index sets, from its domain to its codomain.
pub trait IndexMapping<V: Coords, U: Coords> {
    fn map(&self, index: V) -> U;

    fn domain(&self) -> &dyn Domain<V>;

    fn codomain(&self) -> &dyn Domain<U>;
}

/// A mapping whose domain is a box. Composing a tensor over one of these gives
/// a boxed tensor, hence a printable one.
pub trait BoxedMapping<V: Coords, U: Coords>: IndexMapping<V, U> {
    fn box_domain(&self) -> &BoxDomain<V>;
}

/// A mapping sending each index of its domain to a set of indices of its
/// codomain. Note that the codomain is the set the returned indices are drawn
/// from, not its powerset.
pub trait PowersetMapping<V: Coords, U: Coords> {
    fn map(&self, index: V) -> Powerset<U>;

    fn domain(&self) -> &dyn Domain<V>;

    fn codomain(&self) -> &dyn Domain<U>;
}

/// A powerset mapping whose domain is a box. Reducing over one of these gives a
/// boxed tensor, hence a printable one.
pub trait BoxedPowersetMapping<V: Coords, U: Coords>: PowersetMapping<V, U> {
    fn box_domain(&self) -> &BoxDomain<V>;
}

pub struct Stride {
    domain: BoxDomain<Multintdex>,
    codomain: BoxDomain<Intdex>,
    strides: Vec<usize>,
}

impl Stride {
    pub fn new(domain: BoxDomain<Multintdex>, strides: Vec<usize>) -> Self {
        assert_eq!(
            domain.ndim(),
            strides.len(),
            "a stride per dimension of the domain is required"
        );
        let codomain = BoxDomain::new(vec![flat_extent(domain.shape(), &strides)]);
        Self {
            domain,
            codomain,
            strides,
        }
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

/// Number of flat positions a strided box reaches: one past its largest index.
fn flat_extent(shape: &[usize], strides: &[usize]) -> usize {
    if shape.contains(&0) {
        return 0;
    }
    1 + shape
        .iter()
        .zip(strides)
        .map(|(extent, stride)| (extent - 1) * stride)
        .sum::<usize>()
}

impl IndexMapping<Multintdex, Intdex> for Stride {
    fn map(&self, index: Multintdex) -> Intdex {
        let mut result = 0;
        for (i, stride) in self.strides.iter().enumerate() {
            result += index.at(i) * stride;
        }
        Intdex::new(result)
    }

    fn domain(&self) -> &dyn Domain<Multintdex> {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain<Intdex> {
        &self.codomain
    }
}

impl BoxedMapping<Multintdex, Intdex> for Stride {
    fn box_domain(&self) -> &BoxDomain<Multintdex> {
        &self.domain
    }
}

pub struct EmptyMapping<V: Coords, U: Coords> {
    domain: BoxDomain<V>,
    codomain: BoxDomain<U>,
}

impl<V: Coords, U: Coords> EmptyMapping<V, U> {
    pub fn new(domain: BoxDomain<V>, codomain: BoxDomain<U>) -> Self {
        Self { domain, codomain }
    }
}

impl<V: Coords, U: Coords> PowersetMapping<V, U> for EmptyMapping<V, U> {
    fn map(&self, _index: V) -> Powerset<U> {
        Powerset::new()
    }

    fn domain(&self) -> &dyn Domain<V> {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain<U> {
        &self.codomain
    }
}

impl<V: Coords, U: Coords> BoxedPowersetMapping<V, U> for EmptyMapping<V, U> {
    fn box_domain(&self) -> &BoxDomain<V> {
        &self.domain
    }
}

pub struct COO {
    domain: BoxDomain<Multintdex>,
    codomain: BoxDomain<Intdex>,
    inverse_coords: Vec<Vec<usize>>,
}

impl COO {
    pub fn new(domain: BoxDomain<Multintdex>, inverse_coords: Vec<Vec<usize>>) -> Self {
        let codomain = BoxDomain::new(vec![inverse_coords.len()]);
        Self {
            domain,
            codomain,
            inverse_coords,
        }
    }

    pub fn inverse_coords(&self) -> &[Vec<usize>] {
        &self.inverse_coords
    }
}

impl PowersetMapping<Multintdex, Intdex> for COO {
    fn map(&self, index: Multintdex) -> Powerset<Intdex> {
        let mut result = Powerset::new();
        for (j, coords) in self.inverse_coords.iter().enumerate() {
            let matches = coords
                .iter()
                .enumerate()
                .all(|(i, coord)| *coord == index.at(i));
            if matches {
                result.insert(Intdex::new(j));
            }
        }
        result
    }

    fn domain(&self) -> &dyn Domain<Multintdex> {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain<Intdex> {
        &self.codomain
    }
}

impl BoxedPowersetMapping<Multintdex, Intdex> for COO {
    fn box_domain(&self) -> &BoxDomain<Multintdex> {
        &self.domain
    }
}

pub struct Diagonal {
    domain: BoxDomain<Multintdex>,
    codomain: BoxDomain<Intdex>,
}

impl Diagonal {
    pub fn new(domain: BoxDomain<Multintdex>) -> Self {
        // Only indices repeated along every dimension are hit, so the diagonal
        // is as long as the shortest dimension.
        let extent = domain.shape().iter().copied().min().unwrap_or(0);
        Self {
            domain,
            codomain: BoxDomain::new(vec![extent]),
        }
    }
}

impl PowersetMapping<Multintdex, Intdex> for Diagonal {
    fn map(&self, index: Multintdex) -> Powerset<Intdex> {
        let mut result = Powerset::new();
        let first = index.at(0);
        for i in 0..index.ndim() {
            if index.at(i) != first {
                return result;
            }
        }
        result.insert(Intdex::new(first));
        result
    }

    fn domain(&self) -> &dyn Domain<Multintdex> {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain<Intdex> {
        &self.codomain
    }
}

impl BoxedPowersetMapping<Multintdex, Intdex> for Diagonal {
    fn box_domain(&self) -> &BoxDomain<Multintdex> {
        &self.domain
    }
}
