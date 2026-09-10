use std::marker::PhantomData;

use crate::index::{Index, Intdex, Multintdex, Powerset};

pub trait IndexMapping<V: Index, U: Index> {
    fn map(&self, index: V) -> U;
}

pub trait PowersetMapping<V: Index, U: Index>: IndexMapping<V, Powerset<U>> {}

impl<M, V, U> PowersetMapping<V, U> for M
where
    V: Index,
    U: Index,
    M: IndexMapping<V, Powerset<U>> + ?Sized,
{
}

pub struct Stride {
    strides: Vec<usize>,
}

impl Stride {
    pub fn new(strides: Vec<usize>) -> Self {
        Self { strides }
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

impl IndexMapping<Multintdex, Intdex> for Stride {
    fn map(&self, index: Multintdex) -> Intdex {
        let mut result = 0;
        for (i, stride) in self.strides.iter().enumerate() {
            result += index.at(i) * stride;
        }
        Intdex::new(result)
    }
}

pub struct EmptyMapping<V: Index, U: Index> {
    _marker: PhantomData<fn(V) -> U>,
}

impl<V: Index, U: Index> EmptyMapping<V, U> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<V: Index, U: Index> Default for EmptyMapping<V, U> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Index, U: Index> IndexMapping<V, Powerset<U>> for EmptyMapping<V, U> {
    fn map(&self, _index: V) -> Powerset<U> {
        Powerset::new()
    }
}

pub struct COO {
    inverse_coords: Vec<Vec<usize>>,
}

impl COO {
    pub fn new(inverse_coords: Vec<Vec<usize>>) -> Self {
        Self { inverse_coords }
    }

    pub fn inverse_coords(&self) -> &[Vec<usize>] {
        &self.inverse_coords
    }
}

impl IndexMapping<Multintdex, Powerset<Intdex>> for COO {
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
}

pub struct Diagonal;

impl Diagonal {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Diagonal {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexMapping<Multintdex, Powerset<Intdex>> for Diagonal {
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
}
