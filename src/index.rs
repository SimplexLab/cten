use std::collections::HashSet;
use std::collections::hash_set;
use std::hash::{Hash, Hasher};

pub trait Index: Clone + Eq + Hash {}

/// Index types constructible from a coordinate vector. Only these can be
/// printed, since formatting walks a shape and rebuilds indices from it.
pub trait FromCoords: Index {
    fn from_coords(coords: &[usize]) -> Self;
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Multintdex {
    value: Vec<usize>,
}

impl Multintdex {
    pub fn new(value: Vec<usize>) -> Self {
        Self { value }
    }

    pub fn ndim(&self) -> usize {
        self.value.len()
    }

    pub fn at(&self, i: usize) -> usize {
        self.value[i]
    }

    pub fn coords(&self) -> &[usize] {
        &self.value
    }
}

impl Index for Multintdex {}

impl FromCoords for Multintdex {
    fn from_coords(coords: &[usize]) -> Self {
        Self::new(coords.to_vec())
    }
}

impl From<Vec<usize>> for Multintdex {
    fn from(value: Vec<usize>) -> Self {
        Self::new(value)
    }
}

impl From<&[usize]> for Multintdex {
    fn from(value: &[usize]) -> Self {
        Self::new(value.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Multintdex {
    fn from(value: [usize; N]) -> Self {
        Self::new(value.to_vec())
    }
}

/// A `Multintdex` of one coordinate. Widen with the `From` conversion below.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Intdex {
    value: usize,
}

impl Intdex {
    pub fn new(value: usize) -> Self {
        Self { value }
    }

    pub fn get(self) -> usize {
        self.value
    }

    pub fn ndim(self) -> usize {
        1
    }

    pub fn at(self, i: usize) -> usize {
        assert!(i == 0, "index {i} out of range for Intdex");
        self.value
    }
}

impl Index for Intdex {}

impl From<usize> for Intdex {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl From<Intdex> for Multintdex {
    fn from(value: Intdex) -> Self {
        Multintdex::new(vec![value.get()])
    }
}

#[derive(Clone, Debug, Default)]
pub struct Powerset<T: Index> {
    values: HashSet<T>,
}

impl<T: Index> Powerset<T> {
    pub fn new() -> Self {
        Self {
            values: HashSet::new(),
        }
    }

    pub fn contains(&self, value: &T) -> bool {
        self.values.contains(value)
    }

    pub fn size(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn insert(&mut self, value: T) {
        self.values.insert(value);
    }

    pub fn iter(&self) -> hash_set::Iter<'_, T> {
        self.values.iter()
    }
}

impl<T: Index> FromIterator<T> for Powerset<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }
}

impl<T: Index> IntoIterator for Powerset<T> {
    type Item = T;
    type IntoIter = hash_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<'a, T: Index> IntoIterator for &'a Powerset<T> {
    type Item = &'a T;
    type IntoIter = hash_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<T: Index> PartialEq for Powerset<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<T: Index> Eq for Powerset<T> {}

impl<T: Index> Hash for Powerset<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut seed: u64 = 0;
        for elem in &self.values {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            elem.hash(&mut hasher);
            seed ^= hasher.finish();
        }
        state.write_u64(seed);
    }
}

impl<T: Index> Index for Powerset<T> {}
