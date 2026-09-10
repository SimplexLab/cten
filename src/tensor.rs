use std::fmt;
use std::fmt::Write as _;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::index::{FromCoords, Index, Intdex, Multintdex};
use crate::mapping::{IndexMapping, PowersetMapping};

/// A mapping from an index set to a value set. The value set is fixed to the
/// float numbers for now, for simplicity.
pub trait Tensor<U: Index> {
    fn get(&self, index: U) -> f32;
}

pub type Vector = dyn Tensor<Intdex>;

pub struct Physical {
    values: Vec<f32>,
}

impl Physical {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }
}

impl Tensor<Intdex> for Physical {
    fn get(&self, index: Intdex) -> f32 {
        self.values[index.get()]
    }
}

impl fmt::Display for Physical {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = [self.values.len()];
        f.write_str(&format_tensor(&shape, &|index| self.values[index[0]]))
    }
}

// For now, the virtual index type is forced to be Multintdex.
// TODO: relax that (different printing, no concept of shape, etc.)
pub struct RightComposition<U: Index> {
    u: Rc<dyn Tensor<U>>,
    mapping: Rc<dyn IndexMapping<Multintdex, U>>,
    shape: Vec<usize>,
}

impl<U: Index> RightComposition<U> {
    pub fn new(
        u: Rc<dyn Tensor<U>>,
        mapping: Rc<dyn IndexMapping<Multintdex, U>>,
        shape: Vec<usize>,
    ) -> Self {
        Self { u, mapping, shape }
    }
}

impl<U: Index> Tensor<Multintdex> for RightComposition<U> {
    fn get(&self, virtual_index: Multintdex) -> f32 {
        let underlying_index = self.mapping.map(virtual_index);
        self.u.get(underlying_index)
    }
}

impl<U: Index> fmt::Display for RightComposition<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_tensor(&self.shape, &|index| {
            self.get(Multintdex::from_coords(index))
        }))
    }
}

pub struct LeftComposition<U: Index> {
    u: Rc<dyn Tensor<U>>,
    f: fn(f32) -> f32,
    shape: Vec<usize>,
}

impl<U: Index> LeftComposition<U> {
    pub fn new(u: Rc<dyn Tensor<U>>, f: fn(f32) -> f32, shape: Vec<usize>) -> Self {
        Self { u, f, shape }
    }
}

impl<U: Index> Tensor<U> for LeftComposition<U> {
    fn get(&self, index: U) -> f32 {
        (self.f)(self.u.get(index))
    }
}

impl<U: FromCoords> fmt::Display for LeftComposition<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_tensor(&self.shape, &|index| {
            self.get(U::from_coords(index))
        }))
    }
}

/// Wrapper around a 1D Multintdex tensor making it an Intdex Tensor.
pub struct Intdexed {
    u: Rc<dyn Tensor<Multintdex>>,
}

impl Intdexed {
    pub fn new(u: Rc<dyn Tensor<Multintdex>>) -> Self {
        Self { u }
    }
}

impl Tensor<Intdex> for Intdexed {
    fn get(&self, index: Intdex) -> f32 {
        self.u.get(Multintdex::new(vec![index.get()]))
    }
}

pub trait Reduction {
    const NEUTRAL: f32;
    fn combine(a: f32, b: f32) -> f32;
}

pub struct SumReduction;

impl Reduction for SumReduction {
    const NEUTRAL: f32 = 0.0;
    fn combine(a: f32, b: f32) -> f32 {
        a + b
    }
}

pub struct ProductReduction;

impl Reduction for ProductReduction {
    const NEUTRAL: f32 = 1.0;
    fn combine(a: f32, b: f32) -> f32 {
        a * b
    }
}

pub struct SupReduction;

impl Reduction for SupReduction {
    const NEUTRAL: f32 = f32::NEG_INFINITY;
    fn combine(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}

pub struct InfReduction;

impl Reduction for InfReduction {
    const NEUTRAL: f32 = f32::INFINITY;
    fn combine(a: f32, b: f32) -> f32 {
        a.min(b)
    }
}

pub struct ReductionTensor<V: Index, U: Index, R: Reduction> {
    u: Rc<dyn Tensor<U>>,
    mapping: Rc<dyn PowersetMapping<V, U>>,
    shape: Vec<usize>,
    _marker: PhantomData<fn(V) -> R>,
}

impl<V: Index, U: Index, R: Reduction> ReductionTensor<V, U, R> {
    pub fn new(
        u: Rc<dyn Tensor<U>>,
        mapping: Rc<dyn PowersetMapping<V, U>>,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            u,
            mapping,
            shape,
            _marker: PhantomData,
        }
    }
}

impl<V: Index, U: Index, R: Reduction> Tensor<V> for ReductionTensor<V, U, R> {
    fn get(&self, index: V) -> f32 {
        let indices = self.mapping.map(index);
        let mut result = R::NEUTRAL;
        for i in &indices {
            result = R::combine(result, self.u.get(i.clone()));
        }
        result
    }
}

impl<V: FromCoords, U: Index, R: Reduction> fmt::Display for ReductionTensor<V, U, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_tensor(&self.shape, &|index| {
            self.get(V::from_coords(index))
        }))
    }
}

pub type Sum<V, U> = ReductionTensor<V, U, SumReduction>;
pub type Product<V, U> = ReductionTensor<V, U, ProductReduction>;
pub type Sup<V, U> = ReductionTensor<V, U, SupReduction>;
pub type Inf<V, U> = ReductionTensor<V, U, InfReduction>;

fn format_tensor(shape: &[usize], accessor: &dyn Fn(&[usize]) -> f32) -> String {
    if shape.is_empty() {
        return format!("tensor({})", accessor(&[]));
    }

    let mut index = vec![0; shape.len()];
    let width = max_width(shape, &mut index, 0, accessor);
    let mut out = "tensor(".to_string();
    // "tensor(" is 8 characters wide counting the opening bracket, so rows of
    // the outermost dimension line up under it.
    format_nested(
        &mut out,
        shape,
        &mut index,
        0,
        &" ".repeat(8),
        width,
        accessor,
    );
    out.push(')');
    out
}

fn max_width(
    shape: &[usize],
    index: &mut [usize],
    dim: usize,
    accessor: &dyn Fn(&[usize]) -> f32,
) -> usize {
    let mut width = 0;
    for i in 0..shape[dim] {
        index[dim] = i;
        width = width.max(if dim == shape.len() - 1 {
            accessor(index).to_string().len()
        } else {
            max_width(shape, index, dim + 1, accessor)
        });
    }
    width
}

fn format_nested(
    out: &mut String,
    shape: &[usize],
    index: &mut [usize],
    dim: usize,
    indent: &str,
    width: usize,
    accessor: &dyn Fn(&[usize]) -> f32,
) {
    let innermost = dim == shape.len() - 1;
    out.push('[');
    for i in 0..shape[dim] {
        index[dim] = i;
        if innermost {
            let _ = write!(out, "{:>width$}", accessor(index));
        } else {
            format_nested(
                out,
                shape,
                index,
                dim + 1,
                &format!("{indent} "),
                width,
                accessor,
            );
        }
        if i + 1 < shape[dim] {
            if innermost {
                out.push_str(", ");
            } else {
                out.push_str(",\n");
                // Separate blocks of rank 3 and above with a blank line.
                if dim + 2 < shape.len() {
                    out.push('\n');
                }
                out.push_str(indent);
            }
        }
    }
    out.push(']');
}
