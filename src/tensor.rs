use std::fmt;
use std::fmt::Write as _;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::domain::{BoxDomain, Domain};
use crate::index::{Coords, Intdex, Multintdex};
use crate::mapping::{BoxedMapping, BoxedPowersetMapping, IndexMapping, PowersetMapping};

/// A mapping from an index set to a value set. The index set is the tensor's
/// domain; the value set is fixed to the float numbers for now, for simplicity.
pub trait Tensor<U: Coords> {
    fn get(&self, index: U) -> f32;

    fn domain(&self) -> &dyn Domain<U>;
}

/// A tensor whose domain is a box. Only these can be printed, since formatting
/// walks a shape and rebuilds indices from it.
pub trait BoxedTensor<U: Coords>: Tensor<U> {
    fn box_domain(&self) -> &BoxDomain<U>;
}

/// Renders a boxed tensor. Every `Display` implementation below defers to this,
/// and each is written for the boxed instantiation of its tensor only.
pub fn format_boxed<U: Coords, T: BoxedTensor<U> + ?Sized>(tensor: &T) -> String {
    format_tensor(tensor.box_domain().shape(), &|coords| {
        tensor.get(U::from_coords(coords))
    })
}

pub type Vector = dyn Tensor<Intdex>;

pub struct Physical {
    values: Vec<f32>,
    domain: BoxDomain<Intdex>,
}

impl Physical {
    pub fn new(values: Vec<f32>) -> Self {
        let domain = BoxDomain::new(vec![values.len()]);
        Self { values, domain }
    }
}

impl Tensor<Intdex> for Physical {
    fn get(&self, index: Intdex) -> f32 {
        self.values[index.get()]
    }

    fn domain(&self) -> &dyn Domain<Intdex> {
        &self.domain
    }
}

impl BoxedTensor<Intdex> for Physical {
    fn box_domain(&self) -> &BoxDomain<Intdex> {
        &self.domain
    }
}

impl fmt::Display for Physical {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

// For now, the virtual index type is forced to be Multintdex.
// TODO: relax that (different printing, no concept of shape, etc.)
//
// The domain is the mapping's own domain, so it is never passed in. The mapping
// type is a parameter rather than always `dyn IndexMapping` so that composing
// over a boxed mapping is known to give a boxed, printable tensor. Pass
// `dyn BoxedMapping<Multintdex, U>` or `dyn IndexMapping<Multintdex, U>` for it
// to erase the mapping.
pub struct RightComposition<U: Coords, M: ?Sized> {
    u: Rc<dyn Tensor<U>>,
    mapping: Rc<M>,
}

impl<U: Coords, M: IndexMapping<Multintdex, U> + ?Sized> RightComposition<U, M> {
    pub fn new(u: Rc<dyn Tensor<U>>, mapping: Rc<M>) -> Self {
        Self { u, mapping }
    }
}

impl<U: Coords, M: IndexMapping<Multintdex, U> + ?Sized> Tensor<Multintdex>
    for RightComposition<U, M>
{
    fn get(&self, virtual_index: Multintdex) -> f32 {
        let underlying_index = self.mapping.map(virtual_index);
        self.u.get(underlying_index)
    }

    fn domain(&self) -> &dyn Domain<Multintdex> {
        self.mapping.domain()
    }
}

impl<U: Coords, M: BoxedMapping<Multintdex, U> + ?Sized> BoxedTensor<Multintdex>
    for RightComposition<U, M>
{
    fn box_domain(&self) -> &BoxDomain<Multintdex> {
        self.mapping.box_domain()
    }
}

impl<U: Coords, M: BoxedMapping<Multintdex, U> + ?Sized> fmt::Display for RightComposition<U, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

// Transforming the values leaves the index set alone, so the domain is the
// underlying tensor's. Its type is a parameter so that a transform of a boxed
// tensor is known to be boxed, and printable, in turn.
pub struct LeftComposition<U: Coords, T: ?Sized> {
    u: Rc<T>,
    f: fn(f32) -> f32,
    _marker: PhantomData<fn(U)>,
}

impl<U: Coords, T: Tensor<U> + ?Sized> LeftComposition<U, T> {
    pub fn new(u: Rc<T>, f: fn(f32) -> f32) -> Self {
        Self {
            u,
            f,
            _marker: PhantomData,
        }
    }
}

impl<U: Coords, T: Tensor<U> + ?Sized> Tensor<U> for LeftComposition<U, T> {
    fn get(&self, index: U) -> f32 {
        (self.f)(self.u.get(index))
    }

    fn domain(&self) -> &dyn Domain<U> {
        self.u.domain()
    }
}

impl<U: Coords, T: BoxedTensor<U> + ?Sized> BoxedTensor<U> for LeftComposition<U, T> {
    fn box_domain(&self) -> &BoxDomain<U> {
        self.u.box_domain()
    }
}

impl<U: Coords, T: BoxedTensor<U> + ?Sized> fmt::Display for LeftComposition<U, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

/// Wrapper around a 1D Multintdex tensor making it an Intdex Tensor. Its domain
/// has to be given explicitly: the wrapped tensor only exposes a domain over
/// Multintdex, which is not a domain over Intdex.
pub struct Intdexed<D> {
    u: Rc<dyn Tensor<Multintdex>>,
    domain: D,
}

impl<D: Domain<Intdex>> Intdexed<D> {
    pub fn new(u: Rc<dyn Tensor<Multintdex>>, domain: D) -> Self {
        Self { u, domain }
    }
}

impl<D: Domain<Intdex>> Tensor<Intdex> for Intdexed<D> {
    fn get(&self, index: Intdex) -> f32 {
        self.u.get(Multintdex::new(vec![index.get()]))
    }

    fn domain(&self) -> &dyn Domain<Intdex> {
        &self.domain
    }
}

impl BoxedTensor<Intdex> for Intdexed<BoxDomain<Intdex>> {
    fn box_domain(&self) -> &BoxDomain<Intdex> {
        &self.domain
    }
}

impl fmt::Display for Intdexed<BoxDomain<Intdex>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
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

// As in a right composition, the domain is the mapping's own domain, and the
// mapping type is a parameter so that reducing over a boxed mapping is known to
// give a boxed, printable tensor.
pub struct ReductionTensor<V: Coords, U: Coords, R: Reduction, M: ?Sized> {
    u: Rc<dyn Tensor<U>>,
    mapping: Rc<M>,
    _marker: PhantomData<fn(V) -> R>,
}

impl<V: Coords, U: Coords, R: Reduction, M: PowersetMapping<V, U> + ?Sized>
    ReductionTensor<V, U, R, M>
{
    pub fn new(u: Rc<dyn Tensor<U>>, mapping: Rc<M>) -> Self {
        Self {
            u,
            mapping,
            _marker: PhantomData,
        }
    }
}

impl<V: Coords, U: Coords, R: Reduction, M: PowersetMapping<V, U> + ?Sized> Tensor<V>
    for ReductionTensor<V, U, R, M>
{
    fn get(&self, index: V) -> f32 {
        let indices = self.mapping.map(index);
        let mut result = R::NEUTRAL;
        for i in &indices {
            result = R::combine(result, self.u.get(i.clone()));
        }
        result
    }

    fn domain(&self) -> &dyn Domain<V> {
        self.mapping.domain()
    }
}

impl<V: Coords, U: Coords, R: Reduction, M: BoxedPowersetMapping<V, U> + ?Sized> BoxedTensor<V>
    for ReductionTensor<V, U, R, M>
{
    fn box_domain(&self) -> &BoxDomain<V> {
        self.mapping.box_domain()
    }
}

impl<V: Coords, U: Coords, R: Reduction, M: BoxedPowersetMapping<V, U> + ?Sized> fmt::Display
    for ReductionTensor<V, U, R, M>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

pub type Sum<V, U, M> = ReductionTensor<V, U, SumReduction, M>;
pub type Product<V, U, M> = ReductionTensor<V, U, ProductReduction, M>;
pub type Sup<V, U, M> = ReductionTensor<V, U, SupReduction, M>;
pub type Inf<V, U, M> = ReductionTensor<V, U, InfReduction, M>;

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
