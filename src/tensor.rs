use std::fmt;
use std::fmt::Write as _;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::domain::{BoxDomain, Domain};
use crate::mapping::{BoxedMapping, BoxedPowersetMapping, IndexMapping, PowersetMapping};

/// A mapping from an index set to a value set. The index set is the tensor's
/// domain; the value set is fixed to the float numbers for now, for simplicity.
pub trait Tensor {
    fn get(&self, index: &[usize]) -> f32;

    fn domain(&self) -> &dyn Domain;
}

/// A tensor whose domain is a box. Only these can be printed, since formatting
/// walks a shape and visits every index of it.
pub trait BoxedTensor: Tensor {
    fn box_domain(&self) -> &BoxDomain;
}

/// Renders a boxed tensor. Every `Display` implementation below defers to this,
/// and each is written for the boxed instantiation of its tensor only.
pub fn format_boxed(tensor: &dyn BoxedTensor) -> String {
    format_tensor(tensor.box_domain().shape(), &|index| tensor.get(index))
}

/// Checks that a mapping lands on the indices the underlying tensor reads,
/// which is all that can be checked without enumerating a domain. In
/// particular, a mapping into flat positions needs a one-dimensional tensor
/// under it.
fn check_composable(codomain: &dyn Domain, u: &dyn Tensor) {
    assert_eq!(
        codomain.ndim(),
        u.domain().ndim(),
        "the mapping lands on {}-dimensional indices, but the underlying tensor is indexed by {} coordinates",
        codomain.ndim(),
        u.domain().ndim()
    );
}

pub struct Physical {
    values: Vec<f32>,
    domain: BoxDomain,
}

impl Physical {
    pub fn new(values: Vec<f32>) -> Self {
        let domain = BoxDomain::new(vec![values.len()]);
        Self { values, domain }
    }
}

impl Tensor for Physical {
    fn get(&self, index: &[usize]) -> f32 {
        assert_eq!(
            index.len(),
            1,
            "a physical tensor is indexed by a single coordinate"
        );
        self.values[index[0]]
    }

    fn domain(&self) -> &dyn Domain {
        &self.domain
    }
}

impl BoxedTensor for Physical {
    fn box_domain(&self) -> &BoxDomain {
        &self.domain
    }
}

impl fmt::Display for Physical {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

// The domain is the mapping's own domain, so it is never passed in. The mapping
// type is a parameter rather than always `dyn IndexMapping` so that composing
// over a boxed mapping is known to give a boxed, printable tensor. Pass
// `dyn BoxedMapping` or `dyn IndexMapping` for it to erase the mapping.
pub struct RightComposition<M: ?Sized> {
    u: Rc<dyn Tensor>,
    mapping: Rc<M>,
}

impl<M: IndexMapping + ?Sized> RightComposition<M> {
    pub fn new(u: Rc<dyn Tensor>, mapping: Rc<M>) -> Self {
        check_composable(mapping.codomain(), &*u);
        Self { u, mapping }
    }
}

impl<M: IndexMapping + ?Sized> Tensor for RightComposition<M> {
    fn get(&self, virtual_index: &[usize]) -> f32 {
        let underlying_index = self.mapping.map(virtual_index);
        self.u.get(&underlying_index)
    }

    fn domain(&self) -> &dyn Domain {
        self.mapping.domain()
    }
}

impl<M: BoxedMapping + ?Sized> BoxedTensor for RightComposition<M> {
    fn box_domain(&self) -> &BoxDomain {
        self.mapping.box_domain()
    }
}

impl<M: BoxedMapping + ?Sized> fmt::Display for RightComposition<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

// Transforming the values leaves the index set alone, so the domain is the
// underlying tensor's. Its type is a parameter so that a transform of a boxed
// tensor is known to be boxed, and printable, in turn.
pub struct LeftComposition<T: ?Sized> {
    u: Rc<T>,
    f: fn(f32) -> f32,
}

impl<T: Tensor + ?Sized> LeftComposition<T> {
    pub fn new(u: Rc<T>, f: fn(f32) -> f32) -> Self {
        Self { u, f }
    }
}

impl<T: Tensor + ?Sized> Tensor for LeftComposition<T> {
    fn get(&self, index: &[usize]) -> f32 {
        (self.f)(self.u.get(index))
    }

    fn domain(&self) -> &dyn Domain {
        self.u.domain()
    }
}

impl<T: BoxedTensor + ?Sized> BoxedTensor for LeftComposition<T> {
    fn box_domain(&self) -> &BoxDomain {
        self.u.box_domain()
    }
}

impl<T: BoxedTensor + ?Sized> fmt::Display for LeftComposition<T> {
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
pub struct ReductionTensor<R: Reduction, M: ?Sized> {
    u: Rc<dyn Tensor>,
    mapping: Rc<M>,
    _marker: PhantomData<fn() -> R>,
}

impl<R: Reduction, M: PowersetMapping + ?Sized> ReductionTensor<R, M> {
    pub fn new(u: Rc<dyn Tensor>, mapping: Rc<M>) -> Self {
        check_composable(mapping.codomain(), &*u);
        Self {
            u,
            mapping,
            _marker: PhantomData,
        }
    }
}

impl<R: Reduction, M: PowersetMapping + ?Sized> Tensor for ReductionTensor<R, M> {
    fn get(&self, index: &[usize]) -> f32 {
        let indices = self.mapping.map(index);
        let mut result = R::NEUTRAL;
        for i in &indices {
            result = R::combine(result, self.u.get(i));
        }
        result
    }

    fn domain(&self) -> &dyn Domain {
        self.mapping.domain()
    }
}

impl<R: Reduction, M: BoxedPowersetMapping + ?Sized> BoxedTensor for ReductionTensor<R, M> {
    fn box_domain(&self) -> &BoxDomain {
        self.mapping.box_domain()
    }
}

impl<R: Reduction, M: BoxedPowersetMapping + ?Sized> fmt::Display for ReductionTensor<R, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

pub type Sum<M> = ReductionTensor<SumReduction, M>;
pub type Product<M> = ReductionTensor<ProductReduction, M>;
pub type Sup<M> = ReductionTensor<SupReduction, M>;
pub type Inf<M> = ReductionTensor<InfReduction, M>;

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
