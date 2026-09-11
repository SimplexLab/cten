use std::marker::PhantomData;
use std::rc::Rc;

use crate::display::BoxedTensor;
use crate::domain::{BoxDomain, Domain};
use crate::mapping::{BoxedMapping, BoxedPowersetMapping, IndexMapping, PowersetMapping};

/// A mapping from an index set to a value set. The index set is the tensor's
/// domain; the value set is fixed to the float numbers for now, for simplicity.
pub trait Tensor {
    fn get(&self, index: &[usize]) -> f32;

    fn domain(&self) -> &dyn Domain;
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

// Transforming the values leaves the index set alone, so the domain is the
// underlying tensor's. Its type is a parameter so that a transform of a boxed
// tensor is known to be boxed, and printable, in turn.
pub struct LeftComposition<T: ?Sized> {
    u: Rc<T>,
    // Held as a closure rather than a plain `fn` pointer so that a transform can
    // carry values of its own, as `scale` and `clamp` do.
    f: Rc<dyn Fn(f32) -> f32>,
}

impl<T: Tensor + ?Sized> LeftComposition<T> {
    pub fn new(u: Rc<T>, f: impl Fn(f32) -> f32 + 'static) -> Self {
        Self { u, f: Rc::new(f) }
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

pub type Sum<M> = ReductionTensor<SumReduction, M>;
pub type Product<M> = ReductionTensor<ProductReduction, M>;
pub type Sup<M> = ReductionTensor<SupReduction, M>;
pub type Inf<M> = ReductionTensor<InfReduction, M>;
