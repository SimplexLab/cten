use std::marker::PhantomData;
use std::rc::Rc;

use crate::display::BoxedTensor;
use crate::domain::{Domain, HRect, JaggedDomain};
use crate::mapping::{BoxedMapping, BoxedPowersetMapping, IndexMapping, PowersetMapping};
use crate::multiset::Multiset;

/// A mapping from an index set to a value set. The index set is the tensor's
/// domain, and the value set is its value type `A`, `f32` unless said
/// otherwise.
pub trait Tensor<A = f32> {
    fn get(&self, index: &[usize]) -> A;

    fn domain(&self) -> &dyn Domain;
}

/// Checks that a mapping lands on the indices the underlying tensor reads,
/// which is all that can be checked without enumerating a domain. In
/// particular, a mapping into flat positions needs a one-dimensional tensor
/// under it.
fn check_composable<A>(codomain: &dyn Domain, u: &dyn Tensor<A>) {
    assert_eq!(
        codomain.ndim(),
        u.domain().ndim(),
        "the mapping lands on {}-dimensional indices, but the underlying tensor is indexed by {} coordinates",
        codomain.ndim(),
        u.domain().ndim()
    );
}

pub struct Physical<A = f32> {
    values: Vec<A>,
    domain: HRect,
}

impl<A> Physical<A> {
    pub fn new(values: Vec<A>) -> Self {
        let domain = HRect::new(vec![values.len()]);
        Self { values, domain }
    }
}

impl<A: Clone> Tensor<A> for Physical<A> {
    fn get(&self, index: &[usize]) -> A {
        assert_eq!(
            index.len(),
            1,
            "a physical tensor is indexed by a single coordinate"
        );
        self.values[index[0]].clone()
    }

    fn domain(&self) -> &dyn Domain {
        &self.domain
    }
}

impl<A: Clone> BoxedTensor<A> for Physical<A> {
    fn hrect(&self) -> &HRect {
        &self.domain
    }
}

/// Several tensors read as one: `[t, rest..]` reads part `t` at `rest`. This is
/// the underlying that concatenating, stacking and `where` remap into. Its domain
/// is jagged rather than a hrect, so it is not printable.
pub struct JaggedTensor<A = f32> {
    parts: Vec<Rc<dyn BoxedTensor<A>>>,
    domain: JaggedDomain,
}

impl<A> JaggedTensor<A> {
    pub fn new(parts: Vec<Rc<dyn BoxedTensor<A>>>) -> Self {
        let domain = JaggedDomain::new(parts.iter().map(|part| part.hrect().clone()).collect());
        Self { parts, domain }
    }

    pub fn jagged_domain(&self) -> &JaggedDomain {
        &self.domain
    }
}

impl<A> Tensor<A> for JaggedTensor<A> {
    fn get(&self, index: &[usize]) -> A {
        self.parts[index[0]].get(&index[1..])
    }

    fn domain(&self) -> &dyn Domain {
        &self.domain
    }
}

// The domain is the mapping's own domain, so it is never passed in. The mapping
// type is a parameter rather than always `dyn IndexMapping` so that composing
// over a boxed mapping is known to give a boxed, printable tensor. Pass
// `dyn BoxedMapping` or `dyn IndexMapping` for it to erase the mapping. The
// values are the underlying tensor's, read elsewhere.
pub struct RightComposition<M: ?Sized, A = f32> {
    u: Rc<dyn Tensor<A>>,
    mapping: Rc<M>,
}

impl<M: IndexMapping + ?Sized, A> RightComposition<M, A> {
    pub fn new(u: Rc<dyn Tensor<A>>, mapping: Rc<M>) -> Self {
        check_composable(mapping.codomain(), &*u);
        Self { u, mapping }
    }
}

impl<M: IndexMapping + ?Sized, A> Tensor<A> for RightComposition<M, A> {
    fn get(&self, virtual_index: &[usize]) -> A {
        let underlying_index = self.mapping.map(virtual_index);
        self.u.get(&underlying_index)
    }

    fn domain(&self) -> &dyn Domain {
        self.mapping.domain()
    }
}

impl<M: BoxedMapping + ?Sized, A> BoxedTensor<A> for RightComposition<M, A> {
    fn hrect(&self) -> &HRect {
        self.mapping.hrect()
    }
}

// Transforming the values leaves the index set alone, so the domain is the
// underlying tensor's. Its type is a parameter so that a transform of a boxed
// tensor is known to be boxed, and printable, in turn. The transform may change
// the value type, from `A` to `B`, as a comparison turns numbers into booleans.
pub struct LeftComposition<U: ?Sized, A = f32, B = A> {
    u: Rc<U>,
    // Held as a closure rather than a plain `fn` pointer so that a transform can
    // carry values of its own, as `scale` and `clamp` do.
    f: Rc<dyn Fn(A) -> B>,
}

impl<U: Tensor<A> + ?Sized, A, B> LeftComposition<U, A, B> {
    pub fn new(u: Rc<U>, f: impl Fn(A) -> B + 'static) -> Self {
        Self { u, f: Rc::new(f) }
    }
}

impl<U: Tensor<A> + ?Sized, A, B> Tensor<B> for LeftComposition<U, A, B> {
    fn get(&self, index: &[usize]) -> B {
        (self.f)(self.u.get(index))
    }

    fn domain(&self) -> &dyn Domain {
        self.u.domain()
    }
}

impl<U: BoxedTensor<A> + ?Sized, A, B> BoxedTensor<B> for LeftComposition<U, A, B> {
    fn hrect(&self) -> &HRect {
        self.u.hrect()
    }
}

/// A way of combining the values of a tensor of value type `A` into one.
pub trait Reduction<A> {
    const NEUTRAL: A;
    fn combine(a: A, b: A) -> A;

    /// Reduces a multiset to a single value: the neutral element for the empty
    /// multiset, its element for a multiset of one, and otherwise its elements
    /// combined in any order, as `combine` is associative and commutative.
    fn reduce(multiset: Multiset<A>) -> A {
        multiset.into_iter().fold(Self::NEUTRAL, Self::combine)
    }
}

pub struct SumReduction;
pub struct ProductReduction;
pub struct SupReduction;
pub struct InfReduction;

// The four reductions for each value type they apply to: a sum starts from 0, a
// product from 1, and a sup and an inf from the bottom and top of the type.
macro_rules! reductions {
    ($($value:ty: $zero:expr, $one:expr, $bottom:expr, $top:expr;)*) => {$(
        impl Reduction<$value> for SumReduction {
            const NEUTRAL: $value = $zero;
            fn combine(a: $value, b: $value) -> $value {
                a + b
            }
        }

        impl Reduction<$value> for ProductReduction {
            const NEUTRAL: $value = $one;
            fn combine(a: $value, b: $value) -> $value {
                a * b
            }
        }

        impl Reduction<$value> for SupReduction {
            const NEUTRAL: $value = $bottom;
            fn combine(a: $value, b: $value) -> $value {
                a.max(b)
            }
        }

        impl Reduction<$value> for InfReduction {
            const NEUTRAL: $value = $top;
            fn combine(a: $value, b: $value) -> $value {
                a.min(b)
            }
        }
    )*};
}

reductions! {
    f32: 0.0, 1.0, f32::NEG_INFINITY, f32::INFINITY;
    f64: 0.0, 1.0, f64::NEG_INFINITY, f64::INFINITY;
    i64: 0, 1, i64::MIN, i64::MAX;
}

/// The extension of a tensor `U` to multisets, written `Ū`: it maps a multiset of
/// indices to the multiset of the values of `U` at them, repeats included. Its
/// indices being multisets, it is not a tensor itself; composed with an index
/// function, it gives one.
pub struct Bar<A = f32> {
    u: Rc<dyn Tensor<A>>,
}

impl<A> Bar<A> {
    pub fn new(u: Rc<dyn Tensor<A>>) -> Self {
        Self { u }
    }

    pub fn get(&self, indices: &Multiset<Vec<usize>>) -> Multiset<A> {
        indices.iter().map(|index| self.u.get(index)).collect()
    }
}

// `Ū ∘ ψ`: at each index, the multiset of the values of `U` at the indices `ψ`
// maps it to. This is the multiset-valued tensor a reduction reduces. As in a
// right composition, the domain is the mapping's own domain, and the mapping
// type is a parameter so that a boxed mapping is known to give a boxed,
// printable tensor.
pub struct BarComposition<M: ?Sized, A = f32> {
    bar: Bar<A>,
    mapping: Rc<M>,
}

impl<M: PowersetMapping + ?Sized, A> BarComposition<M, A> {
    pub fn new(u: Rc<dyn Tensor<A>>, mapping: Rc<M>) -> Self {
        check_composable(mapping.codomain(), &*u);
        Self {
            bar: Bar::new(u),
            mapping,
        }
    }
}

impl<M: PowersetMapping + ?Sized, A> Tensor<Multiset<A>> for BarComposition<M, A> {
    fn get(&self, index: &[usize]) -> Multiset<A> {
        self.bar.get(&self.mapping.map(index))
    }

    fn domain(&self) -> &dyn Domain {
        self.mapping.domain()
    }
}

impl<M: BoxedPowersetMapping + ?Sized, A> BoxedTensor<Multiset<A>> for BarComposition<M, A> {
    fn hrect(&self) -> &HRect {
        self.mapping.hrect()
    }
}

// `Ω ∘ Ū ∘ ψ`: the reduction `R` of the multisets that a bar composition gathers.
pub struct ReductionTensor<R, M: ?Sized, A = f32> {
    gathered: BarComposition<M, A>,
    _marker: PhantomData<fn() -> R>,
}

impl<R: Reduction<A>, M: PowersetMapping + ?Sized, A> ReductionTensor<R, M, A> {
    pub fn new(u: Rc<dyn Tensor<A>>, mapping: Rc<M>) -> Self {
        Self {
            gathered: BarComposition::new(u, mapping),
            _marker: PhantomData,
        }
    }
}

impl<R: Reduction<A>, M: PowersetMapping + ?Sized, A> Tensor<A> for ReductionTensor<R, M, A> {
    fn get(&self, index: &[usize]) -> A {
        R::reduce(self.gathered.get(index))
    }

    fn domain(&self) -> &dyn Domain {
        self.gathered.domain()
    }
}

impl<R: Reduction<A>, M: BoxedPowersetMapping + ?Sized, A> BoxedTensor<A>
    for ReductionTensor<R, M, A>
{
    fn hrect(&self) -> &HRect {
        self.gathered.hrect()
    }
}

pub type Sum<M, A = f32> = ReductionTensor<SumReduction, M, A>;
pub type Product<M, A = f32> = ReductionTensor<ProductReduction, M, A>;
pub type Sup<M, A = f32> = ReductionTensor<SupReduction, M, A>;
pub type Inf<M, A = f32> = ReductionTensor<InfReduction, M, A>;
