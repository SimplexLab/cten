use std::marker::PhantomData;
use std::rc::Rc;

use crate::domain::{Domain, HRect, JaggedDomain};
use crate::function::{Boxed, Compose, Function, MultisetLift};
use crate::multiset::Multiset;

/// A function from indices to values, with values of type `A`, `f32` unless
/// said otherwise. Nothing implements this directly: every function of that
/// type is a tensor.
pub trait Tensor<A = f32>: Function<Input = [usize], Output = A> {}

impl<A, T: Function<Input = [usize], Output = A> + ?Sized> Tensor<A> for T {}

/// A tensor whose domain is a hrect, hence a printable one.
pub trait BoxedTensor<A = f32>: Tensor<A> + Boxed {}

impl<A, T: Tensor<A> + Boxed + ?Sized> BoxedTensor<A> for T {}

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

impl<A: Clone> Function for Physical<A> {
    type Input = [usize];
    type Output = A;

    fn apply(&self, index: &[usize]) -> A {
        assert_eq!(
            index.len(),
            1,
            "a physical tensor is indexed by a single coordinate"
        );
        self.values[index[0]].clone()
    }

    fn domain(&self) -> Option<&dyn Domain> {
        Some(&self.domain)
    }
}

impl<A: Clone> Boxed for Physical<A> {
    fn hrect(&self) -> &HRect {
        &self.domain
    }
}

/// Several tensors read as one: `[t, rest..]` reads part `t` at `rest`. This is
/// the paper's stack, which concatenating, stacking and `where` remap into. Its
/// domain is jagged rather than a hrect, so it is not printable.
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

impl<A> Function for JaggedTensor<A> {
    type Input = [usize];
    type Output = A;

    fn apply(&self, index: &[usize]) -> A {
        self.parts[index[0]].apply(&index[1..])
    }

    fn domain(&self) -> Option<&dyn Domain> {
        Some(&self.domain)
    }
}

/// A way of combining values of type `A` into one, from a neutral element.
pub trait Reduction<A> {
    const NEUTRAL: A;
    fn combine(a: A, b: A) -> A;
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

/// The reduction `R` as a function from multisets of values to values, `Ω` in
/// the paper: the neutral element for the empty multiset, the element of a
/// multiset of one, and otherwise its elements combined in any order, as
/// `combine` is associative and commutative.
pub struct Reduce<R, A = f32> {
    _marker: PhantomData<fn() -> (R, A)>,
}

impl<R, A> Reduce<R, A> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

// Written out rather than derived, which would require `R: Default` and `A: Default`.
impl<R, A> Default for Reduce<R, A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Reduction<A>, A: Clone> Function for Reduce<R, A> {
    type Input = Multiset<A>;
    type Output = A;

    fn apply(&self, multiset: &Multiset<A>) -> A {
        multiset.iter().cloned().fold(R::NEUTRAL, R::combine)
    }
}

pub type Sum<A = f32> = Reduce<SumReduction, A>;
pub type Product<A = f32> = Reduce<ProductReduction, A>;
pub type Sup<A = f32> = Reduce<SupReduction, A>;
pub type Inf<A = f32> = Reduce<InfReduction, A>;

/// `Ω ∘ Ū ∘ ψ`, the reduction tensor of the paper, with reduction `R`,
/// underlying tensor `U` and index function `M`: at each index, the reduction
/// of the multiset of the values of `U` at the indices `M` maps it to. Built
/// as `compose(Sum::new(), compose(multiset_lift(u), psi))`.
pub type ReductionTensor<R, U, M> = Compose<R, Compose<MultisetLift<U>, M>>;
