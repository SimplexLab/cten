//! Reductions: functions from multisets of values to values, `Ω` in the paper.

use std::marker::PhantomData;

use crate::function::Function;
use crate::multiset::Multiset;
use crate::sets::{Multisets, Values};

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

/// The reduction `R` as a function from multisets of values to values: the
/// neutral element for the empty multiset, the element of a multiset of one,
/// and otherwise its elements combined in any order, as `combine` is
/// associative and commutative.
pub struct ReductionFunction<R, A = f32> {
    _marker: PhantomData<fn() -> (R, A)>,
}

impl<R, A> ReductionFunction<R, A> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

// Written out rather than derived, which would require `R: Default` and `A: Default`.
impl<R, A> Default for ReductionFunction<R, A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Reduction<A>, A: Clone> Function for ReductionFunction<R, A> {
    type Domain = Multisets<Values<A>>;
    type Codomain = Values<A>;

    fn apply(&self, multiset: &Multiset<A>) -> A {
        multiset.iter().cloned().fold(R::NEUTRAL, R::combine)
    }

    fn domain(&self) -> Multisets<Values<A>> {
        Multisets::new(Values::new())
    }

    fn codomain(&self) -> Values<A> {
        Values::new()
    }
}

pub type Sum<A = f32> = ReductionFunction<SumReduction, A>;
pub type Product<A = f32> = ReductionFunction<ProductReduction, A>;
pub type Sup<A = f32> = ReductionFunction<SupReduction, A>;
pub type Inf<A = f32> = ReductionFunction<InfReduction, A>;
