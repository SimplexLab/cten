//! Rewriting a tensor into an equivalent one that is cheaper to evaluate, with
//! the identities between compositions of Section 4 of the paper. Only
//! Theorem 1 is used so far.

use crate::function::{
    Composition, Function, MultiFunction, ValueFunction, compose, multiset_lift,
};
use crate::index_functions::Sparsity;
use crate::reductions::{Reduction, Sum, SumReduction};
use crate::sets::{Multisets, Values};
use crate::tensor::ReductionTensor;

/// A tensor that an identity of the paper can rewrite into an equivalent one.
pub trait Rewrite: Sized {
    type Rewritten;

    /// The equivalent tensor, or `Err` with the tensor unchanged when the
    /// identity does not apply to it.
    fn rewrite(self) -> Result<Self::Rewritten, Self>;
}

/// Rewrites `tensor` into an equivalent one that is cheaper to evaluate, or
/// gives it back unchanged as `Err` when no identity applies to it.
pub fn rewrite<T: Rewrite>(tensor: T) -> Result<T::Rewritten, T> {
    tensor.rewrite()
}

/// Theorem 1: a function on values after a sparse tensor only has to be applied
/// to its stored values, `f ∘ Σ ∘ Ū ∘ ψ = Σ ∘ (f ∘ U)‾ ∘ ψ`. This holds when
/// `f(0) = 0` and `ψ` sends each index to at most one index, which is checked.
impl<A, U, M> Rewrite for Composition<ValueFunction<A>, ReductionTensor<Sum<A>, U, M>>
where
    A: Clone + PartialEq,
    SumReduction: Reduction<A>,
    U: Function<Codomain = Values<A>>,
    M: Function<Codomain = Multisets<U::Domain>> + Sparsity,
{
    type Rewritten = ReductionTensor<Sum<A>, Composition<ValueFunction<A>, U>, M>;

    fn rewrite(self) -> Result<Self::Rewritten, Self> {
        let zero = <SumReduction as Reduction<A>>::NEUTRAL;
        if self.f.apply(&zero) != zero || !self.g.g.g.is_sparse() {
            return Err(self);
        }
        let Composition {
            f,
            g:
                Composition {
                    g:
                        Composition {
                            f: MultiFunction { f: u },
                            g: psi,
                        },
                    ..
                },
        } = self;
        Ok(compose(
            Sum::new(),
            compose(multiset_lift(compose(f, u)), psi),
        ))
    }
}
