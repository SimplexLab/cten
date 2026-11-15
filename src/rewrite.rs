//! Rewriting a tensor into an equivalent one that is cheaper to evaluate, with
//! the identities between compositions of Section 4 of the paper. So far:
//! Theorem 1, which Lemma 6 extends from sums to any reduction.

use std::rc::Rc;

use crate::sets::Set;
use crate::tensor::Tensor;

/// Rewrites `tensor` into an equivalent one that is cheaper to evaluate,
/// wherever an identity of the paper applies inside it, and leaves it as it is
/// elsewhere. Any tensor can be given, and so can the result, again.
///
/// A function `f` after a reduction tensor `Ω ∘ Ū ∘ ψ` is rewritten as
/// `Ω ∘ f̄ ∘ Ū ∘ ψ`, which is `Ω ∘ (f ∘ U)‾ ∘ ψ`, so that `f` is only applied to
/// the stored values. This needs `ψ` to send each index to at most one index,
/// and `f` to keep the neutral element of `Ω`, such as `f(0) = 0` for a sum.
pub fn rewrite<A: 'static, D: Set + 'static, T: Tensor<A, D> + 'static>(
    tensor: T,
) -> Rc<dyn Tensor<A, D>> {
    match tensor.rewritten() {
        Some(rewritten) => Rc::new(rewritten),
        None => Rc::new(tensor),
    }
}
