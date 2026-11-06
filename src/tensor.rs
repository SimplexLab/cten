use std::rc::Rc;

use crate::function::{Composition, Function, MultiFunction};
use crate::sets::{HRect, JaggedDomain, Set, Values};

/// A function from the indices of `D` to values of type `A`: by default, a
/// tensor on a hrect, which can be printed. Nothing implements this directly:
/// every function of that kind is a tensor.
pub trait Tensor<A = f32, D = HRect>: Function<Domain = D, Codomain = Values<A>> {}

impl<A, D: Set, T: Function<Domain = D, Codomain = Values<A>> + ?Sized> Tensor<A, D> for T {}

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
    type Domain = HRect;
    type Codomain = Values<A>;

    fn apply(&self, index: &[usize]) -> A {
        assert_eq!(
            index.len(),
            1,
            "a physical tensor is indexed by a single coordinate"
        );
        self.values[index[0]].clone()
    }

    fn domain(&self) -> HRect {
        self.domain.clone()
    }

    fn codomain(&self) -> Values<A> {
        Values::new()
    }
}

/// Several tensors read as one: `[t, rest..]` reads part `t` at `rest`. This is
/// the paper's stack, which concatenating, stacking and `where` remap into. Its
/// domain is jagged rather than a hrect, so it is not printable.
pub struct JaggedTensor<A = f32> {
    parts: Vec<Rc<dyn Tensor<A>>>,
    domain: JaggedDomain,
}

impl<A> JaggedTensor<A> {
    pub fn new(parts: Vec<Rc<dyn Tensor<A>>>) -> Self {
        let domain = JaggedDomain::new(parts.iter().map(|part| part.domain()).collect());
        Self { parts, domain }
    }
}

impl<A> Function for JaggedTensor<A> {
    type Domain = JaggedDomain;
    type Codomain = Values<A>;

    fn apply(&self, index: &[usize]) -> A {
        self.parts[index[0]].apply(&index[1..])
    }

    fn domain(&self) -> JaggedDomain {
        self.domain.clone()
    }

    fn codomain(&self) -> Values<A> {
        Values::new()
    }
}

/// `Ω ∘ Ū ∘ ψ`, the reduction tensor of the paper, with reduction `R`,
/// underlying tensor `U` and index function `M`: at each index, the reduction
/// of the multiset of the values of `U` at the indices `M` maps it to. Built
/// as `compose(Sum::new(), compose(multiset_lift(u), psi))`.
pub type ReductionTensor<R, U, M> = Composition<R, Composition<MultiFunction<U>, M>>;
