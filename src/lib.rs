//! Draft tensor library that considers tensors as functions and operations as
//! compositions or functional transforms.

pub mod domain;
pub mod index;
pub mod mapping;
pub mod tensor;

pub use domain::{BoxDomain, Domain};
pub use index::{Coords, Index, Intdex, Multintdex, Powerset};
pub use mapping::{
    BoxedMapping, BoxedPowersetMapping, COO, Diagonal, EmptyMapping, IndexMapping, PowersetMapping,
    Stride,
};
pub use tensor::{
    BoxedTensor, Inf, InfReduction, Intdexed, LeftComposition, Physical, Product, ProductReduction,
    Reduction, ReductionTensor, RightComposition, Sum, SumReduction, Sup, SupReduction, Tensor,
    Vector, format_boxed,
};
