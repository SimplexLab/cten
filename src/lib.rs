//! Draft tensor library that considers tensors as functions and operations as
//! compositions or functional transforms.

pub mod domain;
pub mod mapping;
pub mod tensor;

pub use domain::{BoxDomain, Domain};
pub use mapping::{
    BoxedMapping, BoxedPowersetMapping, COO, Diagonal, EmptyMapping, IndexMapping, Powerset,
    PowersetMapping, Stride,
};
pub use tensor::{
    BoxedTensor, Inf, InfReduction, LeftComposition, Physical, Product, ProductReduction,
    Reduction, ReductionTensor, RightComposition, Sum, SumReduction, Sup, SupReduction, Tensor,
    format_boxed,
};
