//! Draft tensor library that considers tensors as functions and operations as
//! compositions or functional transforms.

pub mod index;
pub mod mapping;
pub mod tensor;

pub use index::{FromCoords, Index, Intdex, Multintdex, Powerset};
pub use mapping::{COO, Diagonal, EmptyMapping, IndexMapping, PowersetMapping, Stride};
pub use tensor::{
    Inf, InfReduction, Intdexed, LeftComposition, Physical, Product, ProductReduction, Reduction,
    ReductionTensor, RightComposition, Sum, SumReduction, Sup, SupReduction, Tensor, Vector,
};
