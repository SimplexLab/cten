//! Draft tensor library that considers tensors as functions and operations as
//! compositions or functional transforms.

pub mod display;
pub mod domain;
pub mod functions;
pub mod mapping;
pub mod tensor;

pub use display::{BoxedTensor, format_boxed};
pub use domain::{Domain, HRect};
pub use functions::{
    clamp, diagonalize, eye, eye_rect, movedim, negate, ones, permute, pow, scale, square,
    transpose, zeros,
};
pub use mapping::{
    BoxedMapping, BoxedPowersetMapping, COO, Diagonal, EmptyMapping, IndexMapping, Permutation,
    Powerset, PowersetMapping, Stride,
};
pub use tensor::{
    Inf, InfReduction, LeftComposition, Physical, Product, ProductReduction, Reduction,
    ReductionTensor, RightComposition, Sum, SumReduction, Sup, SupReduction, Tensor,
};
