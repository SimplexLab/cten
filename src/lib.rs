//! Draft tensor library that considers tensors as functions and operations as
//! compositions or functional transforms.

pub mod display;
pub mod domain;
pub mod functions;
pub mod mapping;
pub mod tensor;

pub use display::{BoxedTensor, format_boxed};
pub use domain::{Domain, HRect, JaggedDomain};
pub use functions::{
    clamp, concatenate, diagonalize, eye, eye_rect, movedim, negate, ones, permute, pow, scale,
    sparse_coo_tensor, square, squeeze, squeeze_dim, stack, transpose, unsqueeze, where_, zeros,
};
pub use mapping::{
    BoxedMapping, BoxedPowersetMapping, COO, Concat, Diagonal, EmptyMapping, HRectSignature,
    IndexMapping, Permutation, Powerset, PowersetMapping, Signature, Squeeze, Stride, Unsqueeze,
    Where,
};
pub use tensor::{
    Inf, InfReduction, JaggedTensor, LeftComposition, Physical, Product, ProductReduction,
    Reduction, ReductionTensor, RightComposition, Sum, SumReduction, Sup, SupReduction, Tensor,
};
