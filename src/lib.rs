//! Draft tensor library that considers tensors as functions and operations as
//! compositions or functional transforms.

pub mod display;
pub mod domain;
pub mod functions;
pub mod mapping;
pub mod tensor;

pub use display::{BoxedTensor, format_boxed};
pub use domain::{Domain, HRect, JaggedDomain, MultiDomain};
pub use functions::{
    Dims, clamp, concatenate, diagonalize, eye, eye_rect, mm, movedim, negate, ones, permute, pow,
    prod, scale, sparse_coo_tensor, square, squeeze, stack, sum, transpose, unsqueeze, where_,
    zeros,
};
pub use mapping::{
    BoxedMapping, BoxedPowersetMapping, COO, Concat, Diagonal, EmptyMapping, HRectSignature,
    IndexMapping, InvProjection, Permutation, Powerset, PowersetMapping, Projection, Signature,
    Squeeze, Stack, Stride, Unsqueeze, Where,
};
pub use tensor::{
    Inf, InfReduction, LeftComposition, MultiTensor, Physical, Product, ProductReduction,
    Reduction, ReductionTensor, RightComposition, Sum, SumReduction, Sup, SupReduction, Tensor,
};
