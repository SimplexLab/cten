//! Draft tensor library that considers tensors as functions and operations as
//! compositions or functional transforms.

pub mod display;
pub mod domain;
pub mod functions;
mod lattice;
pub mod mapping;
pub mod signature;
pub mod tensor;

pub use display::{BoxedTensor, format_boxed};
pub use domain::{Domain, HRect, JaggedDomain};
pub use functions::{
    Dims, clamp, concatenate, diagonalize, eye, eye_rect, mm, movedim, negate, ones, permute, pow,
    prod, scale, sparse_coo_tensor, sparse_latticed_tensor, square, squeeze, stack, sum, transpose,
    unsqueeze, where_, zeros,
};
pub use mapping::{
    Affine, BoxedMapping, BoxedPowersetMapping, COO, Concat, Diagonal, EmptyMapping, IndexMapping,
    InvAffine, InvProjection, Permutation, Powerset, PowersetMapping, Projection, Squeeze, Stride,
    Unsqueeze, Where,
};
pub use signature::{HRectSignature, Signature};
pub use tensor::{
    Inf, InfReduction, JaggedTensor, LeftComposition, Physical, Product, ProductReduction,
    Reduction, ReductionTensor, RightComposition, Sum, SumReduction, Sup, SupReduction, Tensor,
};
