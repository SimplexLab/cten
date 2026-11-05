//! Draft tensor library that considers tensors as functions and operations as
//! compositions or functional transforms.

pub mod display;
pub mod domain;
pub mod function;
pub mod functions;
mod lattice;
pub mod mapping;
pub mod multiset;
pub mod set;
pub mod signature;
pub mod tensor;

pub use display::format_boxed;
pub use domain::{Domain, HRect, JaggedDomain};
pub use function::{Compose, Function, MultiFunction, ValueFunction, compose, multiset_lift};
pub use functions::{
    Dims, clamp, concatenate, diagonalize, eye, eye_rect, mm, movedim, negate, ones, permute, prod,
    scale, sparse_coo_tensor, sparse_latticed_tensor, square, squeeze, stack, sum, transpose,
    unsqueeze, where_, zeros,
};
pub use mapping::{
    Affine, COO, Concat, Diagonal, EmptyMapping, IndexFunctionSignature, InvAffine, InvProjection,
    Mapping, Permutation, Projection, Squeeze, Stride, Unsqueeze, Where,
};
pub use multiset::Multiset;
pub use set::{Multisets, Set, Values};
pub use signature::{HRectSignature, Signature};
pub use tensor::{
    Inf, InfReduction, JaggedTensor, Physical, Product, ProductReduction, Reduce, Reduction,
    ReductionTensor, Sum, SumReduction, Sup, SupReduction, Tensor,
};
