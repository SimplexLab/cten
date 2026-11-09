//! Draft tensor library that considers tensors as functions and operations as
//! compositions or functional transforms.

pub mod display;
pub mod function;
pub mod functions;
pub mod index_functions;
mod lattice;
pub mod multiset;
pub mod reductions;
pub mod rewrite;
pub mod sets;
pub mod signature;
pub mod tensor;

pub use display::format_boxed;
pub use function::{Composition, Function, MultiFunction, ValueFunction, compose, multiset_lift};
pub use functions::{
    Dims, clamp, concatenate, diagonalize, eye, eye_rect, mm, movedim, negate, ones, permute, prod,
    scale, sparse_coo_tensor, sparse_latticed_tensor, square, squeeze, stack, sum, transpose,
    unsqueeze, where_, zeros,
};
pub use index_functions::{
    Affine, COO, Concat, Diagonal, EmptyMapping, IndexFunction, InvAffine, InvProjection,
    MultiIndexFunctionSignature, Permutation, Projection, Sparsity, Squeeze, Stride, Unsqueeze,
    Where,
};
pub use multiset::Multiset;
pub use reductions::{
    Inf, InfReduction, Product, ProductReduction, Reduction, ReductionFunction, Sum, SumReduction,
    Sup, SupReduction,
};
pub use rewrite::{Rewrite, rewrite};
pub use sets::{HRect, IndexSet, JaggedDomain, Multisets, Set, Values};
pub use signature::{HRectSignature, Signature};
pub use tensor::{JaggedTensor, Physical, ReductionTensor, Tensor};
