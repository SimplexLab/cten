//! Printing of boxed tensors: those whose domain is a hrect. The `Display`
//! implementations live here rather than next to each tensor, since they all
//! render the same way, by walking a shape.

use std::fmt;
use std::fmt::{Display, Write as _};

use crate::domain::HRect;
use crate::mapping::{BoxedMapping, BoxedPowersetMapping};
use crate::tensor::{
    LeftComposition, Physical, Reduction, ReductionTensor, RightComposition, Tensor,
};

/// A tensor whose domain is a hrect. Only these can be printed, since formatting
/// walks a shape and visits every index of it.
pub trait BoxedTensor<T = f32>: Tensor<T> {
    fn hrect(&self) -> &HRect;
}

/// Renders a boxed tensor. Every `Display` implementation below defers to this,
/// and each is written for the boxed instantiation of its tensor only.
pub fn format_boxed<T: Display>(tensor: &dyn BoxedTensor<T>) -> String {
    format_tensor(tensor.hrect().shape(), &|index| tensor.get(index))
}

impl<T: Clone + Display> Display for Physical<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

impl<M: BoxedMapping + ?Sized, T: Display> Display for RightComposition<M, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

impl<U: BoxedTensor<A> + ?Sized, A, B: Display> Display for LeftComposition<U, A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

impl<R: Reduction<T>, M: BoxedPowersetMapping + ?Sized, T: Display> Display
    for ReductionTensor<R, M, T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

fn format_tensor<T: Display>(shape: &[usize], accessor: &dyn Fn(&[usize]) -> T) -> String {
    if shape.is_empty() {
        return format!("tensor({})", accessor(&[]));
    }

    let mut index = vec![0; shape.len()];
    let width = max_width(shape, &mut index, 0, accessor);
    let mut out = "tensor(".to_string();
    // "tensor(" is 8 characters wide counting the opening bracket, so rows of
    // the outermost dimension line up under it.
    format_nested(
        &mut out,
        shape,
        &mut index,
        0,
        &" ".repeat(8),
        width,
        accessor,
    );
    out.push(')');
    out
}

fn max_width<T: Display>(
    shape: &[usize],
    index: &mut [usize],
    dim: usize,
    accessor: &dyn Fn(&[usize]) -> T,
) -> usize {
    let mut width = 0;
    for i in 0..shape[dim] {
        index[dim] = i;
        width = width.max(if dim == shape.len() - 1 {
            accessor(index).to_string().len()
        } else {
            max_width(shape, index, dim + 1, accessor)
        });
    }
    width
}

fn format_nested<T: Display>(
    out: &mut String,
    shape: &[usize],
    index: &mut [usize],
    dim: usize,
    indent: &str,
    width: usize,
    accessor: &dyn Fn(&[usize]) -> T,
) {
    let innermost = dim == shape.len() - 1;
    out.push('[');
    for i in 0..shape[dim] {
        index[dim] = i;
        if innermost {
            let _ = write!(out, "{:>width$}", accessor(index));
        } else {
            format_nested(
                out,
                shape,
                index,
                dim + 1,
                &format!("{indent} "),
                width,
                accessor,
            );
        }
        if i + 1 < shape[dim] {
            if innermost {
                out.push_str(", ");
            } else {
                out.push_str(",\n");
                // Separate blocks of rank 3 and above with a blank line.
                if dim + 2 < shape.len() {
                    out.push('\n');
                }
                out.push_str(indent);
            }
        }
    }
    out.push(']');
}
