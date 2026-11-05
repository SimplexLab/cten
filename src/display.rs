//! Printing of boxed tensors: those whose domain is a hrect. The `Display`
//! implementations live here rather than next to each tensor, since they all
//! render the same way, by walking a shape.

use std::borrow::Borrow;
use std::fmt;
use std::fmt::{Display, Write as _};

use crate::function::{Boxed, Compose, Function};
use crate::tensor::{BoxedTensor, Physical};

/// Renders a boxed tensor. Every `Display` implementation below defers to this,
/// and each is written for boxed tensors only.
pub fn format_boxed<A: Display>(tensor: &dyn BoxedTensor<A>) -> String {
    format_tensor(tensor.hrect().shape(), &|index| tensor.apply(index))
}

impl<A: Clone + Display> Display for Physical<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

// A composition is a boxed tensor when the function applied first is one on a
// hrect and the one applied last gives values that print.
impl<F: Function, G: Boxed + Function<Input = [usize]>> Display for Compose<F, G>
where
    G::Output: Borrow<F::Input>,
    F::Output: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_boxed(self))
    }
}

fn format_tensor<A: Display>(shape: &[usize], accessor: &dyn Fn(&[usize]) -> A) -> String {
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

fn max_width<A: Display>(
    shape: &[usize],
    index: &mut [usize],
    dim: usize,
    accessor: &dyn Fn(&[usize]) -> A,
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

fn format_nested<A: Display>(
    out: &mut String,
    shape: &[usize],
    index: &mut [usize],
    dim: usize,
    indent: &str,
    width: usize,
    accessor: &dyn Fn(&[usize]) -> A,
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
