use std::rc::Rc;

use bten::{
    COO, HRect, Physical, Product, RightComposition, Stride, Sum, Sup, Tensor, diagonalize, eye,
    movedim, negate, ones, square, transpose, zeros,
};

fn main() {
    // Mappings
    let stride = Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3]));
    let repeat = Rc::new(Stride::new(HRect::new(vec![10, 13]), vec![0, 1]));
    let coo = Rc::new(COO::new(
        HRect::new(vec![6, 5]),
        vec![
            vec![0, 0],
            vec![0, 3],
            vec![4, 2],
            vec![5, 4],
            vec![1, 3],
            vec![2, 0],
        ],
    ));
    let coo_vector = Rc::new(COO::new(
        HRect::new(vec![13]),
        vec![vec![0], vec![4], vec![5], vec![1], vec![10], vec![12]],
    ));

    // Tensors
    let physical = Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let strided = Rc::new(RightComposition::new(physical.clone(), stride));
    let sparse_coo_vector = Rc::new(Sum::new(physical.clone(), coo_vector));
    let repeated_sparse = Rc::new(RightComposition::new(sparse_coo_vector.clone(), repeat));
    let sparse_coo = Rc::new(Sum::new(physical.clone(), coo.clone()));
    let prod_coo = Rc::new(Product::new(physical.clone(), coo.clone()));
    let sup_coo = Rc::new(Sup::new(physical.clone(), coo));
    let diag2d = Rc::new(diagonalize(physical.clone(), 2));
    let diag3d = Rc::new(diagonalize(physical.clone(), 3));
    let identity = Rc::new(eye(6));
    let strided_squared = Rc::new(square(strided.clone()));
    let sparse_coo_squared = Rc::new(square(sparse_coo.clone()));

    // Tensors built with the functions only
    let transposed = Rc::new(transpose(strided.clone()));
    let moved = Rc::new(movedim(Rc::new(ones(vec![2, 3, 4])), 0, 2));
    let negated = Rc::new(negate(physical.clone()));
    let zeroed = Rc::new(zeros(vec![2, 3]));
    let transposed_identity = Rc::new(transpose(identity.clone()));

    // Printing
    println!("strided:\n{strided}");
    println!("\nsparse_coo:\n{sparse_coo}");
    println!("\nsparse_coo_vector:\n{sparse_coo_vector}");
    println!("\nrepeated_sparse:\n{repeated_sparse}");
    println!("\nprod_coo:\n{prod_coo}");
    println!("\nsup_coo:\n{sup_coo}");
    println!("\ndiag2d:\n{diag2d}");
    println!("\ndiag3d:\n{diag3d}");
    println!("\nidentity:\n{identity}");
    println!("\nstrided_squared:\n{strided_squared}");
    println!("\nsparse_coo_squared:\n{sparse_coo_squared}");

    let index = [2, 1];
    println!(
        "strided({}, {}) = {}",
        index[0],
        index[1],
        strided.get(&index)
    );

    println!("\ntransposed:\n{transposed}");
    println!("\nmoved:\n{moved}");
    println!("\nnegated:\n{negated}");
    println!("\nzeroed:\n{zeroed}");
    // Transposing an identity built over no data at all gives it back.
    println!("\ntransposed_identity:\n{transposed_identity}");
}
