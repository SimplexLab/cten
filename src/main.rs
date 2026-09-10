use std::rc::Rc;

use bten::{
    COO, Diagonal, EmptyMapping, Intdex, Intdexed, LeftComposition, Multintdex, Physical, Product,
    RightComposition, Stride, Sum, Sup, Tensor,
};

fn square(x: f32) -> f32 {
    x * x
}

fn main() {
    // Mappings
    let stride = Rc::new(Stride::new(vec![1, 3]));
    let repeat = Rc::new(Stride::new(vec![0, 1]));
    let coo = Rc::new(COO::new(vec![
        vec![0, 0],
        vec![0, 3],
        vec![4, 2],
        vec![5, 4],
        vec![1, 3],
        vec![2, 0],
    ]));
    let coo_vector = Rc::new(COO::new(vec![
        vec![0],
        vec![4],
        vec![5],
        vec![1],
        vec![10],
        vec![12],
    ]));
    let diagonalize = Rc::new(Diagonal::new());
    let empty_map = Rc::new(EmptyMapping::<Intdex, Intdex>::new());

    // Tensors
    let physical = Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let strided = Rc::new(RightComposition::<Intdex>::new(
        physical.clone(),
        stride,
        vec![3, 2],
    ));
    let sparse_coo_vector = Rc::new(Sum::<Multintdex, Intdex>::new(
        physical.clone(),
        coo_vector,
        vec![13],
    ));
    let intdexed_sparse_coo_vector = Rc::new(Intdexed::new(sparse_coo_vector.clone()));
    let repeated_sparse = Rc::new(RightComposition::<Intdex>::new(
        intdexed_sparse_coo_vector,
        repeat,
        vec![10, 13],
    ));
    let sparse_coo = Rc::new(Sum::<Multintdex, Intdex>::new(
        physical.clone(),
        coo.clone(),
        vec![6, 5],
    ));
    let prod_coo = Rc::new(Product::<Multintdex, Intdex>::new(
        physical.clone(),
        coo.clone(),
        vec![6, 5],
    ));
    let sup_coo = Rc::new(Sup::<Multintdex, Intdex>::new(
        physical.clone(),
        coo,
        vec![6, 5],
    ));
    let diag2d = Rc::new(Sum::<Multintdex, Intdex>::new(
        physical.clone(),
        diagonalize.clone(),
        vec![6, 6],
    ));
    let diag3d = Rc::new(Sum::<Multintdex, Intdex>::new(
        physical,
        diagonalize.clone(),
        vec![6, 6, 6],
    ));
    let empty = Rc::new(Physical::new(vec![]));
    let ones = Rc::new(Product::<Intdex, Intdex>::new(empty, empty_map, vec![6]));
    let identity = Rc::new(Sum::<Multintdex, Intdex>::new(
        ones,
        diagonalize,
        vec![6, 6],
    ));
    let strided_squared = Rc::new(LeftComposition::<Multintdex>::new(
        strided.clone(),
        square,
        vec![3, 2],
    ));
    let sparse_coo_squared = Rc::new(LeftComposition::<Multintdex>::new(
        sparse_coo.clone(),
        square,
        vec![6, 5],
    ));

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

    let index = vec![2, 1];
    println!(
        "strided({}, {}) = {}",
        index[0],
        index[1],
        strided.get(Multintdex::new(index.clone()))
    );
}
