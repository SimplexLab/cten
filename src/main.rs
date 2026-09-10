use std::rc::Rc;

use bten::{
    BoxDomain, COO, Diagonal, EmptyMapping, Intdex, Intdexed, LeftComposition, Multintdex,
    Physical, Product, RightComposition, Stride, Sum, Sup, Tensor,
};

fn square(x: f32) -> f32 {
    x * x
}

fn main() {
    // Mappings
    let stride = Rc::new(Stride::new(BoxDomain::new(vec![3, 2]), vec![1, 3]));
    let repeat = Rc::new(Stride::new(BoxDomain::new(vec![10, 13]), vec![0, 1]));
    let coo = Rc::new(COO::new(
        BoxDomain::new(vec![6, 5]),
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
        BoxDomain::new(vec![13]),
        vec![vec![0], vec![4], vec![5], vec![1], vec![10], vec![12]],
    ));
    let diagonalize_2d = Rc::new(Diagonal::new(BoxDomain::new(vec![6, 6])));
    let diagonalize_3d = Rc::new(Diagonal::new(BoxDomain::new(vec![6, 6, 6])));
    let empty_map = Rc::new(EmptyMapping::<Intdex, Intdex>::new(
        BoxDomain::new(vec![6]),
        BoxDomain::new(vec![0]),
    ));

    // Tensors
    let physical = Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let strided = Rc::new(RightComposition::<Intdex, _>::new(physical.clone(), stride));
    let sparse_coo_vector = Rc::new(Sum::<Multintdex, Intdex, _>::new(
        physical.clone(),
        coo_vector,
    ));
    let intdexed_sparse_coo_vector = Rc::new(Intdexed::new(
        sparse_coo_vector.clone(),
        BoxDomain::new(vec![13]),
    ));
    let repeated_sparse = Rc::new(RightComposition::<Intdex, _>::new(
        intdexed_sparse_coo_vector,
        repeat,
    ));
    let sparse_coo = Rc::new(Sum::<Multintdex, Intdex, _>::new(
        physical.clone(),
        coo.clone(),
    ));
    let prod_coo = Rc::new(Product::<Multintdex, Intdex, _>::new(
        physical.clone(),
        coo.clone(),
    ));
    let sup_coo = Rc::new(Sup::<Multintdex, Intdex, _>::new(physical.clone(), coo));
    let diag2d = Rc::new(Sum::<Multintdex, Intdex, _>::new(
        physical.clone(),
        diagonalize_2d.clone(),
    ));
    let diag3d = Rc::new(Sum::<Multintdex, Intdex, _>::new(physical, diagonalize_3d));
    let empty = Rc::new(Physical::new(vec![]));
    let ones = Rc::new(Product::<Intdex, Intdex, _>::new(empty, empty_map));
    let identity = Rc::new(Sum::<Multintdex, Intdex, _>::new(ones, diagonalize_2d));
    let strided_squared = Rc::new(LeftComposition::<Multintdex, _>::new(
        strided.clone(),
        square,
    ));
    let sparse_coo_squared = Rc::new(LeftComposition::<Multintdex, _>::new(
        sparse_coo.clone(),
        square,
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
