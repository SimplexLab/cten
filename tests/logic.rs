use std::rc::Rc;

use bten::{
    BoxDomain, COO, Diagonal, EmptyMapping, Inf, Intdex, Intdexed, LeftComposition, Multintdex,
    Physical, Powerset, PowersetMapping, Product, RightComposition, Stride, Sum, Sup, Tensor,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

fn collect_2d(tensor: &dyn Tensor<Multintdex>, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| tensor.get(Multintdex::from([i, j])))
                .collect()
        })
        .collect()
}

fn coo_2d() -> Rc<COO> {
    Rc::new(COO::new(
        BoxDomain::new(vec![6, 5]),
        vec![
            vec![0, 0],
            vec![0, 3],
            vec![4, 2],
            vec![5, 4],
            vec![1, 3],
            vec![2, 0],
        ],
    ))
}

#[test]
fn physical_reads_its_backing_storage() {
    let physical = physical();
    for i in 0..6 {
        assert_eq!(physical.get(Intdex::new(i)), (i + 1) as f32);
    }
}

#[test]
#[should_panic]
fn physical_rejects_out_of_bounds_indices() {
    physical().get(Intdex::new(6));
}

#[test]
fn right_composition_reads_through_a_stride() {
    let strided = RightComposition::<Intdex, _>::new(
        physical(),
        Rc::new(Stride::new(BoxDomain::new(vec![3, 2]), vec![1, 3])),
    );

    assert_eq!(
        collect_2d(&strided, 3, 2),
        vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
    );
}

#[test]
fn a_zero_stride_broadcasts_along_its_axis() {
    let repeated = RightComposition::<Intdex, _>::new(
        physical(),
        Rc::new(Stride::new(BoxDomain::new(vec![4, 6]), vec![0, 1])),
    );

    let rows = collect_2d(&repeated, 4, 6);
    for row in &rows {
        assert_eq!(row, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

#[test]
fn left_composition_transforms_values_pointwise() {
    let strided: Rc<RightComposition<Intdex, Stride>> = Rc::new(RightComposition::new(
        physical(),
        Rc::new(Stride::new(BoxDomain::new(vec![3, 2]), vec![1, 3])),
    ));
    let squared = LeftComposition::<Multintdex, _>::new(strided, |x| x * x);

    assert_eq!(
        collect_2d(&squared, 3, 2),
        vec![vec![1.0, 16.0], vec![4.0, 25.0], vec![9.0, 36.0]]
    );
}

#[test]
fn intdexed_narrows_a_one_dimensional_multintdex_tensor() {
    let coo_vector = Rc::new(COO::new(
        BoxDomain::new(vec![6]),
        vec![vec![0], vec![4], vec![5]],
    ));
    let sparse = Rc::new(Sum::<Multintdex, Intdex, _>::new(physical(), coo_vector));
    let narrowed = Intdexed::new(sparse.clone(), BoxDomain::new(vec![6]));

    for i in 0..6 {
        assert_eq!(
            narrowed.get(Intdex::new(i)),
            sparse.get(Multintdex::from([i]))
        );
    }
    assert_eq!(narrowed.get(Intdex::new(0)), 1.0);
    assert_eq!(narrowed.get(Intdex::new(4)), 2.0);
    assert_eq!(narrowed.get(Intdex::new(5)), 3.0);
    assert_eq!(narrowed.get(Intdex::new(1)), 0.0);
}

#[test]
fn sum_over_a_coo_pattern_scatters_the_stored_values() {
    let sparse = Sum::<Multintdex, Intdex, _>::new(physical(), coo_2d());

    assert_eq!(
        collect_2d(&sparse, 6, 5),
        vec![
            vec![1.0, 0.0, 0.0, 2.0, 0.0],
            vec![0.0, 0.0, 0.0, 5.0, 0.0],
            vec![6.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 3.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 4.0],
        ]
    );
}

#[test]
fn each_reduction_falls_back_to_its_neutral_element() {
    let off_support = Multintdex::from([3, 3]);

    let sum = Sum::<Multintdex, Intdex, _>::new(physical(), coo_2d());
    let product = Product::<Multintdex, Intdex, _>::new(physical(), coo_2d());
    let sup = Sup::<Multintdex, Intdex, _>::new(physical(), coo_2d());
    let inf = Inf::<Multintdex, Intdex, _>::new(physical(), coo_2d());

    assert_eq!(sum.get(off_support.clone()), 0.0);
    assert_eq!(product.get(off_support.clone()), 1.0);
    assert_eq!(sup.get(off_support.clone()), f32::NEG_INFINITY);
    assert_eq!(inf.get(off_support), f32::INFINITY);
}

// The demo's COO patterns all use distinct coordinates, so every Powerset it
// builds holds at most one index and the fold in ReductionTensor never runs.
// These patterns deliberately store several positions at one coordinate.
#[test]
fn reductions_accumulate_over_every_matching_position() {
    let overlapping = || {
        Rc::new(COO::new(
            BoxDomain::new(vec![2, 2]),
            vec![vec![1, 1], vec![1, 1], vec![1, 1]],
        ))
    };
    let values = || Rc::new(Physical::new(vec![2.0, 3.0, 4.0]));
    let at = Multintdex::from([1, 1]);

    let sum = Sum::<Multintdex, Intdex, _>::new(values(), overlapping());
    let product = Product::<Multintdex, Intdex, _>::new(values(), overlapping());
    let sup = Sup::<Multintdex, Intdex, _>::new(values(), overlapping());
    let inf = Inf::<Multintdex, Intdex, _>::new(values(), overlapping());

    assert_eq!(sum.get(at.clone()), 9.0);
    assert_eq!(product.get(at.clone()), 24.0);
    assert_eq!(sup.get(at.clone()), 4.0);
    assert_eq!(inf.get(at), 2.0);
}

#[test]
fn a_partial_coo_entry_matches_every_index_sharing_its_prefix() {
    let row_pattern = Rc::new(COO::new(BoxDomain::new(vec![4, 3]), vec![vec![2]]));
    let sum = Sum::<Multintdex, Intdex, _>::new(physical(), row_pattern);

    assert_eq!(
        collect_2d(&sum, 4, 3),
        vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ]
    );
}

#[test]
fn diagonal_reduction_builds_a_diagonal_matrix() {
    let diag = Sum::<Multintdex, Intdex, _>::new(
        physical(),
        Rc::new(Diagonal::new(BoxDomain::new(vec![6, 6]))),
    );

    for i in 0..6 {
        for j in 0..6 {
            let expected = if i == j { (i + 1) as f32 } else { 0.0 };
            assert_eq!(diag.get(Multintdex::from([i, j])), expected);
        }
    }
}

#[test]
fn diagonal_reduction_generalises_to_rank_three() {
    let diag = Sum::<Multintdex, Intdex, _>::new(
        physical(),
        Rc::new(Diagonal::new(BoxDomain::new(vec![6, 6, 6]))),
    );

    assert_eq!(diag.get(Multintdex::from([2, 2, 2])), 3.0);
    assert_eq!(diag.get(Multintdex::from([2, 2, 1])), 0.0);
    assert_eq!(diag.get(Multintdex::from([0, 1, 2])), 0.0);
    assert_eq!(diag.get(Multintdex::from([5, 5, 5])), 6.0);
}

// A product over the empty set is 1 everywhere, so a constant tensor of ones
// comes out of an empty store; putting those on the diagonal gives identity.
#[test]
fn empty_mapping_under_a_product_yields_ones_from_an_empty_store() {
    let empty = Rc::new(Physical::new(vec![]));
    let ones = Rc::new(Product::<Intdex, Intdex, _>::new(
        empty,
        Rc::new(EmptyMapping::<Intdex, Intdex>::new(
            BoxDomain::new(vec![6]),
            BoxDomain::new(vec![0]),
        )),
    ));

    for i in 0..6 {
        assert_eq!(ones.get(Intdex::new(i)), 1.0);
    }

    let identity =
        Sum::<Multintdex, Intdex, _>::new(ones, Rc::new(Diagonal::new(BoxDomain::new(vec![6, 6]))));
    for i in 0..6 {
        for j in 0..6 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(identity.get(Multintdex::from([i, j])), expected);
        }
    }
}

#[test]
fn compositions_nest_to_arbitrary_depth() {
    let sparse = Rc::new(Sum::<Multintdex, Intdex, _>::new(
        physical(),
        Rc::new(COO::new(
            BoxDomain::new(vec![13]),
            vec![vec![0], vec![4], vec![5], vec![1], vec![10], vec![12]],
        )),
    ));
    let narrowed = Rc::new(Intdexed::new(sparse, BoxDomain::new(vec![13])));
    let repeated = Rc::new(RightComposition::<Intdex, _>::new(
        narrowed,
        Rc::new(Stride::new(BoxDomain::new(vec![10, 13]), vec![0, 1])),
    ));
    let squared = LeftComposition::<Multintdex, _>::new(repeated, |x| x * x);

    let expected: Vec<f32> = [
        1.0, 4.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 6.0,
    ]
    .iter()
    .map(|x: &f32| x * x)
    .collect();

    for row in collect_2d(&squared, 10, 13) {
        assert_eq!(row, expected);
    }
}

#[test]
fn a_powerset_mapping_can_be_used_directly() {
    let diagonal = Diagonal::new(BoxDomain::new(vec![6, 6]));
    let selected: Powerset<Intdex> = diagonal.map(Multintdex::from([4, 4]));

    assert_eq!(selected.size(), 1);
    assert!(selected.contains(&Intdex::new(4)));
    assert!(diagonal.map(Multintdex::from([4, 5])).is_empty());
}
