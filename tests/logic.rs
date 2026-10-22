use std::rc::Rc;

use cten::{
    COO, Diagonal, EmptyMapping, HRect, Inf, LeftComposition, Physical, PowersetMapping, Product,
    RightComposition, Stride, Sum, Sup, Tensor,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

fn collect_2d(tensor: &dyn Tensor, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|i| (0..cols).map(|j| tensor.get(&[i, j])).collect())
        .collect()
}

fn coo_2d() -> Rc<COO> {
    Rc::new(COO::new(
        HRect::new(vec![6, 5]),
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
        assert_eq!(physical.get(&[i]), (i + 1) as f32);
    }
}

#[test]
#[should_panic]
fn physical_rejects_out_of_bounds_indices() {
    physical().get(&[6]);
}

#[test]
#[should_panic(expected = "single coordinate")]
fn physical_rejects_indices_of_the_wrong_rank() {
    physical().get(&[0, 0]);
}

#[test]
fn right_composition_reads_through_a_stride() {
    let strided = RightComposition::new(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    );

    assert_eq!(
        collect_2d(&strided, 3, 2),
        vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
    );
}

#[test]
fn a_zero_stride_broadcasts_along_its_axis() {
    let repeated = RightComposition::new(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![4, 6]), vec![0, 1])),
    );

    let rows = collect_2d(&repeated, 4, 6);
    for row in &rows {
        assert_eq!(row, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

// A mapping into flat positions needs a one-dimensional tensor under it. That
// used to be a matter of index types; it is now checked when composing.
#[test]
#[should_panic(expected = "indexed by 2 coordinates")]
fn composing_over_a_tensor_of_the_wrong_rank_is_rejected() {
    let strided = Rc::new(RightComposition::new(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    ));

    RightComposition::new(
        strided,
        Rc::new(Stride::new(HRect::new(vec![2, 3]), vec![1, 1])),
    );
}

#[test]
fn left_composition_transforms_values_pointwise() {
    let strided = Rc::new(RightComposition::new(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    ));
    let squared = LeftComposition::new(strided, |x| x * x);

    assert_eq!(
        collect_2d(&squared, 3, 2),
        vec![vec![1.0, 16.0], vec![4.0, 25.0], vec![9.0, 36.0]]
    );
}

// A one-dimensional tensor is read by a mapping just like a physical one is:
// there is no longer a wrapper between the two.
#[test]
fn a_sparse_vector_can_be_read_through_a_mapping() {
    let coo_vector = Rc::new(COO::new(
        HRect::new(vec![6]),
        vec![vec![0], vec![4], vec![5]],
    ));
    let sparse = Rc::new(Sum::new(physical(), coo_vector));

    assert_eq!(sparse.get(&[0]), 1.0);
    assert_eq!(sparse.get(&[4]), 2.0);
    assert_eq!(sparse.get(&[5]), 3.0);
    assert_eq!(sparse.get(&[1]), 0.0);

    let repeated = RightComposition::new(
        sparse.clone(),
        Rc::new(Stride::new(HRect::new(vec![2, 6]), vec![0, 1])),
    );

    for row in collect_2d(&repeated, 2, 6) {
        assert_eq!(row, vec![1.0, 0.0, 0.0, 0.0, 2.0, 3.0]);
    }
}

#[test]
fn sum_over_a_coo_pattern_scatters_the_stored_values() {
    let sparse = Sum::new(physical(), coo_2d());

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
    let off_support = [3, 3];

    let sum = Sum::new(physical(), coo_2d());
    let product = Product::new(physical(), coo_2d());
    let sup = Sup::new(physical(), coo_2d());
    let inf = Inf::new(physical(), coo_2d());

    assert_eq!(sum.get(&off_support), 0.0);
    assert_eq!(product.get(&off_support), 1.0);
    assert_eq!(sup.get(&off_support), f32::NEG_INFINITY);
    assert_eq!(inf.get(&off_support), f32::INFINITY);
}

// The demo's COO patterns all use distinct coordinates, so every Powerset it
// builds holds at most one index and the fold in ReductionTensor never runs.
// These patterns deliberately store several positions at one coordinate.
#[test]
fn reductions_accumulate_over_every_matching_position() {
    let overlapping = || {
        Rc::new(COO::new(
            HRect::new(vec![2, 2]),
            vec![vec![1, 1], vec![1, 1], vec![1, 1]],
        ))
    };
    let values = || Rc::new(Physical::new(vec![2.0, 3.0, 4.0]));
    let at = [1, 1];

    let sum = Sum::new(values(), overlapping());
    let product = Product::new(values(), overlapping());
    let sup = Sup::new(values(), overlapping());
    let inf = Inf::new(values(), overlapping());

    assert_eq!(sum.get(&at), 9.0);
    assert_eq!(product.get(&at), 24.0);
    assert_eq!(sup.get(&at), 4.0);
    assert_eq!(inf.get(&at), 2.0);
}

#[test]
fn a_partial_coo_entry_matches_every_index_sharing_its_prefix() {
    let row_pattern = Rc::new(COO::new(HRect::new(vec![4, 3]), vec![vec![2]]));
    let sum = Sum::new(physical(), row_pattern);

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
    let diag = Sum::new(physical(), Rc::new(Diagonal::new(HRect::new(vec![6, 6]))));

    for i in 0..6 {
        for j in 0..6 {
            let expected = if i == j { (i + 1) as f32 } else { 0.0 };
            assert_eq!(diag.get(&[i, j]), expected);
        }
    }
}

#[test]
fn diagonal_reduction_generalises_to_rank_three() {
    let diag = Sum::new(
        physical(),
        Rc::new(Diagonal::new(HRect::new(vec![6, 6, 6]))),
    );

    assert_eq!(diag.get(&[2, 2, 2]), 3.0);
    assert_eq!(diag.get(&[2, 2, 1]), 0.0);
    assert_eq!(diag.get(&[0, 1, 2]), 0.0);
    assert_eq!(diag.get(&[5, 5, 5]), 6.0);
}

// A product over the empty set is 1 everywhere, so a constant tensor of ones
// comes out of an empty store; putting those on the diagonal gives identity.
#[test]
fn empty_mapping_under_a_product_yields_ones_from_an_empty_store() {
    let empty = Rc::new(Physical::new(vec![]));
    let ones = Rc::new(Product::new(
        empty,
        Rc::new(EmptyMapping::new(HRect::new(vec![6]), HRect::new(vec![0]))),
    ));

    for i in 0..6 {
        assert_eq!(ones.get(&[i]), 1.0);
    }

    let identity = Sum::new(ones, Rc::new(Diagonal::new(HRect::new(vec![6, 6]))));
    for i in 0..6 {
        for j in 0..6 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(identity.get(&[i, j]), expected);
        }
    }
}

#[test]
fn compositions_nest_to_arbitrary_depth() {
    let sparse = Rc::new(Sum::new(
        physical(),
        Rc::new(COO::new(
            HRect::new(vec![13]),
            vec![vec![0], vec![4], vec![5], vec![1], vec![10], vec![12]],
        )),
    ));
    let repeated = Rc::new(RightComposition::new(
        sparse,
        Rc::new(Stride::new(HRect::new(vec![10, 13]), vec![0, 1])),
    ));
    let squared = LeftComposition::new(repeated, |x| x * x);

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
    let diagonal = Diagonal::new(HRect::new(vec![6, 6]));
    let selected = diagonal.map(&[4, 4]);

    assert_eq!(selected.len(), 1);
    assert!(selected.contains([4].as_slice()));
    assert!(diagonal.map(&[4, 5]).is_empty());
}
