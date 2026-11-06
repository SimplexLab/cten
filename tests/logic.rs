use std::rc::Rc;

use cten::{
    Affine, COO, Diagonal, EmptyMapping, Function, HRect, IndexFunction, Inf,
    MultiIndexFunctionSignature, Multiset, Multisets, Physical, Product, Stride, Sum, Sup, Tensor,
    ValueFunction, compose, multiset_lift,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

fn collect_2d(tensor: &dyn Tensor, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|i| (0..cols).map(|j| tensor.apply(&[i, j])).collect())
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
        assert_eq!(physical.apply(&[i]), (i + 1) as f32);
    }
}

#[test]
#[should_panic]
fn physical_rejects_out_of_bounds_indices() {
    physical().apply(&[6]);
}

#[test]
#[should_panic(expected = "single coordinate")]
fn physical_rejects_indices_of_the_wrong_rank() {
    physical().apply(&[0, 0]);
}

#[test]
fn right_composition_reads_through_a_stride() {
    let strided = compose(
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
    let repeated = compose(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![4, 6]), vec![0, 1])),
    );

    let rows = collect_2d(&repeated, 4, 6);
    for row in &rows {
        assert_eq!(row, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

// An index function into flat positions needs a one-dimensional tensor under it. That
// used to be a matter of index types; it is now checked when composing.
#[test]
#[should_panic(expected = "does not lie in")]
fn composing_over_a_tensor_of_the_wrong_rank_is_rejected() {
    let strided = Rc::new(compose(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    ));

    compose(
        strided,
        Rc::new(Stride::new(HRect::new(vec![2, 3]), vec![1, 1])),
    );
}

#[test]
fn left_composition_transforms_values_pointwise() {
    let strided = Rc::new(compose(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    ));
    let squared = compose(ValueFunction::new(|x| x * x), strided);

    assert_eq!(
        collect_2d(&squared, 3, 2),
        vec![vec![1.0, 16.0], vec![4.0, 25.0], vec![9.0, 36.0]]
    );
}

// A one-dimensional tensor is read by an index function just like a physical one is:
// there is no longer a wrapper between the two.
#[test]
fn a_sparse_vector_can_be_read_through_a_mapping() {
    let coo_vector = Rc::new(COO::new(
        HRect::new(vec![6]),
        vec![vec![0], vec![4], vec![5]],
    ));
    let sparse = Rc::new(compose(
        Sum::new(),
        compose(multiset_lift(physical()), coo_vector),
    ));

    assert_eq!(sparse.apply(&[0]), 1.0);
    assert_eq!(sparse.apply(&[4]), 2.0);
    assert_eq!(sparse.apply(&[5]), 3.0);
    assert_eq!(sparse.apply(&[1]), 0.0);

    let repeated = compose(
        sparse.clone(),
        Rc::new(Stride::new(HRect::new(vec![2, 6]), vec![0, 1])),
    );

    for row in collect_2d(&repeated, 2, 6) {
        assert_eq!(row, vec![1.0, 0.0, 0.0, 0.0, 2.0, 3.0]);
    }
}

#[test]
fn sum_over_a_coo_pattern_scatters_the_stored_values() {
    let sparse = compose(Sum::new(), compose(multiset_lift(physical()), coo_2d()));

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

    let sum = compose(Sum::new(), compose(multiset_lift(physical()), coo_2d()));
    let product = compose(Product::new(), compose(multiset_lift(physical()), coo_2d()));
    let sup = compose(Sup::new(), compose(multiset_lift(physical()), coo_2d()));
    let inf = compose(Inf::new(), compose(multiset_lift(physical()), coo_2d()));

    assert_eq!(sum.apply(&off_support), 0.0);
    assert_eq!(product.apply(&off_support), 1.0);
    assert_eq!(sup.apply(&off_support), f32::NEG_INFINITY);
    assert_eq!(inf.apply(&off_support), f32::INFINITY);
}

// The demo's COO patterns all use distinct coordinates, so every multiset it
// builds holds at most one index and a reduction has nothing to combine. These
// patterns deliberately store several positions at one coordinate.
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

    let sum = compose(Sum::new(), compose(multiset_lift(values()), overlapping()));
    let product = compose(
        Product::new(),
        compose(multiset_lift(values()), overlapping()),
    );
    let sup = compose(Sup::new(), compose(multiset_lift(values()), overlapping()));
    let inf = compose(Inf::new(), compose(multiset_lift(values()), overlapping()));

    assert_eq!(sum.apply(&at), 9.0);
    assert_eq!(product.apply(&at), 24.0);
    assert_eq!(sup.apply(&at), 4.0);
    assert_eq!(inf.apply(&at), 2.0);
}

#[test]
fn a_partial_coo_entry_matches_every_index_sharing_its_prefix() {
    let row_pattern = Rc::new(COO::new(HRect::new(vec![4, 3]), vec![vec![2]]));
    let sum = compose(Sum::new(), compose(multiset_lift(physical()), row_pattern));

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
    let diag = compose(
        Sum::new(),
        compose(
            multiset_lift(physical()),
            Rc::new(Diagonal::new(HRect::new(vec![6, 6]))),
        ),
    );

    for i in 0..6 {
        for j in 0..6 {
            let expected = if i == j { (i + 1) as f32 } else { 0.0 };
            assert_eq!(diag.apply(&[i, j]), expected);
        }
    }
}

#[test]
fn diagonal_reduction_generalises_to_rank_three() {
    let diag = compose(
        Sum::new(),
        compose(
            multiset_lift(physical()),
            Rc::new(Diagonal::new(HRect::new(vec![6, 6, 6]))),
        ),
    );

    assert_eq!(diag.apply(&[2, 2, 2]), 3.0);
    assert_eq!(diag.apply(&[2, 2, 1]), 0.0);
    assert_eq!(diag.apply(&[0, 1, 2]), 0.0);
    assert_eq!(diag.apply(&[5, 5, 5]), 6.0);
}

// A product over the empty set is 1 everywhere, so a constant tensor of ones
// comes out of an empty store; putting those on the diagonal gives identity.
#[test]
fn empty_mapping_under_a_product_yields_ones_from_an_empty_store() {
    let empty = Rc::new(Physical::<f32>::new(vec![]));
    let ones = Rc::new(compose(
        Product::new(),
        compose(
            multiset_lift(empty),
            Rc::new(EmptyMapping::new(HRect::new(vec![6]), HRect::new(vec![0]))),
        ),
    ));

    for i in 0..6 {
        assert_eq!(ones.apply(&[i]), 1.0);
    }

    let identity = compose(
        Sum::new(),
        compose(
            multiset_lift(ones),
            Rc::new(Diagonal::new(HRect::new(vec![6, 6]))),
        ),
    );
    for i in 0..6 {
        for j in 0..6 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(identity.apply(&[i, j]), expected);
        }
    }
}

#[test]
fn compositions_nest_to_arbitrary_depth() {
    let sparse = Rc::new(compose(
        Sum::new(),
        compose(
            multiset_lift(physical()),
            Rc::new(COO::new(
                HRect::new(vec![13]),
                vec![vec![0], vec![4], vec![5], vec![1], vec![10], vec![12]],
            )),
        ),
    ));
    let repeated = Rc::new(compose(
        sparse,
        Rc::new(Stride::new(HRect::new(vec![10, 13]), vec![0, 1])),
    ));
    let squared = compose(ValueFunction::new(|x| x * x), repeated);

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
fn an_index_function_can_be_used_directly() {
    let diagonal = Diagonal::new(HRect::new(vec![6, 6]));
    let selected = diagonal.map(&[4, 4]);

    assert_eq!(selected.len(), 1);
    assert!(selected.iter().any(|index| index == &[4]));
    assert!(diagonal.map(&[4, 5]).is_empty());
}

// Read through an affine map, a tensor gives its values on a lattice: here the
// diagonal of a matrix, as torch's `diagonal` does.
#[test]
fn an_affine_mapping_reads_a_lattice_out_of_a_tensor() {
    let values: Vec<f32> = (1..=9).map(|value| value as f32).collect();
    let matrix = Rc::new(compose(
        Rc::new(Physical::new(values)),
        Rc::new(Stride::new(HRect::new(vec![3, 3]), vec![3, 1])),
    ));
    let diagonal = compose(
        matrix,
        Rc::new(Affine::new(
            HRect::new(vec![3]),
            vec![vec![1], vec![1]],
            vec![0, 0],
            HRect::new(vec![3, 3]),
        )),
    );

    let read: Vec<f32> = (0..3).map(|i| diagonal.apply(&[i])).collect();
    assert_eq!(read, vec![1.0, 5.0, 9.0]);
}

/// An index function sending each index to the multiset holding it twice.
struct Twice {
    signature: MultiIndexFunctionSignature,
}

impl IndexFunction for Twice {
    type Codomain = Multisets<HRect>;

    fn map(&self, index: &[usize]) -> Multiset<Vec<usize>> {
        Multiset::new(vec![index.to_vec(), index.to_vec()])
    }

    fn signature(&self) -> &MultiIndexFunctionSignature {
        &self.signature
    }
}

// Index functions map to multisets, so an index can be met more than once, and
// a reduction then counts its value as many times.
#[test]
fn a_reduction_counts_an_index_as_often_as_it_is_repeated() {
    let twice = Rc::new(Twice {
        signature: MultiIndexFunctionSignature::new(
            HRect::new(vec![6]),
            Multisets::new(HRect::new(vec![6])),
        ),
    });
    let doubled = compose(
        Sum::new(),
        compose(multiset_lift(physical()), twice.clone()),
    );
    let squared = compose(Product::new(), compose(multiset_lift(physical()), twice));

    for i in 0..6 {
        let value = (i + 1) as f32;
        assert_eq!(doubled.apply(&[i]), 2.0 * value);
        assert_eq!(squared.apply(&[i]), value * value);
    }
}
