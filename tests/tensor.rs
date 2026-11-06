use std::rc::Rc;

use cten::{
    COO, Composition, Diagonal, Function, HRect, IndexSet, InvProjection, JaggedTensor, Multiset,
    Multisets, Permutation, Physical, Stride, Sum, ValueFunction, Values, compose, multiset_lift,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

#[test]
fn a_physical_tensor_is_defined_on_the_hrect_of_its_storage() {
    let physical = physical();

    assert_eq!(physical.domain().ndim(), 1);
    assert!(physical.domain().contains(&[5]));
    assert!(!physical.domain().contains(&[6]));
    assert_eq!(physical.domain().shape(), &[6]);
}

#[test]
fn a_composition_takes_its_domain_from_the_function_applied_first() {
    let strided = Rc::new(compose(
        physical(),
        Stride::new(HRect::new(vec![3, 2]), vec![1, 3]),
    ));
    let transposed = compose(physical(), Stride::new(HRect::new(vec![2, 3]), vec![3, 1]));

    assert_eq!(strided.domain().shape(), &[3, 2]);
    assert!(strided.domain().contains(&[2, 1]));
    assert!(!strided.domain().contains(&[2, 2]));
    // The same storage under an index function of another shape is another domain.
    assert_eq!(transposed.domain().shape(), &[2, 3]);
    assert!(transposed.domain().contains(&[1, 2]));
}

#[test]
fn a_reduction_tensor_takes_its_domain_from_its_index_function() {
    let sparse = compose(
        Sum::new(),
        compose(
            multiset_lift(physical()),
            COO::new(HRect::new(vec![2, 3]), vec![vec![0, 0]]),
        ),
    );
    let diagonal = compose(
        Sum::new(),
        compose(
            multiset_lift(physical()),
            Diagonal::new(HRect::new(vec![4, 4, 4])),
        ),
    );

    assert_eq!(sparse.domain().shape(), &[2, 3]);
    assert!(!sparse.domain().contains(&[2, 0]));
    assert_eq!(diagonal.domain().shape(), &[4, 4, 4]);
}

#[test]
fn a_function_on_values_keeps_the_domain_of_the_tensor_it_follows() {
    let strided = Rc::new(compose(
        physical(),
        Stride::new(HRect::new(vec![3, 2]), vec![1, 3]),
    ));
    let squared = compose(ValueFunction::new(|x: f32| x * x), strided.clone());

    assert_eq!(squared.domain().shape(), strided.domain().shape());
    assert!(squared.domain().contains(&[2, 1]));
    assert!(!squared.domain().contains(&[3, 0]));
    assert_eq!(squared.apply(&[2, 1]), 36.0);
}

// Composition is not only for tensors: permuting the axes of indices one way and
// then back is the identity on indices.
#[test]
fn index_functions_compose_like_any_other_functions() {
    let forth = Permutation::new(HRect::new(vec![4, 2, 3]), vec![1, 2, 0]);
    let back = Permutation::new(HRect::new(vec![2, 3, 4]), vec![2, 0, 1]);
    let round_trip = compose(back, forth);

    assert_eq!(round_trip.domain().shape(), &[2, 3, 4]);
    for index in [[0, 0, 0], [1, 2, 3], [0, 1, 2]] {
        assert_eq!(round_trip.apply(&index), index);
    }
}

// The multiset extension of a tensor reads a multiset of values at a multiset of
// indices, repeats included.
#[test]
fn a_lifted_tensor_reads_a_multiset_of_values_at_a_multiset_of_indices() {
    let lifted = multiset_lift(physical());
    let read = lifted.apply(&Multiset::new(vec![vec![0], vec![2], vec![2]]));

    assert_eq!(read.into_iter().collect::<Vec<f32>>(), vec![1.0, 3.0, 3.0]);
    assert!(lifted.apply(&Multiset::default()).is_empty());
}

#[test]
fn a_lifted_function_on_values_applies_to_each_element() {
    let squared = multiset_lift(ValueFunction::new(|x: f32| x * x));
    let read = squared.apply(&Multiset::new(vec![1.0, 2.0, 2.0]));

    assert_eq!(read.into_iter().collect::<Vec<f32>>(), vec![1.0, 4.0, 4.0]);
}

/// The matrix of the paper's row sums example, `[[1, 2, 3], [0, 4, 1], [2, 2, 5]]`.
fn figure_matrix() -> Rc<Composition<Physical, Stride>> {
    Rc::new(compose(
        Physical::new(vec![1.0, 2.0, 3.0, 0.0, 4.0, 1.0, 2.0, 2.0, 5.0]),
        Stride::new(HRect::new(vec![3, 3]), vec![3, 1]),
    ))
}

// Before reduction, each row index reads the multiset of its row; reducing those
// multisets with a sum gives the row sums.
#[test]
fn a_lifted_tensor_after_an_index_function_is_a_reduction_before_the_reduction() {
    let rows = Rc::new(InvProjection::new(HRect::new(vec![3, 3]), 1));
    let gathered = Rc::new(compose(multiset_lift(figure_matrix()), rows));
    let row_sums = compose(Sum::new(), gathered.clone());

    assert_eq!(gathered.domain().shape(), &[3]);
    let before: Vec<Vec<f32>> = (0..3)
        .map(|i| gathered.apply(&[i]).into_iter().collect())
        .collect();
    assert_eq!(
        before,
        vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 4.0, 1.0],
            vec![2.0, 2.0, 5.0]
        ]
    );
    let after: Vec<f32> = (0..3).map(|i| row_sums.apply(&[i])).collect();
    assert_eq!(after, vec![6.0, 5.0, 9.0]);
}

#[test]
fn a_physical_tensor_holds_values_of_any_type() {
    let ints = Physical::new(vec![3i64, -1]);
    let flags = Physical::new(vec![true, false]);

    assert_eq!(ints.apply(&[1]), -1);
    assert!(flags.apply(&[0]));
    assert!(!flags.apply(&[1]));
}

// A function on values may change their type, as a comparison turns numbers
// into booleans.
#[test]
fn a_function_on_values_can_change_the_value_type() {
    let above_three = compose(ValueFunction::new(|x: f32| x > 3.0), physical());

    assert_eq!(above_three.domain().shape(), &[6]);
    assert!(!above_three.apply(&[2]));
    assert!(above_three.apply(&[3]));
}

#[test]
fn a_jagged_tensor_reads_each_part_under_its_first_coordinate() {
    let jagged = JaggedTensor::new(vec![physical(), Rc::new(Physical::new(vec![10.0, 20.0]))]);

    assert_eq!(jagged.apply(&[0, 4]), 5.0);
    assert_eq!(jagged.apply(&[1, 1]), 20.0);
    // Its domain is the two storages side by side, each keeping its own length.
    let domain = jagged.domain();
    assert_eq!(domain.ndim(), 2);
    assert!(domain.contains(&[0, 5]));
    assert!(domain.contains(&[1, 1]));
    assert!(!domain.contains(&[1, 2]));
    assert_eq!(domain.parts(), &[HRect::new(vec![6]), HRect::new(vec![2])]);
}

// Every function has a domain and a codomain, functions on values and on
// multisets included: a lifted tensor goes from the multisets of its indices to
// multisets of values, and an index function to multisets of indices.
#[test]
fn every_function_has_a_domain_and_a_codomain() {
    let lifted = multiset_lift(physical());
    let lifted_domain: Multisets<HRect> = lifted.domain();
    let _: Multisets<Values<f32>> = lifted.codomain();
    assert_eq!(lifted_domain.set(), &HRect::new(vec![6]));

    let diagonal = Diagonal::new(HRect::new(vec![6, 6]));
    let diagonal_codomain: Multisets<HRect> = diagonal.codomain();
    assert_eq!(diagonal_codomain.set(), &HRect::new(vec![6]));

    let _: Multisets<Values<f32>> = Sum::new().domain();
    let _: Values<f32> = Sum::new().codomain();
    let _: Values<f32> = ValueFunction::new(|x: f32| x * x).domain();
}

// The codomain of the inner function must lie in the domain of the outer one:
// here the index function reaches seven stored values, and there are six.
#[test]
#[should_panic(expected = "does not lie in")]
fn composing_over_too_few_stored_values_is_rejected() {
    let coo = COO::new(HRect::new(vec![7]), (0..7).map(|i| vec![i]).collect());
    compose(multiset_lift(physical()), coo);
}
