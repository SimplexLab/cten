use bten::{
    COO, Diagonal, EmptyMapping, HRect, IndexMapping, Permutation, PowersetMapping, Stride,
};

#[test]
fn a_stride_goes_from_its_hrect_to_the_flat_positions_it_reaches() {
    let stride = Stride::new(HRect::new(vec![3, 2]), vec![1, 3]);

    assert!(stride.domain().contains(&[2, 1]));
    assert!(!stride.domain().contains(&[3, 0]));
    // The largest index is 2 * 1 + 1 * 3 = 5, so six positions are reached.
    assert_eq!(stride.codomain().ndim(), 1);
    assert!(stride.codomain().contains(&[5]));
    assert!(!stride.codomain().contains(&[6]));
    // A broadcasting stride reaches fewer positions than its domain has indices.
    let repeat = Stride::new(HRect::new(vec![4, 6]), vec![0, 1]);
    assert!(repeat.codomain().contains(&[5]));
    assert!(!repeat.codomain().contains(&[6]));
}

#[test]
fn a_coo_goes_from_its_hrect_to_the_positions_it_stores() {
    let coo = COO::new(
        HRect::new(vec![6, 5]),
        vec![vec![0, 0], vec![4, 2], vec![5, 4]],
    );

    assert!(coo.domain().contains(&[5, 4]));
    assert!(!coo.domain().contains(&[6, 0]));
    assert!(coo.codomain().contains(&[2]));
    assert!(!coo.codomain().contains(&[3]));
}

#[test]
fn a_diagonal_reaches_as_far_as_its_shortest_dimension() {
    let diagonal = Diagonal::new(HRect::new(vec![6, 4, 6]));

    assert_eq!(diagonal.domain().ndim(), 3);
    assert!(diagonal.domain().contains(&[3, 3, 3]));
    assert!(!diagonal.domain().contains(&[3, 5, 3]));
    assert!(diagonal.codomain().contains(&[3]));
    assert!(!diagonal.codomain().contains(&[4]));
}

#[test]
fn a_permutation_reorders_its_domain_and_keeps_its_codomain() {
    let permutation = Permutation::new(HRect::new(vec![2, 3, 4]), vec![2, 0, 1]);

    // Axis 2 of the codomain becomes axis 0 of the domain, and so on, so the
    // domain is the codomain's shape read through `axes`.
    assert_eq!(permutation.axes(), &[2, 0, 1]);
    assert_eq!(permutation.domain().ndim(), 3);
    assert!(permutation.domain().contains(&[3, 1, 2]));
    assert!(!permutation.domain().contains(&[4, 1, 2]));
    assert!(permutation.codomain().contains(&[1, 2, 3]));
    assert!(!permutation.codomain().contains(&[1, 3, 2]));
    assert_eq!(permutation.map(&[3, 1, 2]), vec![1, 2, 3]);
}

#[test]
fn an_empty_mapping_keeps_the_domain_and_codomain_it_was_given() {
    let empty = EmptyMapping::new(HRect::new(vec![6]), HRect::new(vec![0]));

    assert!(empty.domain().contains(&[5]));
    assert!(!empty.domain().contains(&[6]));
    assert!(!empty.codomain().contains(&[0]));
    assert!(empty.map(&[5]).is_empty());
}
