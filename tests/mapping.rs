use std::{assert_eq, rc::Rc};

use cten::{
    COO, Concat, Diagonal, EmptyMapping, HRect, IndexMapping, InvProjection, JaggedDomain,
    Permutation, Physical, Powerset, PowersetMapping, Projection, Squeeze, Stride, Unsqueeze,
    Where,
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
fn a_projection_removes_dim_from_index() {
    let proj = Projection::new(HRect::new(vec![2, 3, 4]), 1);

    assert_eq!(proj.codomain().ndim(), 2);
    assert!(proj.codomain().contains(&[1, 3]));
    assert!(!proj.codomain().contains(&[1, 4]));
    assert!(proj.domain().contains(&[1, 2, 3]));
    assert_eq!(proj.map(&[1, 2, 3]), vec![1, 3])
}

#[test]
fn an_inv_projection_sends_each_index_to_the_line_along_its_dim() {
    let inv_proj = InvProjection::new(HRect::new(vec![2, 3, 4]), 1);

    assert_eq!(inv_proj.domain().ndim(), 2);
    assert!(inv_proj.domain().contains(&[1, 3]));
    assert!(!inv_proj.domain().contains(&[1, 4]));
    assert!(inv_proj.codomain().contains(&[1, 2, 3]));
    assert_eq!(
        inv_proj.map(&[1, 3]),
        Powerset::from([vec![1, 0, 3], vec![1, 1, 3], vec![1, 2, 3]])
    );
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

#[test]
fn an_unsqueeze_adds_an_axis_of_extent_one_to_its_codomain() {
    let unsqueeze = Unsqueeze::new(HRect::new(vec![3, 2]), 1);

    assert!(unsqueeze.domain().contains(&[2, 0, 1]));
    assert!(!unsqueeze.domain().contains(&[2, 1, 1]));
    assert!(unsqueeze.codomain().contains(&[2, 1]));
    assert_eq!(unsqueeze.map(&[2, 0, 1]), vec![2, 1]);
}

#[test]
fn a_squeeze_drops_axes_of_extent_one_from_its_codomain() {
    let squeeze = Squeeze::new(HRect::new(vec![1, 3, 1]), vec![2, 0]);

    assert_eq!(squeeze.domain().ndim(), 1);
    assert!(squeeze.domain().contains(&[2]));
    assert!(!squeeze.domain().contains(&[3]));
    assert!(squeeze.codomain().contains(&[0, 2, 0]));
    // The axes were given out of order; each coordinate still lands back in place.
    assert_eq!(squeeze.map(&[2]), vec![0, 2, 0]);
}

#[test]
#[should_panic(expected = "extent 1")]
fn a_squeeze_rejects_an_axis_that_is_not_of_extent_one() {
    Squeeze::new(HRect::new(vec![2, 3]), vec![0]);
}

#[test]
fn a_concat_goes_from_a_hrect_to_the_jagged_domain_of_its_parts() {
    let concat = Concat::new(
        JaggedDomain::new(vec![HRect::new(vec![2, 3]), HRect::new(vec![2, 1])]),
        1,
    );

    // Extents 3 and 1 along axis 1 add up to 4.
    assert!(concat.domain().contains(&[1, 3]));
    assert!(!concat.domain().contains(&[1, 4]));
    assert!(!concat.domain().contains(&[2, 0]));
    assert_eq!(concat.codomain().ndim(), 3);
    // The part comes first, then the index within it.
    assert_eq!(concat.map(&[1, 2]), vec![0, 1, 2]);
    assert_eq!(concat.map(&[1, 3]), vec![1, 1, 0]);
    assert!(concat.codomain().contains(&[1, 1, 0]));
}

#[test]
#[should_panic(expected = "other extents agree")]
fn a_concat_rejects_parts_whose_other_extents_differ() {
    Concat::new(
        JaggedDomain::new(vec![HRect::new(vec![2, 3]), HRect::new(vec![3, 3])]),
        1,
    );
}

#[test]
fn a_where_sends_each_index_to_the_part_its_condition_picks() {
    let picks = Where::new(
        JaggedDomain::new(vec![HRect::new(vec![3]), HRect::new(vec![3])]),
        Rc::new(Physical::new(vec![1.0, 0.0, 2.0])),
    );

    assert!(picks.domain().contains(&[2]));
    assert!(!picks.domain().contains(&[3]));
    assert_eq!(picks.map(&[0]), vec![0, 0]);
    assert_eq!(picks.map(&[1]), vec![1, 1]);
    assert_eq!(picks.map(&[2]), vec![0, 2]);
}
