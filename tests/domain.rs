use bten::{Domain, HRect, JaggedDomain};

#[test]
fn a_hrect_holds_exactly_the_indices_within_its_shape() {
    let hrect = HRect::new(vec![3, 2]);

    assert_eq!(hrect.shape(), &[3, 2]);
    assert_eq!(hrect.ndim(), 2);
    assert!(hrect.contains(&[2, 1]));
    assert!(!hrect.contains(&[3, 1]));
    assert!(!hrect.contains(&[2, 2]));
    // Rank has to match too.
    assert!(!hrect.contains(&[2]));
    assert!(!hrect.contains(&[2, 1, 0]));
}

#[test]
fn a_one_dimensional_hrect_is_a_flat_range() {
    let hrect = HRect::new(vec![4]);

    assert_eq!(hrect.ndim(), 1);
    assert!(hrect.contains(&[3]));
    assert!(!hrect.contains(&[4]));
}

#[test]
fn a_zero_dimensional_hrect_holds_only_the_empty_index() {
    let hrect = HRect::new(vec![]);

    assert_eq!(hrect.ndim(), 0);
    assert!(hrect.contains(&[]));
    assert!(!hrect.contains(&[0]));
}

#[test]
fn a_jagged_domain_holds_each_part_under_its_own_first_coordinate() {
    let jagged = JaggedDomain::new(vec![HRect::new(vec![2, 3]), HRect::new(vec![4, 1])]);

    assert_eq!(jagged.ndim(), 3);
    assert!(jagged.contains(&[0, 1, 2]));
    assert!(!jagged.contains(&[0, 2, 0]));
    // Each part has its own shape, so what fits one part need not fit another.
    assert!(jagged.contains(&[1, 3, 0]));
    assert!(!jagged.contains(&[1, 0, 1]));
    assert!(!jagged.contains(&[2, 0, 0]));
    assert!(!jagged.contains(&[0, 1]));
    assert!(!jagged.contains(&[]));
}

#[test]
#[should_panic(expected = "same number of coordinates")]
fn a_jagged_domain_needs_parts_of_one_rank() {
    JaggedDomain::new(vec![HRect::new(vec![2, 3]), HRect::new(vec![4])]);
}

#[test]
#[should_panic(expected = "at least one part")]
fn a_jagged_domain_needs_a_part() {
    JaggedDomain::new(vec![]);
}
