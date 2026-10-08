use std::rc::Rc;

use bten::{Domain, HRect, JaggedDomain, MultiDomain};

#[test]
fn a_hrect_holds_exactly_the_indices_within_its_shape() {
    let hrect = HRect::new(vec![3, 2]);

    assert_eq!(hrect.shape(), &[3, 2]);
    assert_eq!(hrect.ndim(), Some(2));
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

    assert_eq!(hrect.ndim(), Some(1));
    assert!(hrect.contains(&[3]));
    assert!(!hrect.contains(&[4]));
}

#[test]
fn a_zero_dimensional_hrect_holds_only_the_empty_index() {
    let hrect = HRect::new(vec![]);

    assert_eq!(hrect.ndim(), Some(0));
    assert!(hrect.contains(&[]));
    assert!(!hrect.contains(&[0]));
}

#[test]
fn a_jagged_domain_holds_each_part_under_its_own_first_coordinate() {
    let jagged = JaggedDomain::new(vec![HRect::new(vec![2, 3]), HRect::new(vec![4, 1])]);

    assert_eq!(jagged.ndim(), Some(3));
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
fn a_multi_domain_can_hold_parts_of_different_ranks() {
    let mixed = JaggedDomain::new(vec![HRect::new(vec![2, 3]), HRect::new(vec![4])]);

    // Its indices are three or two coordinates long, so it has no one rank.
    assert_eq!(mixed.ndim(), None);
    assert!(mixed.contains(&[0, 1, 2]));
    assert!(mixed.contains(&[1, 3]));
    assert!(!mixed.contains(&[0, 1]));
    assert!(!mixed.contains(&[1, 3, 0]));
}

#[test]
fn a_multi_domain_can_hold_domains_of_any_kind() {
    let jagged = JaggedDomain::new(vec![HRect::new(vec![2]), HRect::new(vec![3])]);
    let nested = MultiDomain::new(vec![jagged.clone(), jagged]);
    let shared: MultiDomain<Rc<dyn Domain>> =
        MultiDomain::new(vec![Rc::new(HRect::new(vec![2])), Rc::new(nested.clone())]);

    assert_eq!(nested.ndim(), Some(3));
    assert!(nested.contains(&[1, 1, 2]));
    assert!(!nested.contains(&[1, 0, 2]));
    assert_eq!(shared.ndim(), None);
    assert!(shared.contains(&[0, 1]));
    assert!(shared.contains(&[1, 0, 1, 2]));
    assert!(!shared.contains(&[1, 1]));
}

#[test]
fn a_shared_copy_of_a_domain_holds_the_same_indices() {
    let domain = JaggedDomain::new(vec![HRect::new(vec![2, 3]), HRect::new(vec![4])]);
    let copy = domain.to_rc();

    assert_eq!(copy.ndim(), domain.ndim());
    assert!(copy.contains(&[0, 1, 2]));
    assert!(copy.contains(&[1, 3]));
    assert!(!copy.contains(&[1, 4]));
}

#[test]
#[should_panic(expected = "at least one part")]
fn a_jagged_domain_needs_a_part() {
    JaggedDomain::new(vec![]);
}
