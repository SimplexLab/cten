use bten::{Domain, HRect};

#[test]
fn a_hrect_holds_exactly_the_indices_within_its_shape() {
    let hrect_domain = HRect::new(vec![3, 2]);

    assert_eq!(hrect_domain.shape(), &[3, 2]);
    assert_eq!(hrect_domain.ndim(), 2);
    assert!(hrect_domain.contains(&[2, 1]));
    assert!(!hrect_domain.contains(&[3, 1]));
    assert!(!hrect_domain.contains(&[2, 2]));
    // Rank has to match too.
    assert!(!hrect_domain.contains(&[2]));
    assert!(!hrect_domain.contains(&[2, 1, 0]));
}

#[test]
fn a_one_dimensional_hrect_is_a_flat_range() {
    let hrect_domain = HRect::new(vec![4]);

    assert_eq!(hrect_domain.ndim(), 1);
    assert!(hrect_domain.contains(&[3]));
    assert!(!hrect_domain.contains(&[4]));
}
