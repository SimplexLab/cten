use bten::{Domain, HRect};

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
