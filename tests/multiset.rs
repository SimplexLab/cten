use cten::{InfReduction, Multiset, ProductReduction, Reduction, SumReduction, SupReduction};

#[test]
fn a_multiset_prints_with_double_braces() {
    assert_eq!(Multiset::new(vec![1, 3, 3]).to_string(), "{{1, 3, 3}}");
    assert_eq!(Multiset::<i64>::default().to_string(), "{{}}");
}

#[test]
fn a_multiset_keeps_every_repeat() {
    let multiset: Multiset<i64> = [3, 1, 3].into_iter().collect();

    assert_eq!(multiset.len(), 3);
    assert!(!multiset.is_empty());
    assert!(Multiset::<i64>::default().is_empty());
}

// The example of issue #18: the sum counts the 3 twice.
#[test]
fn a_reduction_reduces_a_multiset_to_one_value() {
    assert_eq!(
        SumReduction::reduce(Multiset::new(vec![1.0, 3.0, 3.0])),
        7.0
    );
    assert_eq!(SumReduction::reduce(Multiset::new(vec![1, 3, 3])), 7);
    assert_eq!(ProductReduction::reduce(Multiset::new(vec![2, 3, 3])), 18);
    assert_eq!(SupReduction::reduce(Multiset::new(vec![2, 7, 5])), 7);
    assert_eq!(InfReduction::reduce(Multiset::new(vec![2, 7, 5])), 2);
}

// Reducing the empty multiset gives the neutral element, and reducing a single
// element gives that element.
#[test]
fn a_reduction_of_no_value_or_one_value_is_as_expected() {
    assert_eq!(SumReduction::reduce(Multiset::<f32>::default()), 0.0);
    assert_eq!(ProductReduction::reduce(Multiset::<f32>::default()), 1.0);
    assert_eq!(
        SupReduction::reduce(Multiset::<f32>::default()),
        f32::NEG_INFINITY
    );
    assert_eq!(InfReduction::reduce(Multiset::new(vec![4.0])), 4.0);
}
