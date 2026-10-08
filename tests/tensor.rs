use std::rc::Rc;

use bten::{
    BoxedTensor, COO, Diagonal, HRect, LeftComposition, MultiTensor, Physical, RightComposition,
    Stride, Sum, Tensor,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

#[test]
fn a_physical_tensor_is_defined_on_the_hrect_of_its_storage() {
    let physical = physical();

    assert_eq!(physical.domain().ndim(), Some(1));
    assert!(physical.domain().contains(&[5]));
    assert!(!physical.domain().contains(&[6]));
    assert_eq!(physical.hrect().shape(), &[6]);
}

#[test]
fn a_right_composition_takes_its_domain_from_its_mapping() {
    let strided = Rc::new(RightComposition::new(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    ));
    let transposed = RightComposition::new(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![2, 3]), vec![3, 1])),
    );

    assert_eq!(strided.hrect().shape(), &[3, 2]);
    assert!(strided.domain().contains(&[2, 1]));
    assert!(!strided.domain().contains(&[2, 2]));
    // The same storage under a mapping of another shape is another domain.
    assert_eq!(transposed.hrect().shape(), &[2, 3]);
    assert!(transposed.domain().contains(&[1, 2]));
}

#[test]
fn a_reduction_takes_its_domain_from_its_mapping() {
    let sparse = Sum::new(
        physical(),
        Rc::new(COO::new(HRect::new(vec![2, 3]), vec![vec![0, 0]])),
    );
    let diagonal = Sum::new(
        physical(),
        Rc::new(Diagonal::new(HRect::new(vec![4, 4, 4]))),
    );

    assert_eq!(sparse.hrect().shape(), &[2, 3]);
    assert!(!sparse.domain().contains(&[2, 0]));
    assert_eq!(diagonal.hrect().shape(), &[4, 4, 4]);
}

#[test]
fn a_left_composition_takes_its_domain_from_the_tensor_it_transforms() {
    let strided = Rc::new(RightComposition::new(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    ));
    let squared = LeftComposition::new(strided.clone(), |x| x * x);

    assert_eq!(squared.hrect().shape(), strided.hrect().shape());
    assert!(squared.domain().contains(&[2, 1]));
    assert!(!squared.domain().contains(&[3, 0]));
    assert_eq!(squared.get(&[2, 1]), 36.0);
}

#[test]
fn a_multi_tensor_reads_each_part_under_its_first_coordinate() {
    let multi = MultiTensor::new(vec![physical(), Rc::new(Physical::new(vec![10.0, 20.0]))]);

    assert_eq!(multi.get(&[0, 4]), 5.0);
    assert_eq!(multi.get(&[1, 1]), 20.0);
    // Its domain is the two storages side by side, each keeping its own length.
    assert_eq!(multi.domain().ndim(), Some(2));
    assert!(multi.domain().contains(&[0, 5]));
    assert!(multi.domain().contains(&[1, 1]));
    assert!(!multi.domain().contains(&[1, 2]));
}

#[test]
fn a_multi_tensor_can_hold_parts_of_different_ranks() {
    let matrix = Rc::new(RightComposition::new(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    ));
    let multi = MultiTensor::new(vec![matrix, Rc::new(Physical::new(vec![10.0, 20.0]))]);

    assert_eq!(multi.get(&[0, 2, 1]), 6.0);
    assert_eq!(multi.get(&[1, 1]), 20.0);
    // Its indices are three or two coordinates long, so it has no one rank.
    assert_eq!(multi.domain().ndim(), None);
    assert!(multi.domain().contains(&[0, 2, 1]));
    assert!(multi.domain().contains(&[1, 1]));
    assert!(!multi.domain().contains(&[0, 2]));
}

// A part need not be printable: here one multi-tensor holds another.
#[test]
fn a_multi_tensor_can_hold_another_multi_tensor() {
    let inner = MultiTensor::new(vec![physical(), Rc::new(Physical::new(vec![10.0, 20.0]))]);
    let outer = MultiTensor::new(vec![Rc::new(inner), physical()]);

    assert_eq!(outer.get(&[0, 1, 0]), 10.0);
    assert_eq!(outer.get(&[1, 3]), 4.0);
    assert!(outer.domain().contains(&[0, 0, 5]));
    assert!(!outer.domain().contains(&[0, 1, 2]));
}
