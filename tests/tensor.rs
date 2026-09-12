use std::rc::Rc;

use bten::{
    BoxedTensor, COO, Diagonal, Domain, HRect, IndexMapping, LeftComposition, Physical, Powerset,
    PowersetMapping, RightComposition, Stride, Sum, Tensor,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

/// A domain that is not a hrect: the indices on and below the diagonal of a
/// square matrix.
struct LowerTriangle {
    extent: usize,
}

impl Domain for LowerTriangle {
    fn ndim(&self) -> usize {
        2
    }

    fn contains(&self, index: &[usize]) -> bool {
        index.len() == 2 && index[0] < self.extent && index[1] <= index[0]
    }
}

/// A mapping laying a lower triangle out row by row. Its domain is not a hrect,
/// so it is not a `BoxedMapping`, and tensors composed over it are not boxed
/// either.
struct TriangularLayout {
    domain: LowerTriangle,
    codomain: HRect,
}

impl TriangularLayout {
    fn new(extent: usize) -> Self {
        Self {
            domain: LowerTriangle { extent },
            codomain: HRect::new(vec![extent * (extent + 1) / 2]),
        }
    }
}

impl IndexMapping for TriangularLayout {
    fn map(&self, index: &[usize]) -> Vec<usize> {
        vec![index[0] * (index[0] + 1) / 2 + index[1]]
    }

    fn domain(&self) -> &dyn Domain {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain {
        &self.codomain
    }
}

impl PowersetMapping for TriangularLayout {
    fn map(&self, index: &[usize]) -> Powerset {
        Powerset::from_iter([IndexMapping::map(self, index)])
    }

    fn domain(&self) -> &dyn Domain {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain {
        &self.codomain
    }
}

#[test]
fn a_physical_tensor_is_defined_on_the_hrect_of_its_storage() {
    let physical = physical();

    assert_eq!(physical.domain().ndim(), 1);
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

    assert_eq!(
        squared.hrect().shape(),
        strided.hrect().shape()
    );
    assert!(squared.domain().contains(&[2, 1]));
    assert!(!squared.domain().contains(&[3, 0]));
    assert_eq!(squared.get(&[2, 1]), 36.0);
}

// A tensor composed over a mapping whose domain is not a hrect reads fine, but
// neither `Display` nor `BoxedTensor` is implemented for it:
// `unboxed.to_string()` and `unboxed.hrect()` do not compile.
#[test]
fn a_tensor_over_a_domain_that_is_not_a_hrect_still_reads() {
    let layout = Rc::new(TriangularLayout::new(3));

    // The layout is both an IndexMapping and a PowersetMapping, so its
    // codomain has to be asked for through one of them.
    assert!(IndexMapping::codomain(&*layout).contains(&[5]));
    assert!(!IndexMapping::codomain(&*layout).contains(&[6]));

    let unboxed = RightComposition::new(physical(), layout);

    assert!(unboxed.domain().contains(&[2, 1]));
    assert!(!unboxed.domain().contains(&[1, 2]));
    // Row 2 starts at position 3, so [2, 1] reads the fifth stored value.
    assert_eq!(unboxed.get(&[2, 1]), 5.0);
}

// Same for a reduction over a mapping whose domain is not a hrect, and for a
// transform of the tensor that comes out: neither is printable.
#[test]
fn a_reduction_over_a_domain_that_is_not_a_hrect_still_reads() {
    let unboxed = Rc::new(Sum::new(physical(), Rc::new(TriangularLayout::new(3))));
    let squared = LeftComposition::new(unboxed.clone(), |x| x * x);

    assert!(unboxed.domain().contains(&[2, 1]));
    assert!(!unboxed.domain().contains(&[1, 2]));
    assert_eq!(unboxed.get(&[2, 1]), 5.0);
    assert!(!squared.domain().contains(&[1, 2]));
    assert_eq!(squared.get(&[2, 1]), 25.0);
}
