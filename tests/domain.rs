use std::rc::Rc;

use bten::{
    BoxDomain, BoxedTensor, COO, Diagonal, Domain, EmptyMapping, IndexMapping, Intdex, Intdexed,
    LeftComposition, Multintdex, Physical, Powerset, PowersetMapping, RightComposition, Stride,
    Sum, Tensor, format_boxed,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

/// A domain that is not a box: the indices on and below the diagonal of a
/// square matrix.
struct LowerTriangle {
    extent: usize,
}

impl Domain<Multintdex> for LowerTriangle {
    fn contains(&self, index: &Multintdex) -> bool {
        index.ndim() == 2 && index.at(0) < self.extent && index.at(1) <= index.at(0)
    }
}

/// A mapping laying a lower triangle out row by row. Its domain is not a box,
/// so it is not a `BoxedMapping`, and tensors composed over it are not boxed
/// either.
struct TriangularLayout {
    domain: LowerTriangle,
    codomain: BoxDomain<Intdex>,
}

impl TriangularLayout {
    fn new(extent: usize) -> Self {
        Self {
            domain: LowerTriangle { extent },
            codomain: BoxDomain::new(vec![extent * (extent + 1) / 2]),
        }
    }
}

impl IndexMapping<Multintdex, Intdex> for TriangularLayout {
    fn map(&self, index: Multintdex) -> Intdex {
        let row = index.at(0);
        Intdex::new(row * (row + 1) / 2 + index.at(1))
    }

    fn domain(&self) -> &dyn Domain<Multintdex> {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain<Intdex> {
        &self.codomain
    }
}

impl PowersetMapping<Multintdex, Intdex> for TriangularLayout {
    fn map(&self, index: Multintdex) -> Powerset<Intdex> {
        Powerset::from_iter([IndexMapping::map(self, index)])
    }

    fn domain(&self) -> &dyn Domain<Multintdex> {
        &self.domain
    }

    fn codomain(&self) -> &dyn Domain<Intdex> {
        &self.codomain
    }
}

#[test]
fn a_box_holds_exactly_the_indices_within_its_shape() {
    let box_domain = BoxDomain::<Multintdex>::new(vec![3, 2]);

    assert_eq!(box_domain.shape(), &[3, 2]);
    assert_eq!(box_domain.ndim(), 2);
    assert_eq!(box_domain.size(), 6);
    assert!(box_domain.contains(&Multintdex::from([2, 1])));
    assert!(!box_domain.contains(&Multintdex::from([3, 1])));
    assert!(!box_domain.contains(&Multintdex::from([2, 2])));
    // Rank has to match too.
    assert!(!box_domain.contains(&Multintdex::from([2])));
    assert!(!box_domain.contains(&Multintdex::from([2, 1, 0])));
}

#[test]
fn a_box_over_intdex_is_a_flat_range() {
    let box_domain = BoxDomain::<Intdex>::new(vec![4]);

    assert!(box_domain.contains(&Intdex::new(3)));
    assert!(!box_domain.contains(&Intdex::new(4)));
    assert_eq!(box_domain.size(), 4);
}

#[test]
fn a_stride_goes_from_its_box_to_the_flat_positions_it_reaches() {
    let stride = Stride::new(BoxDomain::new(vec![3, 2]), vec![1, 3]);

    assert!(stride.domain().contains(&Multintdex::from([2, 1])));
    assert!(!stride.domain().contains(&Multintdex::from([3, 0])));
    // The largest index is 2 * 1 + 1 * 3 = 5, so six positions are reached.
    assert!(stride.codomain().contains(&Intdex::new(5)));
    assert!(!stride.codomain().contains(&Intdex::new(6)));
    // A broadcasting stride reaches fewer positions than its domain has indices.
    let repeat = Stride::new(BoxDomain::new(vec![4, 6]), vec![0, 1]);
    assert!(repeat.codomain().contains(&Intdex::new(5)));
    assert!(!repeat.codomain().contains(&Intdex::new(6)));
}

#[test]
fn a_coo_goes_from_its_box_to_the_positions_it_stores() {
    let coo = COO::new(
        BoxDomain::new(vec![6, 5]),
        vec![vec![0, 0], vec![4, 2], vec![5, 4]],
    );

    assert!(coo.domain().contains(&Multintdex::from([5, 4])));
    assert!(!coo.domain().contains(&Multintdex::from([6, 0])));
    assert!(coo.codomain().contains(&Intdex::new(2)));
    assert!(!coo.codomain().contains(&Intdex::new(3)));
}

#[test]
fn a_diagonal_reaches_as_far_as_its_shortest_dimension() {
    let diagonal = Diagonal::new(BoxDomain::new(vec![6, 4, 6]));

    assert!(diagonal.domain().contains(&Multintdex::from([3, 3, 3])));
    assert!(!diagonal.domain().contains(&Multintdex::from([3, 5, 3])));
    assert!(diagonal.codomain().contains(&Intdex::new(3)));
    assert!(!diagonal.codomain().contains(&Intdex::new(4)));
}

#[test]
fn an_empty_mapping_keeps_the_domain_and_codomain_it_was_given() {
    let empty =
        EmptyMapping::<Intdex, Intdex>::new(BoxDomain::new(vec![6]), BoxDomain::new(vec![0]));

    assert!(empty.domain().contains(&Intdex::new(5)));
    assert!(!empty.domain().contains(&Intdex::new(6)));
    assert!(!empty.codomain().contains(&Intdex::new(0)));
    assert!(PowersetMapping::map(&empty, Intdex::new(5)).is_empty());
}

#[test]
fn a_physical_tensor_is_defined_on_the_box_of_its_storage() {
    let physical = physical();

    assert!(physical.domain().contains(&Intdex::new(5)));
    assert!(!physical.domain().contains(&Intdex::new(6)));
    assert_eq!(physical.box_domain().shape(), &[6]);
}

#[test]
fn a_right_composition_takes_its_domain_from_its_mapping() {
    let strided = Rc::new(RightComposition::<Intdex, _>::new(
        physical(),
        Rc::new(Stride::new(BoxDomain::new(vec![3, 2]), vec![1, 3])),
    ));
    let transposed = RightComposition::<Intdex, _>::new(
        physical(),
        Rc::new(Stride::new(BoxDomain::new(vec![2, 3]), vec![3, 1])),
    );

    assert_eq!(strided.box_domain().shape(), &[3, 2]);
    assert!(strided.domain().contains(&Multintdex::from([2, 1])));
    assert!(!strided.domain().contains(&Multintdex::from([2, 2])));
    // The same storage under a mapping of another shape is another domain.
    assert_eq!(transposed.box_domain().shape(), &[2, 3]);
    assert!(transposed.domain().contains(&Multintdex::from([1, 2])));
}

#[test]
fn a_reduction_takes_its_domain_from_its_mapping() {
    let sparse = Sum::<Multintdex, Intdex, _>::new(
        physical(),
        Rc::new(COO::new(BoxDomain::new(vec![2, 3]), vec![vec![0, 0]])),
    );
    let diagonal = Sum::<Multintdex, Intdex, _>::new(
        physical(),
        Rc::new(Diagonal::new(BoxDomain::new(vec![4, 4, 4]))),
    );

    assert_eq!(sparse.box_domain().shape(), &[2, 3]);
    assert!(!sparse.domain().contains(&Multintdex::from([2, 0])));
    assert_eq!(diagonal.box_domain().shape(), &[4, 4, 4]);
}

#[test]
fn a_left_composition_takes_its_domain_from_the_tensor_it_transforms() {
    let strided = Rc::new(RightComposition::<Intdex, _>::new(
        physical(),
        Rc::new(Stride::new(BoxDomain::new(vec![3, 2]), vec![1, 3])),
    ));
    let squared = LeftComposition::<Multintdex, _>::new(strided.clone(), |x| x * x);

    assert_eq!(squared.box_domain().shape(), strided.box_domain().shape());
    assert!(squared.domain().contains(&Multintdex::from([2, 1])));
    assert!(!squared.domain().contains(&Multintdex::from([3, 0])));
    assert_eq!(squared.get(Multintdex::from([2, 1])), 36.0);
}

#[test]
fn an_intdexed_tensor_is_given_its_domain() {
    let narrowed = Intdexed::new(
        Rc::new(RightComposition::<Intdex, _>::new(
            physical(),
            Rc::new(Stride::new(BoxDomain::new(vec![6]), vec![1])),
        )),
        BoxDomain::new(vec![6]),
    );

    // The wrapped tensor's domain is over Multintdex, so it cannot serve as the
    // domain of an Intdex tensor: this one has to be passed in.
    assert_eq!(narrowed.box_domain().shape(), &[6]);
    assert!(narrowed.domain().contains(&Intdex::new(5)));
    assert!(!narrowed.domain().contains(&Intdex::new(6)));
}

#[test]
fn boxed_tensors_print_through_their_box() {
    let strided = RightComposition::<Intdex, _>::new(
        physical(),
        Rc::new(Stride::new(BoxDomain::new(vec![3, 2]), vec![1, 3])),
    );

    assert_eq!(format_boxed::<Multintdex, _>(&strided), strided.to_string());
    assert_eq!(
        format_boxed::<Intdex, _>(&*physical()),
        "tensor([1, 2, 3, 4, 5, 6])"
    );
}

// A tensor composed over a mapping whose domain is not a box reads fine, but
// neither `Display` nor `BoxedTensor` is implemented for it:
// `unboxed.to_string()` and `unboxed.box_domain()` do not compile.
#[test]
fn a_tensor_over_a_domain_that_is_not_a_box_still_reads() {
    let layout = Rc::new(TriangularLayout::new(3));

    // The layout is both an IndexMapping and a PowersetMapping, so its
    // codomain has to be asked for through one of them.
    assert!(IndexMapping::codomain(&*layout).contains(&Intdex::new(5)));
    assert!(!IndexMapping::codomain(&*layout).contains(&Intdex::new(6)));

    let unboxed = RightComposition::<Intdex, TriangularLayout>::new(physical(), layout);

    assert!(unboxed.domain().contains(&Multintdex::from([2, 1])));
    assert!(!unboxed.domain().contains(&Multintdex::from([1, 2])));
    // Row 2 starts at position 3, so [2, 1] reads the fifth stored value.
    assert_eq!(unboxed.get(Multintdex::from([2, 1])), 5.0);
}

// Same for a reduction over a mapping whose domain is not a box, and for a
// transform of the tensor that comes out: neither is printable.
#[test]
fn a_reduction_over_a_domain_that_is_not_a_box_still_reads() {
    let unboxed = Rc::new(Sum::<Multintdex, Intdex, TriangularLayout>::new(
        physical(),
        Rc::new(TriangularLayout::new(3)),
    ));
    let squared = LeftComposition::<Multintdex, _>::new(unboxed.clone(), |x| x * x);

    assert!(unboxed.domain().contains(&Multintdex::from([2, 1])));
    assert!(!unboxed.domain().contains(&Multintdex::from([1, 2])));
    assert_eq!(unboxed.get(Multintdex::from([2, 1])), 5.0);
    assert!(!squared.domain().contains(&Multintdex::from([1, 2])));
    assert_eq!(squared.get(Multintdex::from([2, 1])), 25.0);
}
