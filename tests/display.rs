use std::rc::Rc;

use cten::{
    COO, Diagonal, Function, HRect, Physical, Stride, Sum, Sup, compose, format_boxed,
    multiset_lift,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

#[test]
fn a_physical_vector_prints_flat() {
    assert_eq!(physical().to_string(), "tensor([1, 2, 3, 4, 5, 6])");
}

#[test]
fn tensors_of_other_value_types_print_the_same_way() {
    let ints = Physical::new(vec![1i64, -20, 3]);
    let flags = Physical::new(vec![true, false]);

    assert_eq!(ints.to_string(), "tensor([  1, -20,   3])");
    assert_eq!(flags.to_string(), "tensor([ true, false])");
}

// The diagonal matrix of the paper's sparse matrices figure, before and after
// its multisets of values are summed.
#[test]
fn a_multiset_valued_tensor_prints_its_multisets() {
    let stored = Physical::new(vec![1.0, 2.0, 4.0]);
    let diagonal = Diagonal::new(HRect::new(vec![3, 3]));
    let before = Rc::new(compose(multiset_lift(stored), diagonal));
    let after = compose(Sum::new(), before.clone());

    assert_eq!(
        before.to_string(),
        "tensor([[{{1}},  {{}},  {{}}],\n        [ {{}}, {{2}},  {{}}],\n        [ {{}},  {{}}, {{4}}]])"
    );
    assert_eq!(
        after.to_string(),
        "tensor([[1, 0, 0],\n        [0, 2, 0],\n        [0, 0, 4]])"
    );
}

#[test]
fn a_rank_two_tensor_prints_one_row_per_line_aligned_under_the_bracket() {
    let strided = compose(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    );

    assert_eq!(
        strided.to_string(),
        "tensor([[1, 4],\n        [2, 5],\n        [3, 6]])"
    );
}

#[test]
fn values_are_right_aligned_to_a_common_width() {
    let wide = Rc::new(Physical::new(vec![1.0, 16.0, 4.0, 25.0]));
    let strided = compose(
        wide,
        Rc::new(Stride::new(HRect::new(vec![2, 2]), vec![1, 2])),
    );

    assert_eq!(strided.to_string(), "tensor([[ 1,  4],\n        [16, 25]])");
}

#[test]
fn the_neutral_element_of_a_sup_prints_as_negative_infinity() {
    let coo = Rc::new(COO::new(
        HRect::new(vec![2, 2]),
        vec![vec![0, 0], vec![1, 1]],
    ));
    let sup = compose(Sup::new(), compose(multiset_lift(physical()), coo));

    assert_eq!(
        sup.to_string(),
        "tensor([[   1, -inf],\n        [-inf,    2]])"
    );
}

#[test]
fn rank_three_tensors_separate_blocks_with_a_blank_line() {
    let diag = compose(
        Sum::new(),
        compose(
            multiset_lift(physical()),
            Rc::new(Diagonal::new(HRect::new(vec![2, 2, 2]))),
        ),
    );

    assert_eq!(
        diag.to_string(),
        concat!(
            "tensor([[[1, 0],\n",
            "         [0, 0]],\n",
            "\n",
            "        [[0, 0],\n",
            "         [0, 2]]])"
        )
    );
}

#[test]
fn display_agrees_with_direct_lookups() {
    let sparse = compose(
        Sum::new(),
        compose(
            multiset_lift(physical()),
            Rc::new(COO::new(
                HRect::new(vec![2, 3]),
                vec![vec![0, 0], vec![1, 2]],
            )),
        ),
    );

    let rendered = sparse.to_string();
    for i in 0..2 {
        for j in 0..3 {
            let value = sparse.apply(&[i, j]);
            assert!(
                rendered.contains(&value.to_string()),
                "{rendered} is missing the value at ({i}, {j})"
            );
        }
    }
}

#[test]
fn boxed_tensors_print_through_their_hrect() {
    let strided = compose(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    );

    assert_eq!(format_boxed(&strided), strided.to_string());
    assert_eq!(format_boxed(&*physical()), "tensor([1, 2, 3, 4, 5, 6])");
}
