use std::rc::Rc;

use bten::{
    COO, Diagonal, Intdex, Multintdex, Physical, RightComposition, Stride, Sum, Sup, Tensor,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

#[test]
fn a_physical_vector_prints_flat() {
    assert_eq!(physical().to_string(), "tensor([1, 2, 3, 4, 5, 6])");
}

#[test]
fn a_rank_two_tensor_prints_one_row_per_line_aligned_under_the_bracket() {
    let strided =
        RightComposition::<Intdex>::new(physical(), Rc::new(Stride::new(vec![1, 3])), vec![3, 2]);

    assert_eq!(
        strided.to_string(),
        "tensor([[1, 4],\n        [2, 5],\n        [3, 6]])"
    );
}

#[test]
fn values_are_right_aligned_to_a_common_width() {
    let wide = Rc::new(Physical::new(vec![1.0, 16.0, 4.0, 25.0]));
    let strided =
        RightComposition::<Intdex>::new(wide, Rc::new(Stride::new(vec![1, 2])), vec![2, 2]);

    assert_eq!(strided.to_string(), "tensor([[ 1,  4],\n        [16, 25]])");
}

#[test]
fn the_neutral_element_of_a_sup_prints_as_negative_infinity() {
    let coo = Rc::new(COO::new(vec![vec![0, 0], vec![1, 1]]));
    let sup = Sup::<Multintdex, Intdex>::new(physical(), coo, vec![2, 2]);

    assert_eq!(
        sup.to_string(),
        "tensor([[   1, -inf],\n        [-inf,    2]])"
    );
}

#[test]
fn rank_three_tensors_separate_blocks_with_a_blank_line() {
    let diag = Sum::<Multintdex, Intdex>::new(physical(), Rc::new(Diagonal::new()), vec![2, 2, 2]);

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
    let sparse = Sum::<Multintdex, Intdex>::new(
        physical(),
        Rc::new(COO::new(vec![vec![0, 0], vec![1, 2]])),
        vec![2, 3],
    );

    let rendered = sparse.to_string();
    for i in 0..2 {
        for j in 0..3 {
            let value = sparse.get(Multintdex::from([i, j]));
            assert!(
                rendered.contains(&value.to_string()),
                "{rendered} is missing the value at ({i}, {j})"
            );
        }
    }
}
