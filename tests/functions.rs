use std::rc::Rc;

use bten::{
    BoxDomain, BoxedTensor, COO, Domain, Physical, RightComposition, Stride, Sum, Tensor, clamp,
    diagonalize, eye, eye_rect, movedim, negate, ones, permute, pow, scale, square, transpose,
    zeros,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

fn strided() -> Rc<RightComposition<Stride>> {
    Rc::new(RightComposition::new(
        physical(),
        Rc::new(Stride::new(BoxDomain::new(vec![3, 2]), vec![1, 3])),
    ))
}

/// A rank-three tensor whose values are all distinct, so a permutation of its
/// axes is checkable: `[i, j, k]` holds `12i + 4j + k`.
fn cube() -> Rc<RightComposition<Stride>> {
    let values: Vec<f32> = (0..24).map(|value| value as f32).collect();
    Rc::new(RightComposition::new(
        Rc::new(Physical::new(values)),
        Rc::new(Stride::new(BoxDomain::new(vec![2, 3, 4]), vec![12, 4, 1])),
    ))
}

fn collect_2d(tensor: &dyn Tensor, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|i| (0..cols).map(|j| tensor.get(&[i, j])).collect())
        .collect()
}

#[test]
fn square_and_negate_transform_values() {
    let squared = square(physical());
    let negated = negate(physical());

    for i in 0..6 {
        let value = (i + 1) as f32;
        assert_eq!(squared.get(&[i]), value * value);
        assert_eq!(negated.get(&[i]), -value);
    }
}

#[test]
fn elementwise_functions_compose() {
    let squared_negative = square(Rc::new(negate(physical())));

    for i in 0..6 {
        let value = (i + 1) as f32;
        assert_eq!(squared_negative.get(&[i]), value * value);
    }
}

// These carry a value of their own, which a plain `fn` pointer could not hold.
#[test]
fn parameterised_transforms_remember_their_argument() {
    let doubled = scale(physical(), 2.0);
    let cubed = pow(physical(), 3.0);
    let bounded = clamp(physical(), 2.0, 4.0);

    for i in 0..6 {
        let value = (i + 1) as f32;
        assert_eq!(doubled.get(&[i]), value * 2.0);
        assert_eq!(cubed.get(&[i]), value * value * value);
    }

    // [1, 2, 3, 4, 5, 6] held between 2 and 4.
    let expected = [2.0, 2.0, 3.0, 4.0, 4.0, 4.0];
    for (i, expected) in expected.iter().enumerate() {
        assert_eq!(bounded.get(&[i]), *expected);
    }
}

#[test]
fn parameterised_transforms_compose() {
    let scaled_twice = scale(Rc::new(scale(physical(), 2.0)), 3.0);

    for i in 0..6 {
        assert_eq!(scaled_twice.get(&[i]), (i + 1) as f32 * 6.0);
    }
}

#[test]
fn several_transforms_of_one_tensor_keep_their_own_arguments() {
    let shared = physical();
    let doubled = scale(shared.clone(), 2.0);
    let tripled = scale(shared, 3.0);

    assert_eq!(doubled.get(&[0]), 2.0);
    assert_eq!(tripled.get(&[0]), 3.0);
}

#[test]
#[should_panic(expected = "must not exceed")]
fn clamp_rejects_a_reversed_range() {
    clamp(physical(), 4.0, 2.0);
}

#[test]
fn an_elementwise_transform_keeps_the_domain() {
    let squared = square(strided());

    assert_eq!(squared.box_domain().shape(), &[3, 2]);
    assert_eq!(
        collect_2d(&squared, 3, 2),
        vec![vec![1.0, 16.0], vec![4.0, 25.0], vec![9.0, 36.0]]
    );
}

#[test]
fn permute_reorders_axes() {
    let cube = cube();
    let permuted = permute(cube.clone(), vec![2, 0, 1]);

    // Axis 2 becomes axis 0, axis 0 becomes axis 1, axis 1 becomes axis 2.
    assert_eq!(permuted.box_domain().shape(), &[4, 2, 3]);

    for a in 0..4 {
        for b in 0..2 {
            for c in 0..3 {
                assert_eq!(permuted.get(&[a, b, c]), cube.get(&[b, c, a]));
            }
        }
    }
}

#[test]
fn transpose_swaps_the_last_two_axes() {
    let transposed = transpose(strided());

    assert_eq!(transposed.box_domain().shape(), &[2, 3]);
    assert_eq!(
        collect_2d(&transposed, 2, 3),
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
    );
}

#[test]
fn transpose_leaves_leading_axes_alone() {
    let cube = cube();
    let transposed = transpose(cube.clone());

    assert_eq!(transposed.box_domain().shape(), &[2, 4, 3]);

    for i in 0..2 {
        for j in 0..4 {
            for k in 0..3 {
                assert_eq!(transposed.get(&[i, j, k]), cube.get(&[i, k, j]));
            }
        }
    }
}

#[test]
fn transpose_is_an_involution() {
    let original = strided();
    let restored = transpose(Rc::new(transpose(original.clone())));

    assert_eq!(restored.box_domain().shape(), &[3, 2]);
    assert_eq!(collect_2d(&restored, 3, 2), collect_2d(&*original, 3, 2));
}

// The mapping permutes indices rather than strides, so it applies to tensors
// that have no strides to permute in the first place.
#[test]
fn transpose_applies_to_a_tensor_with_no_strides() {
    let sparse = Rc::new(Sum::new(
        Rc::new(Physical::new(vec![1.0, 2.0])),
        Rc::new(COO::new(
            BoxDomain::new(vec![2, 3]),
            vec![vec![0, 1], vec![1, 2]],
        )),
    ));
    let transposed = transpose(sparse.clone());

    assert_eq!(transposed.box_domain().shape(), &[3, 2]);
    assert_eq!(transposed.get(&[1, 0]), 1.0);
    assert_eq!(transposed.get(&[2, 1]), 2.0);
    assert_eq!(transposed.get(&[0, 0]), 0.0);

    for a in 0..3 {
        for b in 0..2 {
            assert_eq!(transposed.get(&[a, b]), sparse.get(&[b, a]));
        }
    }
}

#[test]
fn movedim_follows_the_pytorch_convention() {
    let cube = cube();
    let moved = movedim(cube.clone(), 0, 2);

    // Moving axis 0 to the end leaves the others in order: (2,3,4) -> (3,4,2).
    assert_eq!(moved.box_domain().shape(), &[3, 4, 2]);

    for a in 0..3 {
        for b in 0..4 {
            for c in 0..2 {
                assert_eq!(moved.get(&[a, b, c]), cube.get(&[c, a, b]));
            }
        }
    }
}

#[test]
fn movedim_is_reversible() {
    let cube = cube();
    let restored = movedim(Rc::new(movedim(cube.clone(), 0, 2)), 2, 0);

    assert_eq!(restored.box_domain().shape(), &[2, 3, 4]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(restored.get(&[i, j, k]), cube.get(&[i, j, k]));
            }
        }
    }
}

#[test]
fn diagonalize_builds_a_diagonal_matrix() {
    let diagonal = diagonalize(physical(), 2);

    assert_eq!(diagonal.box_domain().shape(), &[6, 6]);
    for i in 0..6 {
        for j in 0..6 {
            let expected = if i == j { (i + 1) as f32 } else { 0.0 };
            assert_eq!(diagonal.get(&[i, j]), expected);
        }
    }
}

#[test]
fn diagonalize_generalises_to_higher_rank() {
    let diagonal = diagonalize(physical(), 3);

    assert_eq!(diagonal.box_domain().shape(), &[6, 6, 6]);
    assert_eq!(diagonal.get(&[2, 2, 2]), 3.0);
    assert_eq!(diagonal.get(&[2, 2, 1]), 0.0);
    assert_eq!(diagonal.get(&[0, 1, 2]), 0.0);
    assert_eq!(diagonal.get(&[5, 5, 5]), 6.0);
}

#[test]
fn zeros_and_ones_are_constant_at_any_rank() {
    assert_eq!(zeros(vec![4]).get(&[3]), 0.0);
    assert_eq!(ones(vec![4]).get(&[3]), 1.0);

    let flat_zeros = zeros(vec![2, 3]);
    let flat_ones = ones(vec![2, 3]);
    assert_eq!(flat_zeros.box_domain().shape(), &[2, 3]);
    assert_eq!(collect_2d(&flat_zeros, 2, 3), vec![vec![0.0; 3]; 2]);
    assert_eq!(collect_2d(&flat_ones, 2, 3), vec![vec![1.0; 3]; 2]);

    assert_eq!(ones(vec![2, 3, 4]).get(&[1, 2, 3]), 1.0);
    assert_eq!(zeros(vec![2, 3, 4]).get(&[1, 2, 3]), 0.0);
}

// A box of no dimensions holds the single empty index, so a constant over it is
// a scalar.
#[test]
fn a_rank_zero_constant_is_a_scalar() {
    let one = ones(vec![]);

    assert_eq!(one.box_domain().ndim(), 0);
    assert_eq!(one.get(&[]), 1.0);
    assert_eq!(one.to_string(), "tensor(1)");
    assert_eq!(zeros(vec![]).to_string(), "tensor(0)");
}

#[test]
fn eye_is_the_identity() {
    let identity = eye(6);

    assert_eq!(identity.box_domain().shape(), &[6, 6]);
    for i in 0..6 {
        for j in 0..6 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(identity.get(&[i, j]), expected);
        }
    }
}

#[test]
fn eye_rect_puts_ones_on_the_diagonal_of_a_rectangle() {
    let wide = eye_rect(2, 4);
    let tall = eye_rect(4, 2);

    assert_eq!(wide.box_domain().shape(), &[2, 4]);
    assert_eq!(
        collect_2d(&wide, 2, 4),
        vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]
    );

    assert_eq!(tall.box_domain().shape(), &[4, 2]);
    assert_eq!(
        collect_2d(&tall, 4, 2),
        vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ]
    );
}

#[test]
fn eye_survives_a_transpose() {
    let identity = Rc::new(eye(4));
    let transposed = transpose(identity.clone());

    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(transposed.get(&[i, j]), identity.get(&[i, j]));
        }
    }
}

#[test]
#[should_panic(expected = "is repeated")]
fn permute_rejects_repeated_axes() {
    permute(cube(), vec![0, 0, 1]);
}

#[test]
#[should_panic(expected = "an axis per dimension")]
fn permute_rejects_the_wrong_number_of_axes() {
    permute(cube(), vec![1, 0]);
}

#[test]
#[should_panic(expected = "at least two dimensions")]
fn transpose_rejects_a_vector() {
    transpose(physical());
}

#[test]
#[should_panic(expected = "out of range")]
fn movedim_rejects_an_axis_out_of_range() {
    movedim(cube(), 0, 3);
}

#[test]
#[should_panic(expected = "takes a vector")]
fn diagonalize_rejects_a_matrix() {
    diagonalize(strided(), 2);
}
