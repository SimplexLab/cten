use std::rc::Rc;

use bten::{
    BoxedTensor, COO, Domain, HRect, Physical, RightComposition, Stride, Sum, Tensor, clamp,
    concatenate, diagonalize, eye, eye_rect, movedim, negate, ones, permute, pow, scale,
    sparse_coo_tensor, square, squeeze, squeeze_dim, stack, transpose, unsqueeze, where_, zeros,
};

fn physical() -> Rc<Physical> {
    Rc::new(Physical::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

fn strided() -> Rc<RightComposition<Stride>> {
    Rc::new(RightComposition::new(
        physical(),
        Rc::new(Stride::new(HRect::new(vec![3, 2]), vec![1, 3])),
    ))
}

/// A rank-three tensor whose values are all distinct, so a permutation of its
/// axes is checkable: `[i, j, k]` holds `12i + 4j + k`.
fn cube() -> Rc<RightComposition<Stride>> {
    let values: Vec<f32> = (0..24).map(|value| value as f32).collect();
    Rc::new(RightComposition::new(
        Rc::new(Physical::new(values)),
        Rc::new(Stride::new(HRect::new(vec![2, 3, 4]), vec![12, 4, 1])),
    ))
}

/// A row-major matrix over stored `values`.
fn matrix(rows: usize, cols: usize, values: Vec<f32>) -> Rc<RightComposition<Stride>> {
    Rc::new(RightComposition::new(
        Rc::new(Physical::new(values)),
        Rc::new(Stride::new(HRect::new(vec![rows, cols]), vec![cols, 1])),
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
#[should_panic(expected = "got 4 and 2")]
fn clamp_rejects_a_reversed_range() {
    clamp(physical(), 4.0, 2.0);
}

// Rejecting these up front is what keeps `f32::clamp` from panicking on a read.
#[test]
#[should_panic(expected = "clamp needs low <= high")]
fn clamp_rejects_a_nan_bound() {
    clamp(physical(), f32::NAN, 4.0);
}

#[test]
fn an_elementwise_transform_keeps_the_domain() {
    let squared = square(strided());

    assert_eq!(squared.hrect().shape(), &[3, 2]);
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
    assert_eq!(permuted.hrect().shape(), &[4, 2, 3]);

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

    assert_eq!(transposed.hrect().shape(), &[2, 3]);
    assert_eq!(
        collect_2d(&transposed, 2, 3),
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
    );
}

#[test]
fn transpose_leaves_leading_axes_alone() {
    let cube = cube();
    let transposed = transpose(cube.clone());

    assert_eq!(transposed.hrect().shape(), &[2, 4, 3]);

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

    assert_eq!(restored.hrect().shape(), &[3, 2]);
    assert_eq!(collect_2d(&restored, 3, 2), collect_2d(&*original, 3, 2));
}

// The mapping permutes indices rather than strides, so it applies to tensors
// that have no strides to permute in the first place.
#[test]
fn transpose_applies_to_a_tensor_with_no_strides() {
    let sparse = Rc::new(Sum::new(
        Rc::new(Physical::new(vec![1.0, 2.0])),
        Rc::new(COO::new(
            HRect::new(vec![2, 3]),
            vec![vec![0, 1], vec![1, 2]],
        )),
    ));
    let transposed = transpose(sparse.clone());

    assert_eq!(transposed.hrect().shape(), &[3, 2]);
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
    assert_eq!(moved.hrect().shape(), &[3, 4, 2]);

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

    assert_eq!(restored.hrect().shape(), &[2, 3, 4]);
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

    assert_eq!(diagonal.hrect().shape(), &[6, 6]);
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

    assert_eq!(diagonal.hrect().shape(), &[6, 6, 6]);
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
    assert_eq!(flat_zeros.hrect().shape(), &[2, 3]);
    assert_eq!(collect_2d(&flat_zeros, 2, 3), vec![vec![0.0; 3]; 2]);
    assert_eq!(collect_2d(&flat_ones, 2, 3), vec![vec![1.0; 3]; 2]);

    assert_eq!(ones(vec![2, 3, 4]).get(&[1, 2, 3]), 1.0);
    assert_eq!(zeros(vec![2, 3, 4]).get(&[1, 2, 3]), 0.0);
}

// A hrect of no dimensions holds the single empty index, so a constant over it is
// a scalar.
#[test]
fn a_rank_zero_constant_is_a_scalar() {
    let one = ones(vec![]);

    assert_eq!(one.hrect().ndim(), 0);
    assert_eq!(one.get(&[]), 1.0);
    assert_eq!(one.to_string(), "tensor(1)");
    assert_eq!(zeros(vec![]).to_string(), "tensor(0)");
}

#[test]
fn eye_is_the_identity() {
    let identity = eye(6);

    assert_eq!(identity.hrect().shape(), &[6, 6]);
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

    assert_eq!(wide.hrect().shape(), &[2, 4]);
    assert_eq!(
        collect_2d(&wide, 2, 4),
        vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]
    );

    assert_eq!(tall.hrect().shape(), &[4, 2]);
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

#[test]
fn unsqueeze_then_squeeze_gives_the_tensor_back() {
    let original = strided();
    let unsqueezed = Rc::new(unsqueeze(original.clone(), 1));

    assert_eq!(unsqueezed.hrect().shape(), &[3, 1, 2]);
    assert_eq!(unsqueezed.get(&[2, 0, 1]), original.get(&[2, 1]));

    let squeezed = squeeze(unsqueezed);
    assert_eq!(squeezed.hrect().shape(), &[3, 2]);
    assert_eq!(collect_2d(&squeezed, 3, 2), collect_2d(&*original, 3, 2));
}

#[test]
fn unsqueeze_can_add_an_axis_before_the_first_or_after_the_last() {
    assert_eq!(unsqueeze(strided(), 0).hrect().shape(), &[1, 3, 2]);
    assert_eq!(unsqueeze(strided(), 2).hrect().shape(), &[3, 2, 1]);
    // A vector with an axis in front is a matrix of one row, and prints as one.
    assert_eq!(
        unsqueeze(physical(), 0).to_string(),
        "tensor([[1, 2, 3, 4, 5, 6]])"
    );
}

#[test]
fn squeeze_drops_every_axis_of_extent_one() {
    let squeezed = squeeze(Rc::new(ones(vec![1, 3, 1, 2])));
    assert_eq!(squeezed.hrect().shape(), &[3, 2]);
    assert_eq!(squeezed.get(&[2, 1]), 1.0);

    // When every axis has extent one, what is left is a scalar.
    let scalar = squeeze(Rc::new(ones(vec![1, 1])));
    assert_eq!(scalar.hrect().ndim(), 0);
    assert_eq!(scalar.to_string(), "tensor(1)");
}

// torch leaves a squeeze of an axis longer than one as a no-op.
#[test]
fn squeeze_dim_leaves_an_axis_that_is_not_of_extent_one_alone() {
    let unsqueezed = Rc::new(unsqueeze(strided(), 0));

    assert_eq!(squeeze_dim(unsqueezed.clone(), 0).hrect().shape(), &[3, 2]);
    assert_eq!(squeeze_dim(unsqueezed, 1).hrect().shape(), &[1, 3, 2]);
}

#[test]
#[should_panic(expected = "out of range")]
fn unsqueeze_rejects_an_axis_past_the_end() {
    unsqueeze(strided(), 3);
}

#[test]
#[should_panic(expected = "out of range")]
fn squeeze_dim_rejects_an_axis_out_of_range() {
    squeeze_dim(strided(), 2);
}

#[test]
fn concatenate_joins_tensors_of_different_lengths_end_to_end() {
    let joined = concatenate(vec![physical(), Rc::new(Physical::new(vec![7.0, 8.0]))], 0);

    assert_eq!(joined.hrect().shape(), &[8]);
    assert_eq!(joined.to_string(), "tensor([1, 2, 3, 4, 5, 6, 7, 8])");
}

// The parts need not be the same kind of tensor: here stored data meets a
// tensor with none at all.
#[test]
fn concatenate_joins_tensors_of_different_kinds() {
    let joined = concatenate(vec![physical(), Rc::new(zeros(vec![2]))], 0);

    assert_eq!(joined.to_string(), "tensor([1, 2, 3, 4, 5, 6, 0, 0])");
}

#[test]
fn concatenate_joins_matrices_along_either_axis() {
    let rows = concatenate(vec![strided(), strided()], 0);
    let columns = concatenate(vec![strided(), strided()], 1);

    assert_eq!(rows.hrect().shape(), &[6, 2]);
    assert_eq!(
        collect_2d(&rows, 6, 2),
        vec![
            vec![1.0, 4.0],
            vec![2.0, 5.0],
            vec![3.0, 6.0],
            vec![1.0, 4.0],
            vec![2.0, 5.0],
            vec![3.0, 6.0],
        ]
    );
    assert_eq!(columns.hrect().shape(), &[3, 4]);
    assert_eq!(
        collect_2d(&columns, 3, 4),
        vec![
            vec![1.0, 4.0, 1.0, 4.0],
            vec![2.0, 5.0, 2.0, 5.0],
            vec![3.0, 6.0, 3.0, 6.0],
        ]
    );
}

#[test]
fn concatenate_passes_over_a_part_of_length_zero() {
    let empty = || Rc::new(Physical::new(vec![]));
    let joined = concatenate(vec![empty(), physical(), empty()], 0);

    assert_eq!(joined.hrect().shape(), &[6]);
    assert_eq!(joined.to_string(), "tensor([1, 2, 3, 4, 5, 6])");
}

#[test]
#[should_panic(expected = "other extents agree")]
fn concatenate_rejects_tensors_whose_other_extents_differ() {
    concatenate(vec![strided(), Rc::new(transpose(strided()))], 0);
}

#[test]
#[should_panic(expected = "at least one tensor")]
fn concatenate_rejects_an_empty_list() {
    concatenate(vec![], 0);
}

#[test]
fn stack_adds_a_new_axis_at_any_position() {
    let parts: Vec<Rc<dyn BoxedTensor>> = vec![strided(), Rc::new(negate(strided()))];
    let first = stack(parts.clone(), 0);
    let middle = stack(parts.clone(), 1);
    let last = stack(parts.clone(), 2);

    assert_eq!(first.hrect().shape(), &[2, 3, 2]);
    assert_eq!(middle.hrect().shape(), &[3, 2, 2]);
    assert_eq!(last.hrect().shape(), &[3, 2, 2]);
    for (t, part) in parts.iter().enumerate() {
        for i in 0..3 {
            for j in 0..2 {
                let expected = part.get(&[i, j]);
                assert_eq!(first.get(&[t, i, j]), expected);
                assert_eq!(middle.get(&[i, t, j]), expected);
                assert_eq!(last.get(&[i, j, t]), expected);
            }
        }
    }
}

#[test]
#[should_panic(expected = "one shape")]
fn stack_rejects_tensors_of_different_shapes() {
    stack(vec![strided(), Rc::new(transpose(strided()))], 0);
}

#[test]
fn where_reads_input_where_the_condition_holds_and_other_elsewhere() {
    let condition = Rc::new(Physical::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0]));
    let picked = where_(condition, physical(), Rc::new(negate(physical())));

    assert_eq!(picked.to_string(), "tensor([ 1, -2, -3,  4, -5,  6])");
}

// Zero reads as false, and so does negative zero, which equals it; anything
// else reads as true, NaN included, as it would in C or Python.
#[test]
fn where_counts_any_condition_other_than_zero_as_true() {
    let condition = Rc::new(Physical::new(vec![-1.0, f32::NAN, 0.5, 0.0, -0.0, 0.0]));
    let picked = where_(condition, physical(), Rc::new(zeros(vec![6])));

    assert_eq!(picked.to_string(), "tensor([1, 2, 3, 0, 0, 0])");
}

// The condition is a tensor like any other: here the identity keeps the
// diagonal of one matrix and takes everything else from another.
#[test]
fn where_can_take_its_condition_from_a_tensor_with_no_data() {
    let m = matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let picked = where_(Rc::new(eye(2)), m.clone(), Rc::new(negate(m)));

    assert_eq!(
        collect_2d(&picked, 2, 2),
        vec![vec![1.0, -2.0], vec![-3.0, 4.0]]
    );
}

#[test]
#[should_panic(expected = "one shape")]
fn where_rejects_a_condition_of_another_shape() {
    where_(Rc::new(ones(vec![2])), physical(), physical());
}

// The example from torch's documentation for `sparse_coo_tensor`.
#[test]
fn sparse_coo_tensor_matches_the_torch_documentation() {
    let sparse = sparse_coo_tensor(
        vec![vec![0, 1, 1], vec![2, 0, 2]],
        Rc::new(Physical::new(vec![3.0, 4.0, 5.0])),
        vec![2, 4],
    );

    assert_eq!(sparse.hrect().shape(), &[2, 4]);
    assert_eq!(
        collect_2d(&sparse, 2, 4),
        vec![vec![0.0, 0.0, 3.0, 0.0], vec![4.0, 0.0, 5.0, 0.0]]
    );
}

// It is the sparse tensor the demo already builds by hand out of a COO, with
// the coordinates given one row per dimension instead of one per value.
#[test]
fn sparse_coo_tensor_is_the_demo_sparse_tensor() {
    let by_hand = Sum::new(
        physical(),
        Rc::new(COO::new(
            HRect::new(vec![6, 5]),
            vec![
                vec![0, 0],
                vec![0, 3],
                vec![4, 2],
                vec![5, 4],
                vec![1, 3],
                vec![2, 0],
            ],
        )),
    );
    let built = sparse_coo_tensor(
        vec![vec![0, 0, 4, 5, 1, 2], vec![0, 3, 2, 4, 3, 0]],
        physical(),
        vec![6, 5],
    );

    assert_eq!(built.to_string(), by_hand.to_string());
}

#[test]
fn sparse_coo_tensor_adds_up_values_at_a_repeated_coordinate() {
    let sparse = sparse_coo_tensor(
        vec![vec![1, 1, 0]],
        Rc::new(Physical::new(vec![2.0, 3.0, 7.0])),
        vec![3],
    );

    assert_eq!(sparse.to_string(), "tensor([7, 5, 0])");
}

#[test]
#[should_panic(expected = "lies outside size")]
fn sparse_coo_tensor_rejects_a_coordinate_outside_its_size() {
    sparse_coo_tensor(vec![vec![3]], Rc::new(Physical::new(vec![1.0])), vec![3]);
}

#[test]
#[should_panic(expected = "one row of indices per dimension")]
fn sparse_coo_tensor_rejects_the_wrong_number_of_rows() {
    sparse_coo_tensor(
        vec![vec![0], vec![0]],
        Rc::new(Physical::new(vec![1.0])),
        vec![3],
    );
}

#[test]
#[should_panic(expected = "a coordinate for each")]
fn sparse_coo_tensor_rejects_a_row_of_the_wrong_length() {
    sparse_coo_tensor(vec![vec![0, 1]], Rc::new(Physical::new(vec![1.0])), vec![3]);
}
