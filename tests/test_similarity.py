from math import fsum, hypot, sqrt

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nexusrag.services.similarity import cosine_similarity


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([1.0, 2.0], [10.0, 20.0], 1.0),
        ([3.0, 4.0], [1.0, 0.0], 0.6),
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 32 / sqrt(1078)),
        ([2.0], [-3.0], -1.0),
    ],
    ids=[
        "same-direction",
        "perpendicular",
        "opposite-direction",
        "different-magnitudes",
        "partial-alignment",
        "three-dimensions",
        "one-dimension",
    ],
)
def test_cosine_similarity_examples(
    left: list[float], right: list[float], expected: float
) -> None:
    assert cosine_similarity(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    "left, right, message",
    [
        ([], [], "empty"),
        ([], [1.0], "empty"),
        ([1.0], [], "empty"),
        ([1.0], [1.0, 2.0], "equal dimensions"),
        ([0.0, 0.0], [1.0, 0.0], "nonzero length"),
        ([1.0, 0.0], [0.0, -0.0], "nonzero length"),
        ([0.0], [0.0], "nonzero length"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(
    left: list[float], right: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cosine_similarity(left, right)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("invalid_side", ["left", "right"])
def test_cosine_similarity_rejects_nonfinite_values(
    invalid: float, invalid_side: str
) -> None:
    left = [invalid, 1.0] if invalid_side == "left" else [1.0, 1.0]
    right = [invalid, 1.0] if invalid_side == "right" else [1.0, 1.0]

    with pytest.raises(ValueError, match="finite"):
        cosine_similarity(left, right)


@pytest.mark.parametrize("magnitude", [1e308, 1e-308, 5e-324])
def test_cosine_similarity_handles_extreme_finite_magnitudes(magnitude: float) -> None:
    assert cosine_similarity([magnitude, magnitude], [1.0, 1.0]) == pytest.approx(1)
    assert cosine_similarity([magnitude, magnitude], [magnitude, -magnitude]) == 0
    assert cosine_similarity([magnitude], [-magnitude]) == pytest.approx(-1)


def test_cosine_similarity_handles_vectors_with_different_scales() -> None:
    assert cosine_similarity([1e308, 1e-308], [1e-308, 1e308]) == pytest.approx(0)


@st.composite
def nonzero_vectors(draw, size: int | None = None):
    if size is None:
        size = draw(st.integers(min_value=1, max_value=20))
    values = draw(
        st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False),
            min_size=size,
            max_size=size,
        )
    )
    # Keep property checks away from underflow; extreme values have examples above.
    values[0] = draw(st.floats(min_value=1, max_value=1000))
    return values


@given(vector=nonzero_vectors())
def test_generated_vectors_have_self_similarity_one(vector: list[float]) -> None:
    assert cosine_similarity(vector, vector) == pytest.approx(1, abs=1e-12)
    assert cosine_similarity(vector, [-value for value in vector]) == pytest.approx(
        -1, abs=1e-12
    )


@given(left=nonzero_vectors(), data=st.data())
def test_generated_similarity_is_symmetric_bounded_and_matches_formula(
    left: list[float], data
) -> None:
    right = data.draw(nonzero_vectors(size=len(left)))
    original_left = left.copy()
    original_right = right.copy()

    score = cosine_similarity(left, right)
    reference = fsum(a * b for a, b in zip(left, right, strict=True)) / (
        hypot(*left) * hypot(*right)
    )

    assert -1 <= score <= 1
    assert score == pytest.approx(cosine_similarity(right, left), abs=1e-12)
    assert score == pytest.approx(reference, abs=1e-12)
    assert left == original_left
    assert right == original_right


@given(
    left=nonzero_vectors(),
    data=st.data(),
    left_scale=st.floats(min_value=0.01, max_value=1000),
    right_scale=st.floats(min_value=0.01, max_value=1000),
)
def test_generated_positive_scaling_does_not_change_similarity(
    left: list[float], data, left_scale: float, right_scale: float
) -> None:
    right = data.draw(nonzero_vectors(size=len(left)))

    assert cosine_similarity(
        [value * left_scale for value in left],
        [value * right_scale for value in right],
    ) == pytest.approx(cosine_similarity(left, right), abs=1e-12)
