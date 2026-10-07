from math import fsum, isfinite, sqrt


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compare vector directions, returning a score between -1 and 1.

    Reject empty vectors, unequal dimensions, non-finite values, and zero
    vectors with ValueError. Inputs are not mutated. Positive scaling leaves
    the score unchanged; scaling before arithmetic avoids overflow/underflow
    when squaring very large or very small finite values.
    """
    if not left or not right:
        raise ValueError("vectors must not be empty")
    if len(left) != len(right):
        raise ValueError("vectors must have equal dimensions")
    if not all(isfinite(value) for value in left) or not all(
        isfinite(value) for value in right
    ):
        raise ValueError("vector values must be finite")

    left_scale = max(abs(value) for value in left)
    right_scale = max(abs(value) for value in right)
    if left_scale == 0 or right_scale == 0:
        raise ValueError("vectors must have nonzero length")

    scaled_left = [value / left_scale for value in left]
    scaled_right = [value / right_scale for value in right]
    dot_product = fsum(a * b for a, b in zip(scaled_left, scaled_right, strict=True))
    left_length = sqrt(fsum(value * value for value in scaled_left))
    right_length = sqrt(fsum(value * value for value in scaled_right))
    similarity = dot_product / (left_length * right_length)

    # Floating-point rounding can put an otherwise valid score just past ±1.
    return max(-1.0, min(1.0, similarity))
