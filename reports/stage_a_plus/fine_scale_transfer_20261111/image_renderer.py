"""Float32 image interface for four named factors controlling two Gaussian spots."""

import json
from pathlib import Path

import numpy as np

FACTOR_NAMES = ('red_x', 'red_y', 'green_x', 'green_y')
DOSE = .0175
SIZE = 64
GRID_Y, GRID_X = np.indices((SIZE, SIZE), dtype=np.float64)


def render(factors):
    factors = np.asarray(factors, dtype=np.float64)
    if factors.ndim == 0 or factors.shape[-1] != 4 or not np.isfinite(factors).all():
        raise ValueError('Expected finite states with four factors on the last axis')
    centers = (32 + 20 * np.tanh(factors / 4)).reshape(factors.shape[:-1] + (2, 2))
    dx = GRID_X - centers[..., :, 0, None, None]
    dy = GRID_Y - centers[..., :, 1, None, None]
    spots = np.exp(-(dx**2 + dy**2) / (2 * 3**2)).astype(np.float32)
    rgb = np.zeros(factors.shape[:-1] + (SIZE, SIZE, 3), dtype=np.float32)
    rgb[..., :2] = np.moveaxis(spots, -3, -1)
    return rgb


def named_dose_images(factors, dose=DOSE):
    """Produce observed images for positive and negative named interventions."""
    if not np.isfinite(dose) or dose <= 0:
        raise ValueError('Expected a positive finite dose')
    factors = np.asarray(factors, dtype=np.float64)
    if factors.ndim == 0 or factors.shape[-1] != 4:
        raise ValueError('Expected four factors on the last axis')
    offsets = dose * np.eye(4)
    return render(factors[..., None, :] + offsets), render(factors[..., None, :] - offsets)


def check_interface():
    states = np.array([[0., 0., 0., 0.], [1., -2., 3., -4.], [-4., 4., 4., -4.],
                       [12., 0., -12., 2.]])
    baseline = render(states)
    plus, minus = named_dose_images(states)
    assert baseline.shape == (4, SIZE, SIZE, 3) and baseline.dtype == np.float32
    assert plus.shape == minus.shape == (4, 4, SIZE, SIZE, 3)
    assert np.array_equal(baseline, render(states))
    for i in range(4):
        unchanged = [channel for channel in range(3) if channel != i // 2]
        np.testing.assert_array_equal(plus[:, i][..., unchanged], baseline[..., unchanged])
        np.testing.assert_array_equal(minus[:, i][..., unchanged], baseline[..., unchanged])
        assert np.any(plus[:, i] != minus[:, i], axis=(1, 2, 3)).all()

    def response(p, m, dose):
        return ((p.astype(np.float64) - m) / (2*dose)).reshape(len(states), 4, -1)

    full = response(plus, minus, DOSE)
    half_plus, half_minus = named_dose_images(states, DOSE / 2)
    half = response(half_plus, half_minus, DOSE / 2)
    singular = np.linalg.svd(full, compute_uv=False)
    relative_delta = np.linalg.norm(full - half, axis=-1) / np.linalg.norm(half, axis=-1)
    assert (singular[:, -1] > 0).all() and np.isfinite(relative_delta).all()
    return dict(role='deterministic_cpu_interface_check', factor_names=FACTOR_NAMES, states=states.tolist(),
                dose=DOSE, image_shape=list(baseline.shape), branch_shape=list(plus.shape), dtype=str(baseline.dtype),
                repeat_exact=True, single_spot_channel_isolation=True,
                response_singular_values=singular.tolist(),
                response_condition=(singular[:, 0] / singular[:, -1]).tolist(),
                half_dose_derivative_relative_delta=relative_delta.tolist())


if __name__ == '__main__':
    result = check_interface()
    destination = Path(__file__).with_name('IMAGE_INTERFACE_CHECK.json')
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('CPU image interface checked on four fixed states')
