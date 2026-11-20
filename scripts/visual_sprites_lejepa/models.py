"""Two-sprite encoders and the common FourRoom readout protocol."""

import functools
import importlib.util
import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def sigreg_path():
    return (ROOT / "work/hcf_stable_id_20261027/v1/vendor/lejepa_identifiability"
            / "experiments/lejepa_id/losses.py")


@functools.cache
def _vendor_losses():
    """Load the original SIGReg without writing cache files beside the source."""
    path = sigreg_path()
    spec = importlib.util.spec_from_file_location("visual_sprites_vendor_losses", path)
    if spec is None:
        raise RuntimeError(f"Cannot import the vendored losses from {path}")
    module = importlib.util.module_from_spec(spec)
    # Execute the source bytes directly, independently of any cached bytecode.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


class SpriteEncoder(nn.Module):
    """Use the same four strided convolutions in both arms."""

    def __init__(self, latent_dim):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 64), nn.LayerNorm(64), nn.Linear(64, latent_dim)
        )

    def forward(self, frames):
        return self.projection(self.convolutions(frames))


def make_encoder(latent_dim, device):
    return SpriteEncoder(latent_dim).to(device, memory_format=torch.channels_last)


def _build_pretrainer(settings, arm, device):
    d = settings["latent_dim"]
    encoder = make_encoder(d, device)
    predictor = None
    if arm == "long_horizon_pred":
        predictor = nn.Sequential(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, d)).to(device)
    elif arm != "visual_lejepa":
        raise ValueError(f"Unknown training arm: {arm}")
    parameters = list(encoder.parameters())
    if predictor is not None:
        parameters.extend(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    return encoder, predictor, _vendor_losses().SIGReg().to(device), optimizer


def _frame_tensor(frames, device):
    tensor = torch.from_numpy(frames).to(device, memory_format=torch.channels_last)
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255
    if tensor.dtype != torch.float32:
        raise TypeError("Rendered input must be float32 or uint8")
    return tensor


def _train_step(encoder, predictor, regularizer, optimizer, first, second, arm, settings):
    optimizer.zero_grad(set_to_none=True)
    if arm == "visual_lejepa":
        latents = encoder(torch.cat([first, second])).reshape(2, len(first), -1)
        matching = (latents[0] - latents[1]).square().mean()
        sigreg = regularizer(latents)
        weight = settings["lejepa_sigreg_weight"]
        loss = (1 - weight) * matching + weight * sigreg
    else:
        current = encoder(first)
        with torch.no_grad():
            target = encoder(second)
        matching = (predictor(current) - target).square().mean()
        sigreg = regularizer(current.unsqueeze(0))
        loss = matching + settings["predictor_sigreg_weight"] * sigreg

    loss.backward()
    optimizer.step()
    return loss, matching, sigreg


def configure_device(name, cpu_threads):
    """Run a small numerical smoke check when MPS is available."""
    if name not in ("auto", "cpu", "mps"):
        raise ValueError(f"Unknown device {name!r}")
    torch.set_num_threads(cpu_threads)
    if name == "cpu" or (name == "auto" and not torch.backends.mps.is_available()):
        return torch.device("cpu")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device("mps")
    settings = {"latent_dim": 8, "learning_rate": .001, "weight_decay": .0001,
                "lejepa_sigreg_weight": .02, "predictor_sigreg_weight": .001}
    try:
        frames = torch.rand(16, 3, 32, 32, device=device).contiguous(memory_format=torch.channels_last)
        for arm in ("visual_lejepa", "long_horizon_pred"):
            components = _build_pretrainer(settings, arm, device)
            terms = _train_step(*components, frames[:8], frames[8:], arm, settings)
            if not all(torch.isfinite(term).item() for term in terms):
                raise FloatingPointError(f"Nonfinite {arm} preflight loss")
            if not all(torch.isfinite(p).all().item() for p in components[0].parameters()):
                raise FloatingPointError(f"Nonfinite {arm} preflight parameters")
    except (FloatingPointError, RuntimeError, NotImplementedError) as exc:
        if name == "mps":
            raise
        LOGGER.warning("MPS numerical check failed, selecting CPU: %s", exc)
        return torch.device("cpu")
    return device


def train_encoder(pretrain_images, cfg, arm, seed, device):
    """Train from image pairs using matched seeds and anchor indices across arms."""
    settings = cfg["encoder"]
    if pretrain_images.ndim != 5 or pretrain_images.shape[2:] != (3, 32, 32):
        raise ValueError("PRETRAIN images must have shape [episode, time, 3, 32, 32]")
    lag = settings["short_lag"] if arm == "visual_lejepa" else settings["long_lag"]
    episodes, frames = pretrain_images.shape[:2]
    n_anchors = frames - max(settings["short_lag"], settings["long_lag"])
    if n_anchors <= 0 or episodes <= 0:
        raise ValueError("PRETRAIN trajectories are shorter than the lag protocol")
    torch.manual_seed(seed)
    # Arm-specific Torch draws cannot change the shared anchor stream.
    pair_rng = np.random.default_rng(seed)
    components = _build_pretrainer(settings, arm, device)
    batch_size = settings["batch_size"]
    trace = []
    start = time.perf_counter()
    for step in range(1, settings["steps"] + 1):
        ep = pair_rng.integers(episodes, size=batch_size)
        pos = pair_rng.integers(n_anchors, size=batch_size)
        first = _frame_tensor(pretrain_images[ep, pos], device)
        second = _frame_tensor(pretrain_images[ep, pos + lag], device)
        terms = _train_step(*components, first, second, arm, settings)
        if step == 1 or step % settings["log_every"] == 0 or step == settings["steps"]:
            values = [term.item() for term in terms]
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Nonfinite loss in {arm} seed {seed} step {step}")
            record = dict(zip(("total_loss", "pair_loss", "sigreg_loss"), values))
            record.update(step=step, elapsed_seconds=time.perf_counter() - start)
            trace.append(record)
            LOGGER.info("%s seed %s step %s/%s: loss %.6f, elapsed %.1fs", arm, seed,
                        step, settings["steps"], values[0], record["elapsed_seconds"])
    encoder = components[0]
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder, trace


@torch.inference_mode()
def encode(encoder, frames, device, batch_size=512):
    """Return CPU float32 features for a channel-first batch of RGB frames."""
    if frames.ndim != 4 or frames.shape[1:] != (3, 32, 32):
        raise ValueError("Encoding expects [frame, 3, 32, 32]")
    if not len(frames) or batch_size <= 0:
        raise ValueError("Encoding requires at least one image and a positive batch size")
    result = np.concatenate([
        encoder(_frame_tensor(frames[start:start + batch_size], device)).cpu().numpy()
        for start in range(0, len(frames), batch_size)
    ])
    if not np.isfinite(result).all():
        raise FloatingPointError("Encoder features are not finite")
    return result


def _make_probe(dimension, hidden, device):
    return nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4)).to(device)


def _standardize(values, mean, std, device):
    return torch.as_tensor((values - mean) / std, dtype=torch.float32, device=device)


def fit_probe(train_x, train_y, val_x, val_y, cfg, seed, device):
    """Fit the FourRoom MLP and select the minimum mean VAL MSE."""
    settings = cfg["probe"]
    torch.manual_seed(seed)
    x_mean, x_std = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
    y_mean, y_std = train_y.mean(0), train_y.std(0)
    if np.any(y_std <= 0) or not np.isfinite(y_std).all():
        raise ValueError("Invalid TRAIN physical-factor variances")
    x = _standardize(train_x, x_mean, x_std, device)
    y = _standardize(train_y, y_mean, y_std, device)
    vx = _standardize(val_x, x_mean, x_std, device)
    vy = _standardize(val_y, y_mean, y_std, device)
    probe = _make_probe(x.shape[1], settings["hidden_dim"], device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    best_loss, best_step, best, trace = float("inf"), None, None, []
    for step in range(1, settings["steps"] + 1):
        ids = torch.randint(len(x), (settings["batch_size"],), device=device)
        loss = (probe(x[ids]) - y[ids]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % settings["log_every"] == 0 or step == settings["steps"]:
            with torch.no_grad():
                val_errors = (probe(vx) - vy).square()
                val_loss = val_errors.mean().item()
                train_loss = loss.item()
                if not np.isfinite([val_loss, train_loss]).all():
                    raise FloatingPointError(f"Nonfinite probe loss at step {step}")
                trace.append({"step": step, "train_loss": train_loss, "val_loss": val_loss,
                              "val_mse_by_axis": val_errors.mean(dim=0).cpu().tolist()})
                if val_loss < best_loss:
                    best_loss, best_step = val_loss, step
                    best = {name: value.detach().cpu().clone() for name, value in probe.state_dict().items()}
    if best is None:
        raise RuntimeError("No readout checkpoint was selected")
    artifact = {"state_dict": best, "input_dim": x.shape[1], "hidden": settings["hidden_dim"],
                "x_mean": torch.from_numpy(x_mean), "x_std": torch.from_numpy(x_std),
                "y_mean": torch.from_numpy(y_mean), "y_std": torch.from_numpy(y_std),
                "best_step": best_step, "validation_loss": best_loss, "seed": seed}
    return artifact, trace


@torch.inference_mode()
def predict(artifact, features, device):
    """Denormalize readout predictions in CPU float64 before scoring."""
    probe = _make_probe(artifact["input_dim"], artifact["hidden"], device)
    probe.load_state_dict(artifact["state_dict"])
    probe.eval()
    mean, std = artifact["x_mean"].numpy(), artifact["x_std"].numpy()
    output = []
    for start in range(0, len(features), 4096):
        x = _standardize(features[start:start + 4096], mean, std, device)
        output.append(probe(x).cpu().numpy())
    output = np.concatenate(output).astype(np.float64)
    output *= artifact["y_std"].numpy()
    output += artifact["y_mean"].numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("Readout predictions are not finite")
    return output


def profile_pretraining(cfg, device, steps=8):
    """Measure compute on random RGB arrays, independently of experiment data."""
    settings = cfg["encoder"]
    batch_size = settings["batch_size"]
    torch.manual_seed(0)
    first = torch.rand(batch_size, 3, 32, 32, device=device).contiguous(memory_format=torch.channels_last)
    second = torch.rand_like(first)
    report = {}
    for arm in ("visual_lejepa", "long_horizon_pred"):
        components = _build_pretrainer(settings, arm, device)
        _train_step(*components, first, second, arm, settings)[0].item()
        start = time.perf_counter()
        for _ in range(steps):
            terms = _train_step(*components, first, second, arm, settings)
        if not np.isfinite(terms[0].item()):
            raise FloatingPointError(f"Nonfinite compute profile for {arm}")
        report[arm] = {"steps": steps, "batch_size": batch_size,
                       "seconds_per_step": (time.perf_counter() - start) / steps}
    return report
