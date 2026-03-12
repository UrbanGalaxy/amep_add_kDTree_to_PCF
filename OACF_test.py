# %%
import amep
import numpy as np
import os
import matplotlib.pyplot as plt

class TrajView:
    """Lightweight wrapper to skip frames from an amep trajectory."""
    def __init__(self, traj, start=0):
        self._traj = traj
        self._start = start
        self.nframes = traj.nframes - start
        self.times = traj.times[start:]

    def __iter__(self):
        for i in range(self._start, self._traj.nframes):
            yield self._traj[i]

    def __getitem__(self, idx):
        return self._traj[self._start + idx]

    def __len__(self):
        return self.nframes


# Usage — drop the first frame

# %%
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "test", "data")
data_folder = os.path.join(
    data_dir,
    "lammps_atom_25000_eta_0.596_e_100_Pe_36.0_D_1.0_Dr_2.44861_seed_5.h5amep"
)
traj = amep.load.traj(data_folder)
# traj = TrajView(traj, start=00)
nframes = traj.nframes  # now 1000

# %%
# ── Built-in OACF (single reference frame, unnormalised) ────────────────────
oacf = amep.evaluate.OACF(
    traj=traj,
    nav=traj.nframes ,
    direction='xy',
    max_workers=1
)

oacf_step = amep.evaluate.OACF(
    traj=traj,
    nav=traj.nframes ,
    direction='xy',
    max_workers=2,
    mode='lag_step',
    max_lag_fraction=0.1,
    max_lag_step = 10,
)

oacf_frame = amep.evaluate.OACF(
    traj=traj,
    nav=traj.nframes ,
    direction='xy',
    max_workers=2,
    mode='lag_frame',
    max_lag_fraction=1,
)



# %%
# ── Custom OACF: multi-origin, normalised ───────────────────────────────────
#
# Definition:
#   C(Δt) = < μ̂(t₀) · μ̂(t₀ + Δt) >_{t₀, particles}
#
# where μ̂ is the unit orientation vector (already unit vectors in LAMMPS ABP
# sims, but we normalise anyway for safety).
#
# We use ALL pairs of frames (t₀, t₀+Δt) that share the same lag Δt,
# which gives much better statistics than fixing a single t₀.
# The result is normalised by C(0) = 1 by construction.
#
# For large trajectories you can set `max_lag_fraction` < 1 to restrict the
# maximum lag considered (default: use all lags up to the full trajectory).

def compute_oacf_custom(traj, direction='xy', max_lag_fraction=0.5, ptype=None):
    """
    Compute the orientational autocorrelation function using all available
    time origins (stationary-process / Green–Kubo style).

    Parameters
    ----------
    traj : amep ParticleTrajectory
        Loaded AMEP trajectory.
    direction : str
        Which components of the orientation vector to use.
        Any combination of 'x', 'y', 'z' (e.g. 'xy' for 2-D systems).
    max_lag_fraction : float
        Maximum lag as a fraction of the total number of frames.
        Use 0.5 for the standard "use at most half the trajectory" rule.
    ptype : int or None
        Particle type filter passed to frame.orientations().

    Returns
    -------
    lag_times : np.ndarray  shape (n_lags,)
        Physical times corresponding to each lag.
    oacf_values : np.ndarray  shape (n_lags,)
        C(Δt), normalised so that C(0) = 1.
    """
    # Map direction string → column indices
    col_map = {'x': 0, 'y': 1, 'z': 2}
    cols = [col_map[d] for d in direction if d in col_map]
    if not cols:
        raise ValueError(f"Invalid direction '{direction}'. Use combinations of 'x','y','z'.")

    # Load all orientation vectors once
    print(f"Loading orientations for {traj.nframes} frames …")
    orientations = []   # list of (N_particles, len(cols)) arrays
    times = []
    for i, frame in enumerate(traj):
        mu = frame.orientations(ptype=ptype)[:, cols]   # shape (N, d)
        # Normalise each row to unit length (safety measure)
        norms = np.linalg.norm(mu, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)        # avoid /0
        orientations.append(mu / norms)
        times.append(frame.time)
        if (i + 1) % max(1, traj.nframes // 10) == 0:
            print(f"  … {i+1}/{traj.nframes} frames loaded")

    orientations = np.array(orientations)   # shape (T, N, d)
    times = np.array(times)                 # shape (T,)

    n_frames = len(times)
    max_lag = int(n_frames * max_lag_fraction)

    # Lag indices (every unique Δi = 0, 1, 2, …, max_lag-1)
    lag_indices = np.arange(max_lag)



    oacf_values = np.zeros(max_lag)
    counts = np.zeros(max_lag, dtype=int)

    print("Computing OACF over all time origins …")
    for lag_i in lag_indices:
        # All origin frames t0 such that t0 + lag_i is still in range
        t0_range = np.arange(n_frames - lag_i)
        # dot product per particle: (T-lag, N, d) element-wise then sum over d
        dots = (orientations[t0_range] * orientations[t0_range + lag_i]).sum(axis=-1)
        # average over particles AND time origins
        oacf_values[lag_i] = dots.mean()
        counts[lag_i] = len(t0_range)

        if lag_i % 10 == 0:
            print(f"{lag_i}/{max_lag}")

    # Normalise so C(0) = 1
    oacf_values /= oacf_values[0]

    # Convert lag indices → physical lag times
    # Assumes uniform time spacing; uses the mean dt for robustness
    dt = np.mean(np.diff(times))
    lag_times = lag_indices * dt

    return lag_times, oacf_values


# Run custom OACF
lag_times, oacf_custom = compute_oacf_custom(
    traj,
    direction='xy',       # 2-D system
    max_lag_fraction=0.02, # use full trajectory (set to 0.5 for better stats)
)

# %%
# ── Comparison plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Reference exponential using Dr from filename (Dr = 2.44861 → τ_r = 1/Dr)
# Adjust if your Dr differs.
Dr = 2.44861
t_ref_builtin = oacf.times[1:]
t_ref_builtin = t_ref_builtin * 0.00001
t_ref_builtin_2 = oacf_step.times[1:]
t_ref_builtin_2 = t_ref_builtin_2 * 0.00001
t_ref_builtin_frame = oacf_frame.times[1:]
t_ref_builtin_frame = t_ref_builtin_frame * 0.00001
# t_ref_custom  = lag_times[1:]
# t_ref_custom = t_ref_custom * 0.00001

# --- Left panel: built-in OACF ---
ax = axes[0]
ax.semilogx()
ax.plot(t_ref_builtin, oacf.frames[1:],
        label="built-in OACF", ls="", marker="o", ms=3, c="darkorange")
ax.plot(t_ref_builtin, np.exp(-Dr * t_ref_builtin),
        c="k", label=r"$\exp(-D_r\,t)$", lw=1.5, ls="--")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\langle\hat{\mu}(t)\cdot\hat{\mu}(0)\rangle$")
ax.set_title("Built-in amep.evaluate.OACF\n(single reference frame, unnormalised)")
ax.legend()

ax = axes[1]
ax.semilogx()
ax.plot(t_ref_builtin_2, oacf_step.frames[1:],
        label="custom OACF", ls="", marker="o", ms=3, c="darkorange")
ax.plot(t_ref_builtin_2, np.exp(-Dr * t_ref_builtin_2),
        c="k", label=r"$\exp(-D_r\,t)$", lw=1.5, ls="--")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\langle\hat{\mu}(t)\cdot\hat{\mu}(0)\rangle$")
ax.set_title("Built-in amep.evaluate.OACF\n(multible reference frames, unnormalised)")
ax.legend()
# --- Right panel: custom OACF ---
ax = axes[2]
ax.semilogx()
ax.plot(t_ref_builtin_frame, oacf_frame.frames[1:],
        label="custom OACF", ls="", marker="o", ms=3, c="darkorange")
ax.plot(t_ref_builtin_frame, np.exp(-Dr * t_ref_builtin_frame),
        c="k", label=r"$\exp(-D_r\,t)$", lw=1.5, ls="--")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\langle\hat{\mu}(t)\cdot\hat{\mu}(0)\rangle$")
ax.set_title("Custom OACF\n(all time origins, normalised, C(0)=1)")
ax.legend()

fig.tight_layout()
output_path = os.path.join(script_dir, "oacf_comparison.png")
fig.savefig(output_path, dpi=150)
plt.show()
print(f"Figure saved to {output_path}")
# %%
frame = traj[0]
mu = frame.orientations()
print(mu.shape)
print(mu[:5])
print(np.linalg.norm(mu, axis=1)[:5])
print(0.0883579**2 + 0.744777**2)
print(0.518026**2 + 0.542356**2)
# %%
