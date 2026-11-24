# Working with the archived experiments

## Verify the saved results

The figure PNGs in [`Plots/`](../Plots/) are the original outputs. The matching return arrays in [`DDPG_self/data/Experiment/`](../DDPG_self/data/Experiment/) contain 51 evaluations per run, including initialization and the final evaluation. An independent check of the archived arrays recovered all 14 area ratios in thesis Tables 4.4–4.6 to the precision reported in the thesis.

The historical plotting script needs NumPy, SciPy, and Matplotlib. To run it in a separate Python 3.9 environment, start at the repository root.

```sh
python3 --version  # Use Python 3.9.x for this setup.
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy==1.26.4 scipy==1.12.0 matplotlib==3.8.4
cd DDPG_self
python plots.py
```

The default configuration reads the Hopper → Walker2D parameter-transfer comparison and reports an area ratio of about `0.0431`. Use `MPLBACKEND=Agg python plots.py` for the numerical output without opening a plot window. The script reads existing arrays and leaves them intact.

| Comparison | Directory under `DDPG_self/` | File prefix |
| --- | --- | --- |
| Uniform replay transfer | `data/Experiment/1e4/`, `1e5/`, or `1e6/` | `DDPG_` |
| Prioritized replay transfer | The same capacity subdirectories | `Priority_DDPG_` |
| Parameter transfer | `data/Experiment/` | `testing_Priority_DDPG_` |

Each prefix is followed by the target name (`Hopper` or `Walker2D`), an underscore, and the numeric capacity. The parameter-transfer comparison files use `1000000` in their names. The suffixes are `_0.npy` through `_4.npy` for target-only runs and `_Transfer_0.npy` through `_Transfer_4.npy` for transfer runs. The script averages the five curves per condition, smooths each mean with a three-point uniform filter, and integrates with the trapezoidal rule. Its relative change is `(transfer area - baseline area) / baseline area`. A common evaluation-step spacing cancels out of that ratio.

## Inspect or rerun the training code

The original environment manifest was absent. [`requirements.txt`](../requirements.txt) pins a compatibility stack for Python 3.9 and CPU execution. It also includes the plotting dependencies. From the repository root, with the environment above active, install the full set in the same environment.

```sh
python -m pip install -r requirements.txt
```

On Apple Silicon with Xcode 27.2, PyBullet 3.2.7's bundled zlib conflicted with the SDK's `fdopen` declaration during a source build. This local build and installation worked on that setup.

```sh
ARCHFLAGS="-arch arm64" CFLAGS="-include stdio.h" python -m pip wheel --no-deps --wheel-dir /tmp/ddpg-thesis-wheels pybullet==3.2.7
python -m pip install --find-links /tmp/ddpg-thesis-wheels -r requirements.txt
python -m pip check
```

Use the resulting wheel on the architecture that built it.

The training code imports `pybullet_envs` to register its environments, expects `env.seed()`, and uses Gym's four-value `step()` API. Modern Gym or Gymnasium versions therefore require source changes. PyBullet may build from source on platforms without a matching wheel, which requires a C/C++ toolchain. Use CPU execution because the archived code creates CPU tensors in paths that can otherwise select CUDA.

| Script | Role and checked-in settings |
| --- | --- |
| [`main_weight.py`](../DDPG_self/main_weight.py) | Parameter transfer. Defaults to Walker2D → Hopper, prioritized sampling, capacity 100,000, and seed 0. |
| [`main.py`](../DDPG_self/main.py) | Replay-buffer transfer. Defaults to uniform replay and capacity 1,000,000. The source and target are both set to Walker2D while `case = 1` truncates transitions to Hopper dimensions. The non-prioritized source filename also hard-codes run index 1. |
| [`original_main.py`](../DDPG_self/original_main.py) | Standalone baseline trainer with its own relative output directory. |
| [`main_experimental.py`](../DDPG_self/main_experimental.py) | Exploratory buffer-filling script with its own relative output directory. |

For a trial run, copy `DDPG_self/` to a scratch directory, change into that copy, and set the script's `repository` output template to a fresh directory ending in `/%s`. Create the directory before running. The scripts save results at initialization and throughout training and can overwrite archived arrays if run in place. Invoke the intended script from the scratch copy, for example `python main_weight.py` for the checked-in parameter-transfer configuration. A Hopper → Walker2D trial uses `case = 0`, and Walker2D → Hopper uses `case = 1`. The replay-buffer script needs source and target IDs that agree with the transfer direction. It also ends with a shell call to `mr.bigstuff.mp3`. Omit that call in a scratch copy before executing the script.

On Apple Silicon with Python 3.9.6 and the pinned dependencies, shortened CPU runs of all three transfer methods completed in both directions using scratch copies. These checks used 320 training steps per task, a buffer capacity of 128, and a 160-step evaluation interval. Replay runs used the target ID corresponding to each mapping and omitted the shell call. Each run produced finite source and target evaluation arrays. Full 100,000-step training runs remain unverified. The entry points use settings edited in source, lack a per-run configuration manifest, and include settings that differ from the thesis or the archived filenames. For example, the scripts use exploration noise with standard deviation 0.1, while Section 3.2 lists 0.05. Python's `random` sampler used by PER is also left unseeded by the scripts. The training loops check the evaluation schedule at episode boundaries, so individual evaluations can occur after the nominal 2,000-step interval.

## Organization and archival files

The original scripts and their relative paths are the reference layout. Use `DDPG_self/` for source and in-file experiment settings, `DDPG_self/data/` for archived returns, `Plots/` for the thesis figures, and `docs/` for documentation. For future experiments, keep per-run settings in a dedicated configuration directory and write generated outputs to a separate results directory. A migration of the historical layout would require updating the legacy scripts and checking that plots still resolve their source arrays.

The root ignore rules cover local Python, IDE, and OS artifacts. Eight pairs of saved uniform-replay baseline arrays are byte-identical across the 100,000 and 1,000,000 capacity folders. Each path remains available for its corresponding comparison.
