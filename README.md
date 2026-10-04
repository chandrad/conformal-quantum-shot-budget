# Can You Trust Your Shot Budget? — Code & Results

Code and result files for *"Can You Trust Your Shot Budget? Benchmarking Conformal
Risk Control for Variational Quantum Algorithms"* by Chandra Dronavajjala, accepted
at the **SaTQuML (Secure and Trustworthy Quantum Machine Learning) workshop, NeurIPS 2026**.
Paper: https://openreview.net/forum?id=cWuc45q6Pb

Every number in the paper traces to a JSON file in `results/`, produced by a script
in `experiments/`. Simulation uses PennyLane; hardware uses Qiskit Runtime.

## What the paper is about

A variational quantum algorithm (VQA) estimates a scalar (a molecular energy, a cut
value) by measuring a circuit many times. The number of measurements is the **shot
budget**: too few shots and the estimate is noise, too many and you burn scarce
quantum hardware time. In practice the budget is chosen by folklore, with no guarantee
attached. This paper asks whether the shot budget can instead carry a formal,
distribution-free risk guarantee, and it charges every method for the shots its own
guarantee consumes (**strict budget accounting**: an `M`-shot evaluation costs `M`
circuit executions). We benchmark five budget-selection methods, from conformal risk
control (CRC, the risk-controlling / learn-then-test family) to classical statistical
intervals, on molecular ground-state problems (H2 at four bond lengths, LiH) and a
real 156-qubit IBM Heron processor.

### Main results

- **No free lunch among the standard methods.** The distribution-free methods stay
  valid (hold the target error rate) but never save a shot; the thrifty parametric
  methods (Gaussian-*t*, bootstrap) break their promised error rate in up to three of
  every five runs at tight budgets.
- **A hard certification floor.** No method can certify *any* saving below
  `n_min = ceil(ln δ / ln(1 − α)) = 22` calibration evaluations at `(α, δ) = (0.1, 0.1)`.
- **The tradeoff is not fundamental.** An exact Clopper–Pearson method that recycles a
  single shot pool to calibrate many budgets at once keeps both validity and the saving
  once the budget clears the floor, cutting the shot count by more than half.
- **It generalizes.** The same tradeoff and its reuse-based closure recur on QAOA
  (quantum approximate optimization algorithm) MaxCut. Quantum phase estimation (QPE)
  is resolution-limited rather than shot-limited, so the tradeoff dissolves there,
  delimiting the scope of the result.
- **On real hardware the certificate behaves as a trustworthy gate should.** It refuses
  to approve a budget when device noise, not shot noise, dominates (deep H2: every
  budget fails), and it passes a symmetry-reduced 1-qubit H2 circuit at `M = 5000`
  (0 of 22 trials violate; Clopper–Pearson upper bound 0.099 ≤ 0.1).

## Setup
```bash
pip install -r requirements.txt
```
Hardware scripts additionally require an IBM Quantum account configured for
`qiskit-ibm-runtime` (free open plan is sufficient). Simulation scripts need no account.

## Script → result → paper claim

| Script (`experiments/`) | Result (`results/`) | Paper |
|---|---|---|
| `exp_vqe_crc_v4.py` | `vqe_crc_v4_camera_ready.json` | Table 1, Table 4 (main benchmark; 5 methods, H2 x4 bonds + LiH, strict budget accounting) |
| `exp_ibm_vqe_v2.py` | `ibm_vqe_hardware.json` | Table 2, Fig. 1 (VQE on `ibm_kingston`; all budgets refuse) |
| `exp_vqe_crc_v5_cp.py` | `vqe_crc_v5_cp.json` | Sec. 5.6, Fig. 2 (Clopper-Pearson + shot-pool reuse closes the gap; equal-cost ablation) |
| `exp_cp_adaptive.py` | `cp_adaptive.json` | Sec. 5.6 (adaptive-allocation variant; insufficient without reuse) |
| `exp_qaoa_crc.py` | `qaoa_crc.json` | Sec. 5.7, Table 3 (QAOA MaxCut: tradeoff recurs) |
| `exp_qaoa_hardware.py` | `qaoa_hardware.json` | Sec. 5.7 (QAOA on real hardware; noise-limited) |
| `exp_qpe_crc.py` | `qpe_crc.json` | Sec. 5.7 (phase estimation: resolution-limited contrast) |
| `exp_vqe_crc_sensitivity.py` | `vqe_crc_sensitivity.json` | Sec. 5.8, Table 5 (full alpha x epsilon x budget grid) |
| `exp_vqe_hardware_mitigated.py` | `ibm_vqe_hardware_mitigated.json` | Sec. 5.4 (readout + zero-noise extrapolation; still refuses) |
| `exp_h2_reduced_hardware.py` | `ibm_h2_reduced_pass.json` | Sec. 5.4 (symmetry-reduced 1-qubit H2: certificate PASSES on hardware) |
| `gen_figures.py` | (reads the above) | Figure 1 (`hw_vs_sim`) and Figure 2 (`closure`); also a supplementary mitigation plot |

## Notes
- All methods are scored under **strict budget accounting**: each evaluation at `M`
  shots costs `M` circuit executions, charged against the total budget `B`.
- `exp_vqe_crc_v5_cp.py`, the QAOA/QPE scripts, and `exp_cp_adaptive.py` reuse the
  v4 method implementations via a local import of `exp_vqe_crc_v4.py`.
- Hardware scripts need an IBM Quantum account configured for `qiskit-ibm-runtime`
  (via `QiskitRuntimeService`); no token is stored in this repo. The backend is
  chosen at run time by `least_busy`, so the specific device (`ibm_kingston`,
  `ibm_fez`, ...) varies by run. Each hardware JSON records the backend name and
  the IBM Quantum job identifiers for that run.
- Randomness in the simulation experiments enters only through measurement shot
  noise (the ansatz is trained once by deterministic gradient descent); each
  "trial" is an independent set of shot-noise draws, seeded per trial.

## Citation

If you use this code or results, please cite:

```bibtex
@inproceedings{dronavajjala2026shotbudget,
  title     = {Can You Trust Your Shot Budget? Benchmarking Conformal Risk Control
               for Variational Quantum Algorithms},
  author    = {Dronavajjala, Chandra},
  booktitle = {NeurIPS 2026 Workshop on Secure and Trustworthy Quantum Machine
               Learning (SaTQuML)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=cWuc45q6Pb},
}
```
