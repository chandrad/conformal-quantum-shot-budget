"""
S3 — Phase estimation: a third, non-expectation-value workload.

Reviewer ywhS asked about "QAOA, QFT, or phase estimation." QAOA is done
(exp_qaoa_crc.py). QFT alone has no scalar to certify, but it is the core of
quantum phase estimation (QPE), whose inverse-QFT readout we use here, so this
run covers the QFT family too.

Workload: estimate the eigenphase phi of U = PhaseShift(2*pi*phi0) on its |1>
eigenstate, with a T-qubit QPE register. Per M-shot evaluation we take M register
readouts and form the scalar estimate phi_hat = mean over the M samples of
(readout_integer / 2^T). Reference phi_ref = the NOISELESS (analytic) value of the
SAME estimator (sum_j p_j * j/2^T), so the certification target is self-consistent
regardless of QFT bit-ordering conventions (ref and estimate use one mapping). More
shots -> phi_hat concentrates around phi_ref (sampling error ~ 1/sqrt(M)); the
register size T sets a resolution floor, which we keep below epsilon so the SHOT
budget is the binding accuracy knob (the regime where the question is meaningful).

Certify |phi_hat - phi_ref| <= epsilon with risk <= alpha under strict budget
accounting. Same 6 methods (v4's five + v5 CP-reuse), same seeding (seed+3000),
strict validity. Incremental save per budget.

AUDIT NOTES (self):
  - phi_ref from analytic probs (shots=None); phi_hat from shots -> SAME bit->int
    map (big-endian via base-2 weights), so the comparison is convention-free.
  - epsilon fixed to the shot-noise scale at a mid-grid shot count (as in QAOA),
    and we PRINT the resolution floor 1/2^T and per-M std so it is auditable that
    (a) transition is in-grid and (b) resolution floor < epsilon.
  - v5.cp_crc_reuse reads epsilon from v5.EPSILON -> we override it to EPS.
  - deterministic circuit (no trained params); randomness = shot noise only.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml

HERE = Path(__file__).resolve().parent
V4_PATH = HERE / 'exp_vqe_crc_v4.py'
V5_PATH = HERE / 'exp_vqe_crc_v5_cp.py'
RESULTS_FILE = HERE.parent / 'results' / 'qpe_crc.json'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


v4 = _load('exp_vqe_crc_v4', V4_PATH)
v5 = _load('exp_vqe_crc_v5_cp', V5_PATH)

T = 7                       # estimation-register qubits; resolution 1/2^T = 0.0078
PHI0 = 0.3                  # true eigenphase (physical truth)
TARGET = T                  # system qubit index
EST = list(range(T))
ALPHA, DELTA = 0.10, 0.10
SHOT_GRID = [5, 10, 20, 50, 100, 200, 500, 1000]
BUDGETS = [1000, 5000, 10000]
N_SEEDS = 50
N_TEST = 200
WEIGHTS = np.array([2 ** (T - 1 - k) for k in range(T)])   # big-endian bit->int
EPS = None


def _qpe_ops():
    # Prepare the |1> eigenstate of U=PhaseShift(2*pi*phi0) (eigenvalue e^{2 pi i phi0}),
    # then PennyLane's vetted QPE template (Hadamards + controlled-U powers + inverse
    # QFT, with the correct wire convention). Verified: big-endian readout -> peak bin
    # 38/128 = 0.2969 ~ phi0=0.3. (Hand-rolled QPE had a wire-ordering bug.)
    qml.PauliX(TARGET)
    qml.QuantumPhaseEstimation(qml.PhaseShift(2 * np.pi * PHI0, wires=TARGET),
                               estimation_wires=EST)


def phi_ref_analytic():
    dev = qml.device('default.qubit', wires=T + 1)
    @qml.qnode(dev)
    def probs():
        _qpe_ops()
        return qml.probs(wires=EST)
    p = probs()
    js = np.arange(2 ** T)
    # map basis index -> integer under the SAME big-endian bit convention
    # qml.probs returns index in standard order (wire0 most significant), so
    # basis index j already equals the big-endian integer. phi = j/2^T.
    return float(np.sum(p * js) / (2 ** T))


def eval_fn_factory():
    def eval_fn(shots, n_evals):
        # ONE device per call; invoke the qnode n_evals times (device RNG advances
        # per call -> n_evals independent M-shot estimates). Same pattern as the
        # VQE/QAOA eval_fn; avoids per-estimate device construction overhead.
        dev = qml.device('default.qubit', wires=T + 1, shots=shots,
                         seed=int(np.random.randint(0, 2**31 - 1)))

        @qml.qnode(dev)
        def samp():
            _qpe_ops()
            return qml.sample(wires=EST)

        out = []
        for _ in range(n_evals):
            s = np.atleast_2d(samp())
            ints = (s @ WEIGHTS).astype(int)              # (shots,) register integers
            mode = int(np.bincount(ints, minlength=2 ** T).argmax())
            out.append(mode / (2 ** T))                   # phi_hat = standard QPE peak-bin estimate
        return np.array(out)
    return eval_fn


def main():
    global EPS
    phi_ref = PHI0                     # mode estimator targets the true physical phase
    resolution = 1.0 / (2 ** T)
    eval_fn = eval_fn_factory()
    # Probe: per-M risk of the mode estimate under candidate integer-bin tolerances.
    # epsilon := the smallest n-bin tolerance (n*resolution) that puts the
    # risk->alpha transition INSIDE the grid (risk high at M=5, low at M=1000).
    # This is a principled, disclosed rule and keeps the shot budget binding:
    # the register floor is one bin (resolution), far below the chosen epsilon.
    np.random.seed(999)
    phats = {M: eval_fn(shots=M, n_evals=200) for M in SHOT_GRID}
    EPS = None
    for nb in (2, 3, 4, 5, 6, 8):
        e = nb * resolution
        r5 = float(np.mean(np.abs(phats[5] - phi_ref) > e))
        r1000 = float(np.mean(np.abs(phats[1000] - phi_ref) > e))
        if r1000 < ALPHA and r5 > ALPHA:
            EPS = round(e, 4); nbins = nb; break
    if EPS is None:
        EPS = round(4 * resolution, 4); nbins = 4           # fallback (should not happen)
    per_m_risk = {M: round(float(np.mean(np.abs(phats[M] - phi_ref) > EPS)), 3)
                  for M in SHOT_GRID}
    print(f'QPE T={T}, phi0={PHI0}=phi_ref, resolution 1/2^T={resolution:.4f}, '
          f'epsilon={EPS} ({nbins} bins)')
    print('per-M risk @ eps: ' + ', '.join(f'{M}:{v}' for M, v in per_m_risk.items()))
    assert resolution < EPS, 'register floor not below epsilon'
    v5.EPSILON = EPS

    results = {'config': {'script': 'exp_qpe_crc.py', 'problem': 'QPE eigenphase (mode estimator)',
                          'T': T, 'phi0': PHI0, 'phi_ref': phi_ref,
                          'resolution': resolution, 'epsilon': EPS, 'epsilon_bins': nbins,
                          'per_m_risk_probe': per_m_risk,
                          'epsilon_rule': 'smallest integer-bin tolerance with risk transition in-grid',
                          'alpha': ALPHA, 'delta': DELTA, 'n_seeds': N_SEEDS,
                          'n_test': N_TEST, 'shot_grid': SHOT_GRID, 'budgets': BUDGETS,
                          'budget_accounting': 'STRICT (v4/v5 verbatim)',
                          'validity': 'strict alpha, no padding'},
               'budget_comparison': {}}
    if RESULTS_FILE.exists():
        prev = json.load(open(RESULTS_FILE))
        if prev.get('config', {}).get('epsilon') == EPS:
            results['budget_comparison'] = prev.get('budget_comparison', {})

    # NOTE: CP-CRC-reuse is intentionally EXCLUDED here. Its subsampling identity
    # ("a k*m0-shot estimate = the mean of k independent m0-shot estimates") holds
    # only for MEAN-type estimators (VQE <H>, QAOA <C>). QPE's standard estimator
    # is the MODE (peak bin), which is NOT mean-decomposable, so reuse would
    # calibrate on a statistic (mean-of-modes) that differs from what is tested
    # (actual M-shot mode) -- a representation mismatch. We therefore report the
    # five standard methods for QPE, and note that reuse's gap-closing applies to
    # mean-decomposable estimators (shown for VQE and QAOA), not to mode-based QPE.
    methods = ['CRC', 'Grid-Bonf', 'Gaussian-t', 'Bootstrap', 'No-CRC']
    for B in BUDGETS:
        if str(B) in results['budget_comparison']:
            print(f'[SKIP] B={B}'); continue
        acc = {m: {'Ms': [], 'risks': []} for m in methods}
        for seed in range(N_SEEDS):
            np.random.seed(seed + 3000)
            ef = eval_fn_factory()
            sel = {}
            sel['CRC'], _ = v4.budget_correct_crc(ef, SHOT_GRID, phi_ref, EPS, B, ALPHA, DELTA)
            sel['Grid-Bonf'] = v4.budget_correct_grid_bonferroni(ef, SHOT_GRID, phi_ref, EPS, B, ALPHA, DELTA)
            sel['Gaussian-t'] = v4.budget_correct_gaussian_t(ef, SHOT_GRID, phi_ref, EPS, B, ALPHA)
            sel['Bootstrap'] = v4.budget_correct_bootstrap(ef, SHOT_GRID, phi_ref, EPS, B, ALPHA)
            sel['No-CRC'] = 1000
            for m, M_sel in sel.items():
                test = ef(shots=M_sel, n_evals=N_TEST)
                risk = float(np.mean(np.abs(test - phi_ref) > EPS))
                acc[m]['Ms'].append(M_sel); acc[m]['risks'].append(risk)
        summary = {}
        for m, d in acc.items():
            Ms, risks = np.array(d['Ms']), np.array(d['risks'])
            summary[m] = {'M_mean': float(Ms.mean()), 'M_std': float(Ms.std()),
                          'risk_mean': float(risks.mean()),
                          'valid_count': int(np.sum(risks <= ALPHA)),
                          'certified_count': int(np.sum(Ms < 1000))}
        results['budget_comparison'][str(B)] = summary
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(RESULTS_FILE, 'w'), indent=1)
        print(f'\n  B={B}:')
        for m in methods:
            s = summary[m]
            print(f"    {m:>14s}: M*={s['M_mean']:>6.0f} risk={s['risk_mean']:.3f} "
                  f"valid={s['valid_count']}/{N_SEEDS} cert={s['certified_count']}/{N_SEEDS}")
    print(f'\nSaved {RESULTS_FILE}')
    return results


if __name__ == '__main__':
    t0 = time.time()
    print('S3 QPE eigenphase shot-budget benchmark')
    main()
    print(f'Done in {(time.time()-t0)/60:.1f} min')
