"""
S4 — Adaptive allocation: the third stronger-implementation reviewer cBGk named
("tighter binomial bounds, adaptive allocation, and cross-level reuse").

We already benchmark the exact Clopper-Pearson bound (tighter bound) and shot-pool
reuse (v5). This adds a rigorously-valid ADAPTIVE allocation and compares all three
on the SAME H2 config (4 bond lengths x 3 budgets x 50 trials) so a row can drop
straight into Table 'closing'.

CP-CRC-adaptive (two-stage, data-split -> valid):
  Stage 1 (pilot, B_pilot = 0.3 B): cheap scan of the grid to POINT-ESTIMATE the
    smallest shot level whose risk looks <= alpha. Uses equal-cost pilot evals;
    this only PICKS a level, it does not certify.
  Stage 2 (certify, B_cert = 0.7 B): spend the ENTIRE remaining budget at the
    single picked level M_hat -> n_c = B_cert / M_hat evaluations, and apply the
    exact Clopper-Pearson UCB. Certify M_hat iff UCB <= alpha, else fall back to
    the default M=1000.
  Validity: Stage-2 data is drawn FRESH and independently of the Stage-1 pilot
    (sequential eval_fn draws), so M_hat is independent of the Stage-2 sample and
    the CP guarantee at M_hat holds conditionally (standard data splitting). The
    pilot spends real budget and is charged for.

Reuses v4 (build_molecule, compute_exact_energy) and v5 (clopper_pearson_ucb,
N_MIN_CP) via importlib. Strict budget accounting, strict (no-padding) validity,
seeding (seed+3000) identical to v4/v5. Incremental save per (molecule,budget).

AUDIT NOTES (self):
  - E0 = 150-epoch noiseless reference; deployed params = 60-epoch (v4 setup),
    cross-checked against v4 JSON. Same as v5.
  - epsilon=0.02, alpha=delta=0.10, m0 grid identical to v5.
  - Budget charged: pilot cost sum(n_p*M) + cert cost n_c*M_hat <= B (asserted).
  - certified = M_hat < 1000 AND Stage-2 CP passed.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4_PATH = HERE / 'exp_vqe_crc_v4.py'
V5_PATH = HERE / 'exp_vqe_crc_v5_cp.py'
RESULTS_FILE = HERE.parent / 'results' / 'cp_adaptive.json'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v4 = _load('exp_vqe_crc_v4', V4_PATH)
v5 = _load('exp_vqe_crc_v5_cp', V5_PATH)

ALPHA, DELTA, EPS = 0.10, 0.10, 0.02
M0 = 50
M_DEFAULT = 1000
N_MIN_CP = v5.N_MIN_CP          # 22
SHOT_GRID = [5, 10, 20, 50, 100, 200, 500, 1000]
BUDGETS = [1000, 5000, 10000]
MOLS = [('H2', 0.5), ('H2', 0.735), ('H2', 1.0), ('H2', 1.5)]
N_SEEDS = 50
N_TEST = 200
PILOT_FRAC = 0.30


def eval_seeded(H, n_q, sing, doub, hf, params, shots, n_evals):
    import pennylane as qml
    dev = qml.device('default.qubit', wires=n_q, shots=shots,
                     seed=int(np.random.randint(0, 2**31 - 1)))

    @qml.qnode(dev)
    def circ(p):
        qml.AllSinglesDoubles(p, wires=range(n_q), hf_state=hf,
                              singles=sing, doubles=doub)
        return qml.expval(H)
    return np.array([float(circ(params)) for _ in range(n_evals)])


def cp_crc_adaptive(eval_fn, exact_E, B, alpha=ALPHA, delta=DELTA, eps=EPS,
                    grid=SHOT_GRID, pilot_frac=PILOT_FRAC):
    B_pilot = int(pilot_frac * B)
    B_cert = B - B_pilot
    K = len(grid)
    spent = 0

    # Stage 1: pilot point-estimate of the smallest safe level. Equal-cost per
    # level, HARD-CAPPED so total pilot spend <= B_pilot: n_p = floor((B_pilot/K)/M);
    # if n_p == 0 the level is unaffordable at pilot resolution and is skipped
    # (these are the expensive high-M levels we don't need to pick anyway, since
    # we want the SMALLEST safe level). Sum_M n_p*M <= K*(B_pilot/K) = B_pilot.
    per_level = B_pilot / K
    M_hat = None
    for M in sorted(grid):                       # ascending: smallest safe level first
        n_p = int(per_level / M)
        if n_p < 1:
            continue
        est = eval_fn(shots=M, n_evals=n_p); spent += n_p * M
        if np.mean(np.abs(est - exact_E) > eps) <= alpha:
            M_hat = M
            break
    if M_hat is None:
        return M_DEFAULT, {'status': 'pilot_none', 'spent': spent}

    # Stage 2: certify M_hat with the whole remaining budget (fresh data)
    n_c = int(B_cert / M_hat)
    if n_c < N_MIN_CP:
        return M_DEFAULT, {'status': 'cert_floor', 'M_hat': M_hat, 'n_c': n_c,
                           'spent': spent}
    est_c = eval_fn(shots=M_hat, n_evals=n_c); spent += n_c * M_hat
    x = int(np.sum(np.abs(est_c - exact_E) > eps))
    ucb = v5.clopper_pearson_ucb(x, n_c, delta)
    assert spent <= B, f'budget overrun: {spent} > {B}'          # strict budget honesty
    if ucb <= alpha:
        return M_hat, {'status': 'certified', 'M_hat': M_hat, 'n_c': n_c,
                       'ucb': round(ucb, 4), 'spent': spent}
    return M_DEFAULT, {'status': 'cert_fail', 'M_hat': M_hat, 'n_c': n_c,
                       'ucb': round(ucb, 4), 'spent': spent}


def run():
    results = {'config': {'script': 'exp_cp_adaptive.py', 'method': 'CP-CRC-adaptive',
                          'alpha': ALPHA, 'delta': DELTA, 'epsilon': EPS, 'm0': M0,
                          'pilot_frac': PILOT_FRAC, 'n_min_cp': N_MIN_CP,
                          'shot_grid': SHOT_GRID, 'budgets': BUDGETS,
                          'n_seeds': N_SEEDS, 'n_test': N_TEST,
                          'budget_accounting': 'STRICT: pilot+cert <= B',
                          'validity': 'strict alpha, no padding; data-split'},
               'molecules': {}}
    if RESULTS_FILE.exists():
        prev = json.load(open(RESULTS_FILE))
        if prev.get('config', {}).get('n_test') == N_TEST:
            results['molecules'] = prev.get('molecules', {})
            print(f'[RESUME] {sum(len(v) for v in results["molecules"].values())} cells kept')

    for name, bl in MOLS:
        mk = f'{name}_{bl:.3f}'
        H, n_q, elec, sing, doub, hf, n_p = v4.build_molecule(name, bl)
        exact_E, _ = v4.compute_exact_energy(H, n_q, elec, sing, doub, hf, n_p)
        _, params = v4.compute_exact_energy(H, n_q, elec, sing, doub, hf, n_p,
                                            n_epochs=60, lr=0.4)
        v4E = json.load(open(V4_PATH.parent.parent / 'results' /
                             'vqe_crc_v4_camera_ready.json'))['molecules'][mk]['exact_energy']
        assert abs(v4E - exact_E) < 1e-5, f'{mk} E0 mismatch {exact_E} vs {v4E}'
        results['molecules'].setdefault(mk, {})
        print(f'{mk}: {n_q}q E0={exact_E:.5f}')

        for B in BUDGETS:
            if str(B) in results['molecules'][mk]:
                print(f'  [SKIP] B={B}'); continue
            Ms, risks = [], []
            for seed in range(N_SEEDS):
                np.random.seed(seed + 3000)

                def eval_fn(shots, n_evals):
                    return eval_seeded(H, n_q, sing, doub, hf, params, shots, n_evals)

                M_sel, _ = cp_crc_adaptive(eval_fn, exact_E, B)
                test = eval_fn(shots=M_sel, n_evals=N_TEST)
                Ms.append(M_sel)
                risks.append(float(np.mean(np.abs(test - exact_E) > EPS)))
            Ms, risks = np.array(Ms), np.array(risks)
            results['molecules'][mk][str(B)] = {
                'M_mean': float(Ms.mean()), 'M_std': float(Ms.std()),
                'risk_mean': float(risks.mean()),
                'valid_count': int(np.sum(risks <= ALPHA)),
                'certified_count': int(np.sum(Ms < M_DEFAULT))}
            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            json.dump(results, open(RESULTS_FILE, 'w'), indent=1)
            s = results['molecules'][mk][str(B)]
            print(f"  B={B}: M*={s['M_mean']:.0f} valid={s['valid_count']}/{N_SEEDS} "
                  f"cert={s['certified_count']}/{N_SEEDS}")
    print(f'Saved {RESULTS_FILE}')
    return results


if __name__ == '__main__':
    t0 = time.time()
    print('S4 CP-CRC-adaptive (two-stage data-split allocation)')
    run()
    print(f'Done in {(time.time()-t0)/60:.1f} min')
