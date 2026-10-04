"""
v5 — The thesis-killing experiment (SaTQuML follow-up).

Question: does CRC certify a shot reduction once we replace the loose Hoeffding
bound with the exact Clopper-Pearson binomial bound AND reuse the purchased
shot pool across levels?

Methods (new):
  1. CP-CRC-reuse   — buy n0 = floor(B/m0) evaluations at base level m0=50
                      (cost n0*m0 <= B, STRICTLY within budget); synthesize a
                      calibration point at level M = k*m0 by averaging k
                      DISJOINT base estimates (exact in distribution: the
                      k*m0-shot estimator is the mean of k independent m0-shot
                      estimators per Pauli group). FST descends from the
                      largest level with n >= n_min(CP) = ceil(log d / log(1-a)) = 22;
                      the test sequence depends only on (B, m0), fixed before
                      any data is seen, so FST validity holds (and holds under
                      the cross-level dependence induced by pool reuse, since
                      fixed-sequence testing is valid under arbitrary dependence).
  2. CP-CRC-equal   — v4's equal-cost-per-level allocation, CP bound instead of
                      Hoeffding. Ablation: isolates bound-tightness from reuse.

Reference rows (CRC-Hoeffding, Gaussian-t, Bootstrap, No-CRC) come from the v4
JSON — not recomputed here.

Reuses v4 infrastructure via importlib (build_molecule, compute_exact_energy,
evaluate_energy, identical seeding: np.random.seed(seed) for training,
np.random.seed(seed+3000) before calibration). Same strict validity scoring:
test risk <= alpha over n_test=200 evaluations at the selected M, no padding.

Outcomes:
  - If CP-CRC-reuse certifies M* ~= 450 at B=10,000 with >= 45/50 validity, the
    SaTQuML paper's "no method achieves both" thesis is DEAD for large budgets
    -> this becomes a methods paper.
  - If it certifies nothing (true risk at 450 above alpha), the thesis is
    bulletproof: even the exact bound + full reuse cannot beat the 22M floor.
  - At B=1000, n0=20 < 22: certification is impossible by the floor; included
    to demonstrate the floor biting.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from scipy import stats as scipy_stats
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
V4_PATH = HERE / 'exp_vqe_crc_v4.py'
RESULTS_FILE = HERE.parent / 'results' / 'vqe_crc_v5_cp.json'

spec = importlib.util.spec_from_file_location('exp_vqe_crc_v4', V4_PATH)
v4 = importlib.util.module_from_spec(spec)
sys.modules['exp_vqe_crc_v4'] = v4
spec.loader.exec_module(v4)

ALPHA = 0.10
DELTA = 0.10
EPSILON = 0.02
M0 = 50                      # base purchase level for the reuse method
M_DEFAULT = 1000             # practitioner default / fallback
N_MIN_CP = int(np.ceil(np.log(DELTA) / np.log(1.0 - ALPHA)))  # = 22 at (0.1, 0.1)


def evaluate_energy_seeded(H, n_q, singles, doubles, hf_state, params,
                           shots, n_evals):
    """v5 override of v4.evaluate_energy: passes an explicit device seed drawn
    from the numpy global stream, so shot noise is reproducible under
    np.random.seed(...). (v4's device used its own entropy; distributionally
    identical, but not bit-reproducible.)"""
    dev = qml.device('default.qubit', wires=n_q, shots=shots,
                     seed=int(np.random.randint(0, 2**31 - 1)))

    @qml.qnode(dev)
    def circuit(p):
        qml.AllSinglesDoubles(p, wires=range(n_q), hf_state=hf_state,
                              singles=singles, doubles=doubles)
        return qml.expval(H)

    return np.array([float(circuit(params)) for _ in range(n_evals)])


def clopper_pearson_ucb(x, n, delta):
    """One-sided (1-delta) upper confidence bound for a binomial proportion."""
    if x >= n:
        return 1.0
    return float(scipy_stats.beta.ppf(1.0 - delta, x + 1, n - x))


def cp_crc_reuse(eval_fn, exact_E, B, alpha=ALPHA, delta=DELTA, m0=M0):
    """Clopper-Pearson CRC with full shot-pool reuse. Returns (M*, diagnostics)."""
    n0 = int(B // m0)
    if n0 < N_MIN_CP:
        return M_DEFAULT, {'status': 'floor', 'n0': n0,
                           'note': f'n0={n0} < {N_MIN_CP}: certification impossible'}
    base = eval_fn(shots=m0, n_evals=n0)          # cost = n0*m0 <= B, all spent here

    # Fixed test sequence (depends only on B, m0): descending k with n(k) >= 22
    k_top = n0 // N_MIN_CP
    ks = list(range(k_top, 0, -1))

    certified = None
    trace = []
    for k in ks:
        n_k = n0 // k
        # disjoint groups of k base estimates -> exact (k*m0)-shot estimates
        est = base[:n_k * k].reshape(n_k, k).mean(axis=1)
        x = int(np.sum(np.abs(est - exact_E) > EPSILON))
        ucb = clopper_pearson_ucb(x, n_k, delta)
        trace.append({'M': k * m0, 'n': n_k, 'failures': x, 'ucb': round(ucb, 4)})
        if ucb <= alpha:
            certified = k * m0
        else:
            break                                  # FST: stop at first failure

    if certified is None:
        return M_DEFAULT, {'status': 'no_pass', 'n0': n0, 'trace': trace}
    return certified, {'status': 'certified', 'n0': n0, 'trace': trace}


def cp_crc_equal_cost(eval_fn, shot_grid, exact_E, B, alpha=ALPHA, delta=DELTA):
    """v4's equal-cost CRC allocation with the CP bound (ablation, no reuse)."""
    K = len(shot_grid)
    cost_per_level = B / K
    certified = None
    for M in sorted(shot_grid, reverse=True):
        n_cal = max(int(cost_per_level / M), 2)    # v4's allocation, incl. its n>=2 floor
        e = eval_fn(shots=M, n_evals=n_cal)
        x = int(np.sum(np.abs(e - exact_E) > EPSILON))
        if clopper_pearson_ucb(x, n_cal, delta) <= alpha:
            certified = M
        else:
            break
    return certified if certified is not None else M_DEFAULT


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return None


def save_results(results):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=1)


def run(molecules, n_seeds=50, n_test=200):
    shot_grid = [5, 10, 20, 50, 100, 200, 500, 1000]
    budgets = [1000, 5000, 10000]

    results = load_results()
    if results is None:
        results = {
            'config': {
                'script': 'exp_vqe_crc_v5_cp.py',
                'n_seeds': n_seeds, 'alpha': ALPHA, 'delta': DELTA,
                'epsilon': EPSILON, 'n_test': n_test, 'm0': M0,
                'n_min_cp': N_MIN_CP, 'shot_grid': shot_grid, 'budgets': budgets,
                'budget_accounting': 'STRICT: CP-CRC-reuse spends n0*m0 <= B total; '
                                     'CP-CRC-equal uses v4 equal-cost allocation',
                'validity': 'strict alpha, no padding, same as v4',
                'seeding': 'train np.random.seed(seed); calib np.random.seed(seed+3000); '
                           'identical to v4',
            },
            'molecules': {},
        }
        print('[FRESH] new results file')
    else:
        print('[RESUME] loaded existing results')

    for mol_name, bl in molecules:
        mol_key = f'{mol_name}_{bl:.3f}'
        mol_data = results['molecules'].setdefault(mol_key, {'budget_comparison': {}})
        if all(str(B) in mol_data['budget_comparison']
               and not mol_data['budget_comparison'][str(B)].get('_partial')
               for B in budgets):
            print(f'[SKIP] {mol_key} complete')
            continue

        H, n_q, elec, sing, doub, hf, n_p = v4.build_molecule(mol_name, bl)
        exact_E, _ = v4.compute_exact_energy(H, n_q, elec, sing, doub, hf, n_p)
        mol_data.update({'molecule': mol_name, 'bond_length': bl, 'n_qubits': n_q,
                         'n_params': n_p, 'exact_energy': exact_E})
        print(f'\n{mol_key}: {n_q} qubits, {n_p} params, E0 = {exact_E:.6f} Ha')

        # AUDIT FINDING: v4's training is zero-init deterministic GD — the
        # per-seed np.random.seed(seed) has NO effect and all 50 "seeds" share
        # identical trained params. Train ONCE and reuse (equivalent, much
        # faster); seed variation enters only via measurement shot noise,
        # exactly as in v4.
        _, shared_params = v4.compute_exact_energy(H, n_q, elec, sing, doub,
                                                   hf, n_p, n_epochs=60, lr=0.4)
        print(f'  trained shared params once (zero-init deterministic GD, '
              f'identical across v4 seeds)')

        # Cross-check ground truth against the v4 JSON if available
        v4_json = V4_PATH.parent.parent / 'results' / 'vqe_crc_v4_camera_ready.json'
        if v4_json.exists():
            with open(v4_json) as f:
                v4_res = json.load(f)
            v4_E = v4_res['molecules'].get(mol_key, {}).get('exact_energy')
            if v4_E is not None and abs(v4_E - exact_E) > 1e-6:
                print(f'  [WARN] exact_E mismatch vs v4 JSON: '
                      f'{exact_E:.8f} vs {v4_E:.8f} — using v4 value')
                exact_E = v4_E
                mol_data['exact_energy'] = exact_E

        for B in budgets:
            bkey = str(B)
            if bkey in mol_data['budget_comparison'] and \
                    not mol_data['budget_comparison'][bkey].get('_partial'):
                print(f'  [SKIP] B={B}')
                continue
            print(f'\n  Budget B={B} (n0={B // M0}, feasible top level = '
                  f'{(B // M0) // N_MIN_CP * M0})')

            acc = {m: {'Ms': [], 'risks': []} for m in ('CP-CRC-reuse', 'CP-CRC-equal')}
            diags = []
            for seed in tqdm(range(n_seeds), desc=f'  B={B} seeds', ncols=80):
                np.random.seed(seed + 3000)
                params = shared_params

                def eval_fn(shots, n_evals):
                    return evaluate_energy_seeded(H, n_q, sing, doub, hf,
                                                  params, shots=shots,
                                                  n_evals=n_evals)

                M_reuse, diag = cp_crc_reuse(eval_fn, exact_E, B)
                M_equal = cp_crc_equal_cost(eval_fn, shot_grid, exact_E, B)
                diags.append(diag)

                for method, M_sel in (('CP-CRC-reuse', M_reuse),
                                      ('CP-CRC-equal', M_equal)):
                    test_e = eval_fn(shots=M_sel, n_evals=n_test)
                    risk = float(np.mean(np.abs(test_e - exact_E) > EPSILON))
                    acc[method]['Ms'].append(M_sel)
                    acc[method]['risks'].append(risk)

                if (seed + 1) % 10 == 0:
                    mol_data['budget_comparison'][bkey] = _summarize(
                        acc, diags, partial=True)
                    results['molecules'][mol_key] = mol_data
                    save_results(results)

            mol_data['budget_comparison'][bkey] = _summarize(acc, diags, partial=False)
            results['molecules'][mol_key] = mol_data
            save_results(results)
            for m, s in mol_data['budget_comparison'][bkey].items():
                if m.startswith('_'):
                    continue
                print(f"    {m:>14s}: M*={s['M_mean']:.0f}±{s['M_std']:.0f}  "
                      f"risk={s['risk_mean']:.3f}  valid={s['valid_count']}/{n_seeds}  "
                      f"certified={s['certified_count']}/{n_seeds}")

    save_results(results)
    return results


def _summarize(acc, diags, partial):
    out = {}
    for method, d in acc.items():
        Ms, risks = np.array(d['Ms']), np.array(d['risks'])
        out[method] = {
            'M_mean': float(Ms.mean()), 'M_std': float(Ms.std()),
            'risk_mean': float(risks.mean()), 'risk_std': float(risks.std()),
            'valid_count': int(np.sum(risks <= ALPHA)),
            'certified_count': int(np.sum(Ms < M_DEFAULT)),
            '_seeds_completed': len(Ms),
        }
        if partial:
            out[method]['_partial'] = True
    out['_reuse_diagnostics'] = diags[:5]  # first seeds' FST traces, for inspection
    if partial:
        out['_partial'] = True
    return out


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'h2'
    mols = {'h2': [('H2', 0.5), ('H2', 0.735), ('H2', 1.0), ('H2', 1.5)],
            'lih': [('LiH', 1.6)]}[phase]
    t0 = time.time()
    print(f'v5 CP-CRC experiment — phase: {phase}')
    print(f'n_min(CP) at (a,d)=({ALPHA},{DELTA}) = {N_MIN_CP}')
    run(mols)
    print(f'\nDone in {(time.time() - t0) / 60:.1f} min')
