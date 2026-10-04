"""
S1 — Sensitivity of the efficiency-validity tradeoff to (alpha, epsilon, budget).

Reviewer ywhS #2: "a broader sensitivity analysis to show whether the observed
tradeoff remains consistent under different values of the risk threshold, error
tolerance, and candidate shot budgets."

We sweep:
  alpha = delta in {0.05, 0.10, 0.20}   (risk threshold / confidence)
  epsilon in {0.01, 0.02, 0.04} Ha      (error tolerance)
  budget  in {1000, 5000, 10000}        (tight to loose; brackets the tradeoff
                                         regimes from the main paper)
and report, per cell, the two representative methods from the main benchmark:
  CRC         (distribution-free Hoeffding-UCB; always-valid-never-efficient)
  Gaussian-t  (parametric; efficient-but-invalid-at-tight-budget)

Everything reuses the v4 camera-ready infrastructure VERBATIM via importlib:
build_molecule, compute_exact_energy, and the four budget-correct method
functions (budget_correct_crc, budget_correct_gaussian_t). Strict budget
accounting and strict (no-padding) validity, identical to v4/v5.

AUDIT NOTES (self):
  - Training is zero-init deterministic GD (v4), so np.random.seed(seed) does NOT
    change trained params; all 50 seeds share identical params and vary only via
    measurement shot noise. We therefore train ONCE per molecule (the v5 fix) and
    reuse. This is EXACTLY v4's effective behavior, just without the 50x waste.
  - Seeding matches v4: calibration/test draws use np.random.seed(seed+3000).
  - epsilon enters ONLY the loss threshold |E_M - E0| > epsilon; E0 is unchanged.
  - alpha enters the UCB acceptance test and the t critical value; delta=alpha.
  - We report CRC and Gaussian-t only (the valid/invalid representatives); the
    full 5-method table is the main paper's job. One molecule (H2 @ d=0.735,
    equilibrium) keeps the sweep fast (4 qubits); the main paper already shows
    the pattern is molecule- and system-size-robust (4 H2 bonds + LiH 12q).
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4_PATH = HERE / 'exp_vqe_crc_v4.py'
RESULTS_FILE = HERE.parent / 'results' / 'vqe_crc_sensitivity.json'

spec = importlib.util.spec_from_file_location('exp_vqe_crc_v4', V4_PATH)
v4 = importlib.util.module_from_spec(spec)
sys.modules['exp_vqe_crc_v4'] = v4
spec.loader.exec_module(v4)

# FULL sensitivity grid: every (alpha, eps, budget) combination. alpha=delta.
BASE_ALPHA, BASE_EPS = 0.10, 0.02
ALPHAS = [0.05, 0.10, 0.20]       # alpha = delta
EPSILONS = [0.01, 0.02, 0.04]     # Ha
BUDGETS = [1000, 5000, 10000]
SHOT_GRID = [5, 10, 20, 50, 100, 200, 500, 1000]
N_SEEDS = 50
N_TEST = 200


def sweep_cells():
    """Full grid: every (alpha, eps, budget) combination (3 x 3 x 3 = 27 cells)."""
    cells = []
    for a in ALPHAS:
        for e in EPSILONS:
            for B in BUDGETS:
                cells.append((a, e, B))
    return cells


def evaluate_energy_seeded(H, n_q, sing, doub, hf, params, shots, n_evals):
    """Seeded eval, matching v5 (reproducible shot noise under np.random.seed)."""
    import pennylane as qml
    dev = qml.device('default.qubit', wires=n_q, shots=shots,
                     seed=int(np.random.randint(0, 2**31 - 1)))

    @qml.qnode(dev)
    def circuit(p):
        qml.AllSinglesDoubles(p, wires=range(n_q), hf_state=hf,
                              singles=sing, doubles=doub)
        return qml.expval(H)

    return np.array([float(circuit(params)) for _ in range(n_evals)])


def run():
    mol_name, bl = 'H2', 0.735
    H, n_q, elec, sing, doub, hf, n_p = v4.build_molecule(mol_name, bl)
    # Match v4 EXACTLY: E0 reference from the 150-epoch noiseless optimum (v4
    # default), deployed params from 60-epoch GD (v4's per-seed training). The
    # small train/reference gap is shared across all methods, as in the paper.
    exact_E, _ = v4.compute_exact_energy(H, n_q, elec, sing, doub, hf, n_p)
    _, shared_params = v4.compute_exact_energy(
        H, n_q, elec, sing, doub, hf, n_p, n_epochs=60, lr=0.4)
    # Safety cross-check against the v4 JSON reference (should now match tightly)
    v4_json = V4_PATH.parent.parent / 'results' / 'vqe_crc_v4_camera_ready.json'
    if v4_json.exists():
        v4E = json.load(open(v4_json))['molecules'].get(
            f'{mol_name}_{bl:.3f}', {}).get('exact_energy')
        if v4E is not None:
            assert abs(v4E - exact_E) < 1e-5, \
                f'E0 mismatch vs v4: {exact_E:.8f} vs {v4E:.8f}'
            print(f'[OK] E0 matches v4 reference within 1e-5')
    print(f'{mol_name} d={bl}: {n_q} qubits, E0={exact_E:.6f} Ha')

    cfg = {
        'script': 'exp_vqe_crc_sensitivity.py',
        'molecule': f'{mol_name}_{bl}', 'n_qubits': n_q,
        'exact_energy': exact_E, 'n_seeds': N_SEEDS, 'n_test': N_TEST,
        'alphas': ALPHAS, 'epsilons': EPSILONS, 'budgets': BUDGETS,
        'shot_grid': SHOT_GRID, 'delta': 'equal to alpha',
        'budget_accounting': 'STRICT (v4 functions verbatim)',
        'validity': 'strict alpha, no padding',
        'methods': ['CRC', 'Gaussian-t'],
    }
    # Resume: keep completed cells from a prior run with the SAME config.
    results = {'config': cfg, 'cells': {}}
    if RESULTS_FILE.exists():
        prev = json.load(open(RESULTS_FILE))
        if prev.get('config', {}).get('n_test') == N_TEST and \
                prev.get('config', {}).get('n_seeds') == N_SEEDS:
            results['cells'] = prev.get('cells', {})
            print(f'[RESUME] kept {len(results["cells"])} completed cells')

    for (alpha, eps, B) in sweep_cells():
            delta = alpha
            key = f'a{alpha}_e{eps}_B{B}'
            if key in results['cells']:
                print(f'  [SKIP] {key} (done)')
            else:
                _k = key  # keep indentation block below
                acc = {'CRC': {'Ms': [], 'risks': []},
                       'Gaussian-t': {'Ms': [], 'risks': []}}
                for seed in range(N_SEEDS):
                    np.random.seed(seed + 3000)
                    params = shared_params

                    def eval_fn(shots, n_evals):
                        return evaluate_energy_seeded(H, n_q, sing, doub, hf,
                                                      params, shots, n_evals)

                    M_crc, _ = v4.budget_correct_crc(
                        eval_fn, SHOT_GRID, exact_E, eps, B, alpha, delta)
                    M_gauss = v4.budget_correct_gaussian_t(
                        eval_fn, SHOT_GRID, exact_E, eps, B, alpha)

                    for method, M_sel in (('CRC', M_crc), ('Gaussian-t', M_gauss)):
                        test_e = eval_fn(shots=M_sel, n_evals=N_TEST)
                        risk = float(np.mean(np.abs(test_e - exact_E) > eps))
                        acc[method]['Ms'].append(M_sel)
                        acc[method]['risks'].append(risk)

                cell = {}
                for method, d in acc.items():
                    Ms, risks = np.array(d['Ms']), np.array(d['risks'])
                    cell[method] = {
                        'M_mean': float(Ms.mean()), 'M_std': float(Ms.std()),
                        'risk_mean': float(risks.mean()),
                        'valid_count': int(np.sum(risks <= alpha)),
                        'certified_count': int(np.sum(Ms < 1000)),
                    }
                results['cells'][key] = cell
                print(f'  {key}: '
                      f"CRC M*={cell['CRC']['M_mean']:.0f} "
                      f"val={cell['CRC']['valid_count']}/{N_SEEDS} | "
                      f"Gauss M*={cell['Gaussian-t']['M_mean']:.0f} "
                      f"val={cell['Gaussian-t']['valid_count']}/{N_SEEDS}")
                RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
                json.dump(results, open(RESULTS_FILE, 'w'), indent=1)

    print(f'\nSaved {RESULTS_FILE}')
    return results


if __name__ == '__main__':
    t0 = time.time()
    print('S1 sensitivity sweep (alpha x epsilon x budget), H2 d=0.735')
    run()
    print(f'Done in {(time.time() - t0) / 60:.1f} min')
