"""
S2 — Does the efficiency-validity tradeoff generalize beyond VQE? QAOA MaxCut.

Reviewer ywhS #1: "additional experiments on other representative quantum
applications, such as QAOA, QFT, or phase estimation, to better understand how
well the observed efficiency-validity tradeoff generalizes beyond VQE."

We port the exact shot-budget-selection benchmark to QAOA for MaxCut:
  - Problem: MaxCut on a fixed 4-node graph (ring + one chord, 5 edges),
    cut operator  C = sum_{(i,j) in E} 0.5 (1 - Z_i Z_j),  so <C> in [0, |E|].
  - QAOA ansatz (p=1) optimized on the noiseless simulator to (gamma*, beta*);
    C0 = <C>(gamma*, beta*) is the noiseless reference ("ground truth" analog
    of E0). The certification question is identical: how many shots M before
    |C_M - C0| <= epsilon with risk <= alpha, charged M executions per eval.

Methods: the SAME ones as the VQE benchmark, reused VERBATIM via importlib:
  v4.budget_correct_crc / grid_bonferroni / gaussian_t / bootstrap  (+ No-control)
  v5.cp_crc_reuse / cp_crc_equal_cost                                (stronger impl.)
Strict budget accounting and strict (no-padding) validity, identical to VQE.

If the same tradeoff appears (distribution-free valid-but-not-efficient; parametric
efficient-but-invalid-at-tight-budgets; CP+reuse closes it above the 22M floor),
the VQE finding is not VQE-specific.

AUDIT NOTES (self):
  - QAOA angles are optimized by deterministic zero-init GD on the analytic
    simulator; like the VQE case, randomness enters ONLY via measurement shot
    noise. We optimize ONCE and reuse the angles across all trials (seeds vary
    the shot noise), matching the VQE setup.
  - C0 is the noiseless <C> at the optimized angles (analog of the VQE 150-epoch
    reference). The loss threshold epsilon is applied to |C_M - C0|.
  - epsilon is fixed (not tuned per trial); chosen so the pass/fail transition
    lies inside the shot grid, exactly the criterion used for VQE (epsilon ~ 10x
    the M=1000 shot-noise std). We PRINT the measured shot-noise std so the
    choice is auditable and disclose it in the paper.
  - v5.cp_crc_reuse reads epsilon from the v5 MODULE global EPSILON; we override
    v5.EPSILON = EPS so the stronger method uses the SAME tolerance as the rest
    (override-module pattern, not a forked copy).
  - Seeding matches VQE: np.random.seed(seed+3000) before each trial's draws.
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
RESULTS_FILE = HERE.parent / 'results' / 'qaoa_crc.json'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v4 = _load('exp_vqe_crc_v4', V4_PATH)
v5 = _load('exp_vqe_crc_v5_cp', V5_PATH)

# ---- problem definition -------------------------------------------------
N_NODES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]   # ring + one chord, 5 edges
P_LAYERS = 1
ALPHA = 0.10
DELTA = 0.10
SHOT_GRID = [5, 10, 20, 50, 100, 200, 500, 1000]
BUDGETS = [1000, 5000, 10000]
N_SEEDS = 50
N_TEST = 200
# epsilon is set after measuring the shot-noise scale (see main); placeholder:
EPS = None


def cut_hamiltonian():
    """C = sum_{(i,j)} 0.5 (1 - Z_i Z_j): the MaxCut value operator."""
    coeffs, ops = [], []
    for (i, j) in EDGES:
        coeffs.append(0.5)
        ops.append(qml.Identity(0))
        coeffs.append(-0.5)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)


C_H = cut_hamiltonian()


def qaoa_layer(gamma, beta):
    for (i, j) in EDGES:
        qml.CNOT(wires=[i, j])
        qml.RZ(2.0 * gamma, wires=j)
        qml.CNOT(wires=[i, j])
    for w in range(N_NODES):
        qml.RX(2.0 * beta, wires=w)


def make_circuit(dev):
    @qml.qnode(dev)
    def circuit(params):
        for w in range(N_NODES):
            qml.Hadamard(wires=w)
        for l in range(P_LAYERS):
            qaoa_layer(params[l], params[P_LAYERS + l])
        return qml.expval(C_H)
    return circuit


def optimize_qaoa():
    """Deterministic zero-init GD on the analytic simulator. Returns (angles, C0).
    Zero init is a saddle for QAOA, so we use a small FIXED (non-random) offset,
    identical across all trials -> no seed dependence, matching the VQE setup."""
    from pennylane import numpy as pnp
    dev = qml.device('default.qubit', wires=N_NODES)
    circuit = make_circuit(dev)

    def neg_cut(p):
        return -circuit(p)

    # fixed deterministic init (not random): small nonzero to break the H-saddle
    params = pnp.array([0.1] * P_LAYERS + [0.2] * P_LAYERS, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.3)
    for _ in range(200):
        params, _ = opt.step_and_cost(neg_cut, params)
    C0 = float(circuit(params))
    return np.array(params.tolist()), C0


def eval_fn_factory(angles):
    def eval_fn(shots, n_evals):
        dev = qml.device('default.qubit', wires=N_NODES, shots=shots,
                         seed=int(np.random.randint(0, 2**31 - 1)))
        circuit = make_circuit(dev)
        return np.array([float(circuit(angles)) for _ in range(n_evals)])
    return eval_fn


def main():
    global EPS
    angles, C0 = optimize_qaoa()
    # brute-force true MaxCut for context
    best = 0
    for b in range(2 ** N_NODES):
        bits = [(b >> k) & 1 for k in range(N_NODES)]
        cut = sum(1 for (i, j) in EDGES if bits[i] != bits[j])
        best = max(best, cut)
    print(f'QAOA p={P_LAYERS} on {N_NODES} nodes, {len(EDGES)} edges: '
          f'C0(noiseless)={C0:.4f}, true MaxCut={best}')

    # measure shot-noise std at M=1000 to set epsilon (disclosed, not per-trial tuned)
    probe = eval_fn_factory(angles)
    np.random.seed(12345)
    s1000 = probe(shots=1000, n_evals=200).std()
    s_grid = {M: probe(shots=M, n_evals=200).std() for M in SHOT_GRID}
    # Set epsilon to the shot-noise scale at a MID-GRID shot count (M=100), so the
    # pass/fail risk transition lies INSIDE the shot grid, which is the exact
    # criterion the VQE paper states for its epsilon. The QAOA problem is
    # bias-free (deployed angles = reference angles), so unlike the VQE 10x-of-
    # M1000-noise coincidence, we must anchor epsilon to a mid-grid noise scale
    # or the transition collapses to the bottom of the grid and the task is trivial.
    EPS = round(float(s_grid[100]), 2)
    print(f'shot-noise std at M=1000: {s1000:.4f}; at M=100: {s_grid[100]:.4f}; '
          f'chosen epsilon={EPS} (mid-grid-transition criterion). per-M std: ' +
          ', '.join(f'{M}:{v:.3f}' for M, v in s_grid.items()))
    v5.EPSILON = EPS          # make the stronger method use the SAME tolerance

    results = {
        'config': {
            'script': 'exp_qaoa_crc.py', 'problem': 'QAOA MaxCut',
            'n_nodes': N_NODES, 'edges': EDGES, 'p_layers': P_LAYERS,
            'C0_noiseless': C0, 'true_maxcut': best,
            'epsilon': EPS, 'epsilon_rule': 'shot-noise std at mid-grid M=100 (transition-in-grid criterion)',
            'shotnoise_std_M1000': float(s1000), 'shotnoise_std_M100': float(s_grid[100]),
            'alpha': ALPHA, 'delta': DELTA, 'n_seeds': N_SEEDS, 'n_test': N_TEST,
            'shot_grid': SHOT_GRID, 'budgets': BUDGETS,
            'budget_accounting': 'STRICT (v4/v5 functions verbatim)',
            'validity': 'strict alpha, no padding',
        },
        'budget_comparison': {},
    }

    methods = ['CRC', 'Grid-Bonf', 'Gaussian-t', 'Bootstrap', 'No-CRC',
               'CP-CRC-reuse', 'CP-CRC-equal']
    for B in BUDGETS:
        acc = {m: {'Ms': [], 'risks': []} for m in methods}
        for seed in range(N_SEEDS):
            np.random.seed(seed + 3000)
            eval_fn = eval_fn_factory(angles)

            sel = {}
            sel['CRC'], _ = v4.budget_correct_crc(eval_fn, SHOT_GRID, C0, EPS, B, ALPHA, DELTA)
            sel['Grid-Bonf'] = v4.budget_correct_grid_bonferroni(eval_fn, SHOT_GRID, C0, EPS, B, ALPHA, DELTA)
            sel['Gaussian-t'] = v4.budget_correct_gaussian_t(eval_fn, SHOT_GRID, C0, EPS, B, ALPHA)
            sel['Bootstrap'] = v4.budget_correct_bootstrap(eval_fn, SHOT_GRID, C0, EPS, B, ALPHA)
            sel['No-CRC'] = 1000
            sel['CP-CRC-reuse'], _ = v5.cp_crc_reuse(eval_fn, C0, B, ALPHA, DELTA)
            sel['CP-CRC-equal'] = v5.cp_crc_equal_cost(eval_fn, SHOT_GRID, C0, B, ALPHA, DELTA)

            for m, M_sel in sel.items():
                test = eval_fn(shots=M_sel, n_evals=N_TEST)
                risk = float(np.mean(np.abs(test - C0) > EPS))
                acc[m]['Ms'].append(M_sel)
                acc[m]['risks'].append(risk)

        summary = {}
        for m, d in acc.items():
            Ms, risks = np.array(d['Ms']), np.array(d['risks'])
            summary[m] = {
                'M_mean': float(Ms.mean()), 'M_std': float(Ms.std()),
                'risk_mean': float(risks.mean()),
                'valid_count': int(np.sum(risks <= ALPHA)),
                'certified_count': int(np.sum(Ms < 1000)),
            }
        results['budget_comparison'][str(B)] = summary
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(RESULTS_FILE, 'w'), indent=1)
        print(f'\n  B={B}:')
        for m in methods:
            s = summary[m]
            print(f"    {m:>14s}: M*={s['M_mean']:>6.0f}  "
                  f"risk={s['risk_mean']:.3f}  valid={s['valid_count']}/{N_SEEDS}  "
                  f"cert={s['certified_count']}/{N_SEEDS}")

    print(f'\nSaved {RESULTS_FILE}')
    return results


if __name__ == '__main__':
    t0 = time.time()
    print('S2 QAOA MaxCut shot-budget benchmark')
    main()
    print(f'Done in {(time.time() - t0) / 60:.1f} min')
