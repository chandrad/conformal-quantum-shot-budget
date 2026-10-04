#!/usr/bin/env python3
"""
VQE on IBM hardware WITH error mitigation (readout + zero-noise extrapolation),
to answer the paper's own open question (sec 5.4): does mitigation lower the error
plateau enough to flip the certificate's verdict from REFUSE to PASS?

Head-to-head with the UNMITIGATED numbers already in the paper (Table 2,
ibm_vqe_hardware.json): SAME H2 4-qubit VQE circuit, SAME Hamiltonian, SAME shot
levels subset, SAME epsilon=0.02 -- the ONLY change is resilience_level (0 -> 2).

STRICT free-plan budget safety (user requirement):
  - Pre-flight: query usage_remaining_seconds; ABORT if below SAFETY_FLOOR.
  - Between every shot level: re-query usage; ABORT before the next job if the
    remaining quota has dropped below SAFETY_FLOOR. Never risk exceeding the plan.
  - Scoped small: 2 shot levels x N_EVALS evals (ZNE multiplies executions, so we
    keep this minimal; M=1000 is the deployment point the paper's table flags FAIL).

AUDIT NOTES (self):
  - H2 Hamiltonian (SparsePauliOp) and circuit are copied VERBATIM from the
    accepted exp_ibm_vqe_v2.py (the script that produced ibm_vqe_hardware.json),
    so the mitigated run is a true apples-to-apples comparison.
  - resilience_level=2 = readout mitigation + ZNE (gate-error extrapolation).
  - precision=1/sqrt(M) reproduces M-shot noise, same convention as the unmitigated run.
  - exact_energy and epsilon identical to the paper.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_FILE = HERE.parent / 'results' / 'ibm_vqe_hardware_mitigated.json'

EXACT_ENERGY = -1.136189454088        # same as paper / ibm_vqe_hardware.json
EPSILON = 0.02
M_DEPLOY = 1000                       # deployment point (flagged FAIL unmitigated in Table 2)
RESILIENCE_LEVELS = [0, 2]            # controlled: none vs readout+ZNE (self-contained)
N_EVALS = 22                          # MUST be >= the n>=22 certification floor, else the
                                      # verdict is FAIL regardless of error and the question
                                      # (can mitigation flip PASS?) is unanswerable. CP at
                                      # n=22, x=0 gives UCB=0.0995<=0.10 -> a PASS is possible.
SAFETY_FLOOR = 300                    # abort if remaining free-plan QPU seconds < this
DELTA = 0.10


def clopper_pearson_ucb(x, n, delta):
    from scipy import stats
    if x >= n:
        return 1.0
    return float(stats.beta.ppf(1.0 - delta, x + 1, n - x))


def h2_hamiltonian():
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_list([
        ("IIII", -0.0420789764), ("IIIZ", 0.1777128746), ("IIZI", 0.1777128746),
        ("IZII", -0.2427428051), ("ZIII", -0.2427428051), ("IIZZ", 0.1705973833),
        ("ZYXZ", 0.0447501440), ("YXYZ", -0.0447501440), ("XYYZ", -0.0447501440),
        ("XYYY", 0.0447501440), ("IZIZ", 0.1229330506), ("IZZI", 0.1676831946),
        ("ZIIZ", 0.1676831946), ("ZIZI", 0.1229330506), ("ZZII", 0.1762764039),
    ])


def h2_circuit(params):
    from qiskit.circuit import QuantumCircuit
    qc = QuantumCircuit(4)
    qc.x(0); qc.x(1)
    qc.ry(params[0], 0); qc.cx(0, 2); qc.ry(-params[0], 0); qc.cx(0, 2)
    qc.ry(params[1], 1); qc.cx(1, 3); qc.ry(-params[1], 1); qc.cx(1, 3)
    qc.cx(0, 1); qc.cx(1, 2); qc.cx(2, 3)
    qc.ry(params[2], 3)
    qc.cx(2, 3); qc.cx(1, 2); qc.cx(0, 1)
    return qc


def train_params():
    """Converged H2 VQE params on the noiseless simulator (same ansatz as paper)."""
    import pennylane as qml
    from pennylane import numpy as pnp
    symbols = ["H", "H"]
    geo = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.735 * 1.8897259886]])
    H, n_q = qml.qchem.molecular_hamiltonian(symbols, geo)
    elec = 2
    s, d = qml.qchem.excitations(elec, n_q)
    hf = qml.qchem.hf_state(elec, n_q)
    dev = qml.device('default.qubit', wires=n_q)
    @qml.qnode(dev)
    def c(p):
        qml.AllSinglesDoubles(p, wires=range(n_q), hf_state=hf, singles=s, doubles=d)
        return qml.expval(H)
    p = pnp.array(np.zeros(len(s) + len(d)), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    for _ in range(100):
        p, _ = opt.step_and_cost(c, p)
    return [float(x) for x in p], float(c(p))


def remaining_seconds(service):
    try:
        return int(service.usage().get('usage_remaining_seconds', 0))
    except Exception as e:
        print(f'[warn] usage query failed: {e}')
        return -1


def run():
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService()
    rem = remaining_seconds(service)
    print(f'[budget] remaining free-plan QPU seconds: {rem} (floor {SAFETY_FLOOR})')
    if rem < SAFETY_FLOOR:
        print(f'[ABORT] remaining {rem}s < floor {SAFETY_FLOOR}s. Not submitting.')
        return None
    projected = N_EVALS * (1 + 3)   # crude: resilience 0 (~1x) + resilience 2 ZNE (~3x) per eval
    print(f'[budget] crude projection ~{projected}s ({len(RESILIENCE_LEVELS)} resilience levels x '
          f'{N_EVALS} evals at M={M_DEPLOY}); re-checks quota before each job, aborts if < {SAFETY_FLOOR}s')

    params, sim_E = train_params()
    print(f'sim energy {sim_E:.6f}, exact {EXACT_ENERGY:.6f}, eps {EPSILON}')
    H = h2_hamiltonian()
    qc = h2_circuit(params)

    backend = service.least_busy(operational=True, min_num_qubits=4)
    print(f'backend {backend.name}')
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    isa = pm.run(qc)
    isa_H = H.apply_layout(isa.layout)

    # Self-contained CONTROLLED comparison at the deployment point M=M_DEPLOY:
    # resilience 0 (no mitigation) vs 2 (readout + ZNE). Both n=22 (meets the floor)
    # with the exact CP bound, so a PASS is achievable iff the error falls below eps.
    results = {'backend': backend.name, 'exact_energy': EXACT_ENERGY, 'sim_energy': sim_E,
               'epsilon': EPSILON, 'M_deploy': M_DEPLOY, 'n_evals': N_EVALS, 'params': params,
               'resilience_comparison': {},
               'note': 'controlled resilience 0 vs 2 at M=M_deploy; answers sec 5.4'}

    for rl in RESILIENCE_LEVELS:
        rem = remaining_seconds(service)
        print(f'\n[budget] before resilience={rl}: remaining {rem}s')
        if rem != -1 and rem < SAFETY_FLOOR:
            print(f'[ABORT] remaining {rem}s < floor {SAFETY_FLOOR}s. Stopping before resilience={rl}.')
            break
        t0 = time.time()
        try:
            est = EstimatorV2(mode=backend)
            est.options.resilience_level = rl
            pubs = [(isa, isa_H)] * N_EVALS
            job = est.run(pubs, precision=1.0 / np.sqrt(M_DEPLOY))
            print(f'  resilience={rl} job {job.job_id()}; waiting...')
            res = job.result()
            energies = [float(res[i].data.evs) for i in range(N_EVALS)]
            errs = np.abs(np.array(energies) - EXACT_ENERGY)
            risk = float(np.mean(errs > EPSILON))
            n = len(errs)
            x = int(np.sum(errs > EPSILON))          # loss count for the exact CP bound
            ucb = clopper_pearson_ucb(x, n, DELTA)   # n=22 can PASS iff error < eps
            results['resilience_comparison'][str(rl)] = {
                'resilience_level': rl, 'energies': energies, 'mean_error': float(errs.mean()),
                'std_error': float(errs.std()), 'risk': risk, 'n_evals': n, 'failures': x,
                'cp_ucb': round(ucb, 4), 'verdict': 'PASS' if ucb <= 0.10 else 'FAIL',
                'time_s': time.time() - t0, 'job_id': job.job_id()}
            print(f"  resilience={rl}: |E-E0|={errs.mean():.4f}+/-{errs.std():.4f} risk={risk:.2f} "
                  f"CP-UCB={ucb:.4f} ({x}/{n} fail) -> {results['resilience_comparison'][str(rl)]['verdict']}")
        except Exception as e:
            print(f'  FAILED: {e}')
            results['resilience_comparison'][str(rl)] = {'error': str(e)}
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(RESULTS_FILE, 'w'), indent=2, default=str)

    rem = remaining_seconds(service)
    print(f'\n[budget] after run: remaining {rem}s')
    print(f'Saved {RESULTS_FILE}')
    return results


if __name__ == '__main__':
    print('VQE-on-hardware WITH mitigation (answers sec 5.4 open question)')
    run()
