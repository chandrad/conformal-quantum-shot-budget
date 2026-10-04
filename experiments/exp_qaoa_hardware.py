#!/usr/bin/env python3
"""
QAOA MaxCut on IBM hardware — does the shot-budget certificate behave the same
on a non-VQE workload? Mirrors exp_ibm_vqe_v2.py exactly: measure <C> at several
shot counts via EstimatorV2, compare to the noiseless reference C0, run the CRC
UCB verdict. Same 4-node graph / p=1 QAOA / angles as exp_qaoa_crc.py (S2).

The angles are from deterministic zero-offset GD on the analytic simulator
(identical to S2), so this hardware circuit measures the SAME state S2 certifies
in simulation. The question: on real hardware, is <C> noise-limited (certificate
should refuse), as the VQE experiment found?

AUDIT NOTES (self):
  - QAOA cost op C = sum_{(i,j)} 0.5 (I - Z_i Z_j); C0 = noiseless <C> at angles.
    IDENTICAL definition and sign convention to exp_qaoa_crc.py.
  - Qiskit cost layer uses RZZ(2*gamma) per edge; mixer RX(2*beta) per qubit;
    Hadamard init. This matches PennyLane's CNOT-RZ(2g)-CNOT + RX(2b) layer.
  - epsilon is passed in to match whatever S2's mid-grid-transition rule chose,
    so sim and hardware use the SAME tolerance (do NOT hardcode a different one).
  - precision=1/sqrt(M) reproduces M-shot noise in EstimatorV2 (same as VQE hw).
  - No error mitigation (free-tier budget), disclosed, as in the VQE hardware run.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_FILE = HERE.parent / 'results' / 'qaoa_hardware.json'

N_NODES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
P_LAYERS = 1


def optimize_qaoa_angles():
    """Deterministic (fixed-init) QAOA optimization on the analytic simulator.
    Returns (angles=[gamma, beta], C0). Identical to exp_qaoa_crc.optimize_qaoa."""
    import pennylane as qml
    from pennylane import numpy as pnp

    coeffs, ops = [], []
    for (i, j) in EDGES:
        coeffs += [0.5, -0.5]
        ops += [qml.Identity(0), qml.PauliZ(i) @ qml.PauliZ(j)]
    C_H = qml.Hamiltonian(coeffs, ops)

    dev = qml.device('default.qubit', wires=N_NODES)

    @qml.qnode(dev)
    def circuit(params):
        for w in range(N_NODES):
            qml.Hadamard(wires=w)
        for l in range(P_LAYERS):
            for (i, j) in EDGES:
                qml.CNOT(wires=[i, j]); qml.RZ(2.0 * params[l], wires=j); qml.CNOT(wires=[i, j])
            for w in range(N_NODES):
                qml.RX(2.0 * params[P_LAYERS + l], wires=w)
        return qml.expval(C_H)

    params = pnp.array([0.1] * P_LAYERS + [0.2] * P_LAYERS, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.3)
    for _ in range(200):
        params, _ = opt.step_and_cost(lambda p: -circuit(p), params)
    return np.array(params.tolist()), float(circuit(params))


def build_qaoa_qiskit(angles):
    from qiskit.circuit import QuantumCircuit
    qc = QuantumCircuit(N_NODES)
    for w in range(N_NODES):
        qc.h(w)
    for l in range(P_LAYERS):
        gamma, beta = angles[l], angles[P_LAYERS + l]
        for (i, j) in EDGES:
            qc.rzz(2.0 * gamma, i, j)
        for w in range(N_NODES):
            qc.rx(2.0 * beta, w)
    return qc


def cut_sparsepauliop():
    from qiskit.quantum_info import SparsePauliOp
    terms = []
    n_edges = len(EDGES)
    terms.append(("I" * N_NODES, 0.5 * n_edges))      # sum of 0.5*I over edges
    for (i, j) in EDGES:
        label = ['I'] * N_NODES
        label[N_NODES - 1 - i] = 'Z'                   # qiskit little-endian
        label[N_NODES - 1 - j] = 'Z'
        terms.append((''.join(label), -0.5))
    return SparsePauliOp.from_list(terms)


def run(epsilon, shot_counts=(10, 50, 100, 500, 1000), n_evals=10):
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    angles, C0 = optimize_qaoa_angles()
    print(f'QAOA angles={angles.tolist()}, C0(noiseless)={C0:.5f}, epsilon={epsilon}')

    C_op = cut_sparsepauliop()
    qc = build_qaoa_qiskit(angles)
    print(f'circuit depth={qc.depth()}, gates={qc.size()}')

    print('Connecting to IBM Quantum...')
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, min_num_qubits=N_NODES)
    print(f'backend: {backend.name} ({backend.num_qubits}q)')

    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    isa_qc = pm.run(qc)
    isa_op = C_op.apply_layout(isa_qc.layout)
    print(f'transpiled depth={isa_qc.depth()}')

    results = {'backend': backend.name, 'problem': 'QAOA MaxCut',
               'n_nodes': N_NODES, 'edges': EDGES, 'p_layers': P_LAYERS,
               'C0_noiseless': C0, 'epsilon': epsilon,
               'angles': angles.tolist(), 'shot_results': {}}

    for M in shot_counts:
        print(f'\nM={M} on {backend.name}...')
        t0 = time.time()
        try:
            est = EstimatorV2(mode=backend)
            pubs = [(isa_qc, isa_op)] * n_evals
            job = est.run(pubs, precision=1.0 / np.sqrt(M))
            print(f'  job {job.job_id()} submitted; waiting...')
            res = job.result()
            cuts = [float(res[i].data.evs) for i in range(n_evals)]
            errs = np.abs(np.array(cuts) - C0)
            risk = float(np.mean(errs > epsilon))
            results['shot_results'][str(M)] = {
                'cuts': cuts, 'mean_error': float(errs.mean()),
                'std_error': float(errs.std()), 'risk': risk,
                'time_s': time.time() - t0, 'job_id': job.job_id()}
            print(f'  |C-C0|={errs.mean():.4f}+/-{errs.std():.4f}, risk={risk:.3f}')
        except Exception as e:
            print(f'  FAILED: {e}')
            results['shot_results'][str(M)] = {'error': str(e)}
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(RESULTS_FILE, 'w'), indent=2, default=str)

    print('\nCRC UCB verdict (alpha=0.10):')
    for M in sorted(shot_counts, reverse=True):
        r = results['shot_results'].get(str(M), {})
        if 'cuts' not in r:
            continue
        losses = (np.abs(np.array(r['cuts']) - C0) > epsilon).astype(float)
        n = len(losses)
        ucb = losses.mean() + np.sqrt(np.log(1 / 0.10) / (2 * n))
        print(f'  M={M:>5d}: Rhat={losses.mean():.3f} UCB={ucb:.3f} '
              f'[{"PASS" if ucb <= 0.10 else "FAIL"}]')
    print(f'\nSaved {RESULTS_FILE}')
    return results


if __name__ == '__main__':
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else None
    if eps is None:
        # read epsilon chosen by S2 so sim and hardware match
        s2 = HERE.parent / 'results' / 'qaoa_crc.json'
        if s2.exists():
            eps = json.load(open(s2))['config']['epsilon']
            print(f'[epsilon from S2 config: {eps}]')
        else:
            raise SystemExit('Run S2 (exp_qaoa_crc.py) first, or pass epsilon as arg.')
    run(eps)
