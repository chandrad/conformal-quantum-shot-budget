#!/usr/bin/env python3
"""
VQE on IBM Hardware — Using Qiskit Estimator (v2)
==================================================
Uses Qiskit EstimatorV2 directly (not PennyLane-Qiskit).
Measures H2 Hamiltonian expectation value at different shot counts.

Free tier budget: ~2-3 min QPU time.
"""

import numpy as np
import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_h2_qiskit_circuit(params, n_qubits=4):
    """Build the VQE ansatz in Qiskit (AllSinglesDoubles equivalent)."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)

    # Hartree-Fock: |1100>
    qc.x(0)
    qc.x(1)

    # Single excitations: [0,2], [1,3]
    # Double excitation: [0,1,2,3]
    # Simplified parameterized ansatz matching PennyLane's AllSinglesDoubles
    # params[0]: single 0→2, params[1]: single 1→3, params[2]: double 0,1→2,3

    # Single excitation 0→2: Givens rotation
    qc.ry(params[0], 0)
    qc.cx(0, 2)
    qc.ry(-params[0], 0)
    qc.cx(0, 2)

    # Single excitation 1→3
    qc.ry(params[1], 1)
    qc.cx(1, 3)
    qc.ry(-params[1], 1)
    qc.cx(1, 3)

    # Double excitation 0,1→2,3 (simplified)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.ry(params[2], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)

    return qc


def run_ibm_vqe_estimator():
    """Run VQE on IBM hardware using Estimator primitive."""
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    print("=" * 60)
    print("VQE H2 on IBM Hardware (Estimator)")
    print("=" * 60)

    # H2 Hamiltonian in SparsePauliOp format
    # Standard STO-3G Jordan-Wigner at d=0.735 Å
    h2_hamiltonian = SparsePauliOp.from_list([
        ("IIII", -0.0420789764),
        ("IIIZ", 0.1777128746),
        ("IIZI", 0.1777128746),
        ("IZII", -0.2427428051),
        ("ZIII", -0.2427428051),
        ("IIZZ", 0.1705973833),
        ("ZYXZ", 0.0447501440),   # Y0 X1 X2 Y3
        ("YXYZ", -0.0447501440),  # Y0 Y1 X2 X3
        ("XYYZ", -0.0447501440),  # X0 X1 Y2 Y3
        ("XYYY", 0.0447501440),   # X0 Y1 Y2 X3
        ("IZIZ", 0.1229330506),
        ("IZZI", 0.1676831946),
        ("ZIIZ", 0.1676831946),
        ("ZIZI", 0.1229330506),
        ("ZZII", 0.1762764039),
    ])

    exact_energy = -1.136189454088
    epsilon = 0.02  # 20 mHa

    # Get optimal VQE params from PennyLane (pre-computed)
    # These are the converged params for AllSinglesDoubles on H2
    import pennylane as qml
    from pennylane import numpy as pnp

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.735 * 1.8897259886]])
    H_pl, n_q = qml.qchem.molecular_hamiltonian(symbols, geometry)
    electrons = 2
    singles, doubles = qml.qchem.excitations(electrons, n_q)
    hf_state = qml.qchem.hf_state(electrons, n_q)
    n_params = len(singles) + len(doubles)

    dev = qml.device('default.qubit', wires=n_q)

    @qml.qnode(dev)
    def vqe_sim(p):
        qml.AllSinglesDoubles(p, wires=range(n_q), hf_state=hf_state,
                              singles=singles, doubles=doubles)
        return qml.expval(H_pl)

    params_pl = pnp.array(np.zeros(n_params), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    for _ in range(100):
        params_pl, _ = opt.step_and_cost(vqe_sim, params_pl)

    sim_energy = float(vqe_sim(params_pl))
    print(f"  Simulator energy: {sim_energy:.6f} Ha")
    print(f"  Exact energy: {exact_energy:.6f} Ha")
    print(f"  Params: {[float(p) for p in params_pl]}")

    # Connect to IBM
    print("\n  Connecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, min_num_qubits=4)
    print(f"  Backend: {backend.name} ({backend.num_qubits}q)")

    # Build Qiskit circuit with trained params
    params_list = [float(p) for p in params_pl]
    qc = build_h2_qiskit_circuit(params_list, n_qubits=4)
    print(f"  Circuit depth: {qc.depth()}, gates: {qc.size()}")

    # Transpile for backend
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    isa_circuit = pm.run(qc)
    isa_hamiltonian = h2_hamiltonian.apply_layout(isa_circuit.layout)
    print(f"  Transpiled depth: {isa_circuit.depth()}")

    # Run at different shot counts
    shot_counts = [10, 50, 100, 500, 1000]
    n_evals = 10
    results = {
        'backend': backend.name, 'exact_energy': exact_energy,
        'sim_energy': sim_energy, 'epsilon': epsilon,
        'params': params_list, 'shot_results': {},
    }

    for M in shot_counts:
        print(f"\n  M={M} shots on {backend.name}...")
        t0 = time.time()

        try:
            estimator = EstimatorV2(mode=backend)
            # Run n_evals independent estimations
            pubs = [(isa_circuit, isa_hamiltonian)] * n_evals
            job = estimator.run(pubs, precision=1.0/np.sqrt(M))
            print(f"    Job {job.job_id()} submitted. Waiting...")
            result = job.result()

            energies = [float(result[i].data.evs) for i in range(n_evals)]
            elapsed = time.time() - t0

            errors = np.abs(np.array(energies) - exact_energy)
            risk = float(np.mean(errors > epsilon))

            results['shot_results'][str(M)] = {
                'energies': energies,
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'risk': risk,
                'time_s': elapsed,
                'job_id': job.job_id(),
            }
            print(f"    |E-E*|={np.mean(errors):.4f}±{np.std(errors):.4f}, "
                  f"risk={risk:.3f}, time={elapsed:.1f}s")

        except Exception as e:
            print(f"    FAILED: {e}")
            results['shot_results'][str(M)] = {'error': str(e)}

    # CRC analysis
    print("\n  CRC shot budget selection:")
    alpha = 0.10
    for M in sorted(shot_counts, reverse=True):
        M_str = str(M)
        if 'energies' not in results['shot_results'].get(M_str, {}):
            continue
        energies = results['shot_results'][M_str]['energies']
        losses = (np.abs(np.array(energies) - exact_energy) > epsilon).astype(float)
        r = np.mean(losses)
        n = len(losses)
        ucb = r + np.sqrt(np.log(1/0.10) / (2 * n))
        status = "PASS" if ucb <= alpha else "FAIL"
        print(f"    M={M:>5d}: R̂={r:.3f}, UCB={ucb:.3f} [{status}]")

    with open(RESULTS_DIR / 'ibm_vqe_hardware.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to ibm_vqe_hardware.json")

    return results


if __name__ == '__main__':
    run_ibm_vqe_estimator()
