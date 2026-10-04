"""Generate camera-ready figures from result JSONs (reproducible, no hardcoded numbers).
Okabe-Ito colorblind-safe palette; clean paper style. Outputs PDFs into figures/."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)

# Okabe-Ito (colorblind-safe)
OI = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
      'vermillion': '#D55E00', 'gray': '#999999', 'black': '#000000'}

plt.rcParams.update({
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'figure.dpi': 150, 'savefig.bbox': 'tight',
})


def fig_hw_vs_sim():
    """Figure 1: real IBM hardware error (from ibm_vqe_hardware.json) vs the
    noiseless simulator shot-noise reference for the same H2 model, recomputed
    here via the v4 sampler so the figure is fully reproducible."""
    import sys
    sys.path.insert(0, str(HERE))
    from exp_vqe_crc_v4 import build_molecule, compute_exact_energy, evaluate_energy
    d = json.load(open(RES / 'ibm_vqe_hardware.json'))
    eps = d['epsilon']
    sr = d['shot_results']
    Ms = sorted(int(m) for m in sr)
    hw_err = [sr[str(m)]['mean_error'] for m in Ms]
    hw_std = [sr[str(m)]['std_error'] for m in Ms]
    # Same H2 model on a noiseless simulator: finite-shot error around the exact E0.
    H, n_q, elec, singles, doubles, hf, _ = build_molecule('H2', 0.735)
    E0, params = compute_exact_energy(H, n_q, elec, singles, doubles, hf, len(singles) + len(doubles))
    sim_err = []
    for m in Ms:
        e = evaluate_energy(H, n_q, elec, singles, doubles, hf, params, shots=m, n_evals=40)
        sim_err.append(float(np.mean(np.abs(e - E0))))
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.errorbar(Ms, hw_err, yerr=hw_std, color=OI['vermillion'], marker='o', ms=4,
                lw=1.6, capsize=3, label='IBM hardware')
    ax.plot(Ms, sim_err, color=OI['blue'], marker='s', ms=4, lw=1.6, label='simulator')
    ax.axhline(eps, color=OI['gray'], ls='--', lw=1.2)
    ax.text(Ms[-1], eps, f' $\\epsilon$={eps}', color=OI['gray'], va='bottom', ha='right', fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('shots $M$')
    ax.set_ylabel('energy error $|E_M - E_0|$ (Ha)')
    ax.set_title('H$_2$: hardware plateaus, simulator passes', fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(FIG / 'hw_vs_sim.pdf')
    plt.close(fig)
    print('wrote hw_vs_sim.pdf  hw=%s sim=%s'
          % ([round(v, 3) for v in hw_err], [round(v, 3) for v in sim_err]))


def fig_mitigation():
    d = json.load(open(RES / 'ibm_vqe_hardware_mitigated.json'))
    eps = d['epsilon']
    rc = d['resilience_comparison']
    labels = ['No mitigation\n(resilience 0)', 'Readout + ZNE\n(resilience 2)']
    errs = [rc['0']['mean_error'], rc['2']['mean_error']]
    stds = [rc['0']['std_error'], rc['2']['std_error']]
    colors = [OI['gray'], OI['blue']]
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    x = np.arange(2)
    bars = ax.bar(x, errs, yerr=stds, capsize=4, color=colors, width=0.55,
                  edgecolor='white', linewidth=0.8)
    ax.axhline(eps, color=OI['vermillion'], lw=1.6, ls='--')
    ax.text(1.42, eps, f'  tolerance $\\epsilon$={eps}', color=OI['vermillion'],
            va='bottom', ha='right', fontsize=8)
    for xi, e, s in zip(x, errs, stds):
        ax.text(xi, e + s + 0.004, f'{e:.3f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('energy error $|E_M - E_0|$ (Ha)')
    ax.set_ylim(0, max(errs) + max(stds) + 0.02)
    ax.set_title(f"Hardware ({d['backend']}), $M$={d['M_deploy']}, $n$={d['n_evals']}: both FAIL",
                 fontsize=8.5)
    fig.savefig(FIG / 'mitigation.pdf')
    plt.close(fig)
    print('wrote mitigation.pdf  (0:%.4f, 2:%.4f, eps %.3f)' % (errs[0], errs[1], eps))


def fig_closure():
    d = json.load(open(RES / 'vqe_crc_v5_cp.json'))
    budgets = ['1000', '5000', '10000']
    # mean certified fraction over the 4 H2 bond lengths
    h2 = {k: v for k, v in d['molecules'].items() if k.startswith('H2_')}
    def frac(method, B):
        vals = [mv['budget_comparison'][B][method]['certified_count'] / 50.0
                for mv in h2.values() if B in mv['budget_comparison']]
        return 100 * np.mean(vals) if vals else 0.0
    reuse = [frac('CP-CRC-reuse', B) for B in budgets]
    equal = [frac('CP-CRC-equal', B) for B in budgets]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(budgets)); w = 0.38
    b1 = ax.bar(x - w/2, reuse, w, label='CP-CRC-reuse', color=OI['green'],
                edgecolor='white', linewidth=0.8)
    b2 = ax.bar(x + w/2, equal, w, label='CP-CRC-equal (no reuse)', color=OI['orange'],
                edgecolor='white', linewidth=0.8)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f'{h:.0f}', ha='center',
                    va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'$B$={b}' for b in budgets])
    ax.set_ylabel('trials certifying $M^*<1000$ (\\%)')
    ax.set_ylim(0, 70)
    ax.set_title('Closing the gap on H$_2$ (mean over 4 bonds)', fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.savefig(FIG / 'closure.pdf')
    plt.close(fig)
    print('wrote closure.pdf  reuse=%s equal=%s' % ([round(v) for v in reuse], [round(v) for v in equal]))


if __name__ == '__main__':
    fig_hw_vs_sim()   # Figure 1
    fig_closure()     # Figure 2
    fig_mitigation()  # supplementary (hardware mitigation; not in main paper)
    print('done')
