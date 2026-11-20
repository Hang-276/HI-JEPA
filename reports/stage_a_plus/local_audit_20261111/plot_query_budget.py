"""Plot the frozen query-budget results; no recomputation or refitting."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
source = HERE.parent / 'fine_reference_query_budget_20261111' / 'SUMMARY.json'
data = json.loads(source.read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1, 1.15]})
budgets = ('1', '64', '512', 'all')
x = [data['query_cost_per_panel'][b]['reference_nodes'] for b in budgets]
for rep, color in [('fine', '#237b82'), ('raw', '#b66b35')]:
    y = [100 * data['hits'][f'{rep}_{b}']['queried_recall'] for b in budgets]
    axes[0].plot(x, y, 'o-', label=rep.capitalize(), color=color, linewidth=2)
axes[0].set_xscale('log')
axes[0].set_xticks(x, ['1', '64', '512', '18,528\n(all)'])
axes[0].set(xlabel='Queried reference nodes per panel', ylabel='Candidate-error recall (%)', ylim=(20, 83))
axes[0].annotate('Raw64: fallback\nin all 16 panels', (64, 100 * data['hits']['raw_64']['queried_recall']),
                 xytext=(110, 40), arrowprops={'arrowstyle': '->', 'color': '#777777'}, fontsize=9)
axes[0].legend(frameon=False, loc='upper left')
axes[0].set_title('A. Fixed rules and query budgets', loc='left', fontweight='bold')
comparisons = [('fine64_vs_raw64', 'Fine64 minus Raw64', '#237b82'),
               ('fine64_vs_fine_all', 'Fine64 minus Fine-all', '#795787')]
for y, (key, label, color) in zip([1, 0], comparisons):
    row = data['primary'][key]
    value, low, high = 100 * np.array([row['delta_recall'], *row['ci95']])
    axes[1].errorbar(value, y, xerr=[[value - low], [high - value]], fmt='o', color=color, capsize=5)
    axes[1].text(value, y + .16, f'{value:+.2f} [{low:+.2f}, {high:+.2f}]', ha='center', color=color, fontsize=9)
axes[1].axvline(0, linestyle=':', linewidth=1, color='#999999')
axes[1].axvline(-5, linestyle='--', linewidth=1, color='#795787')
axes[1].text(-5, -.43, '-5 pp\nretention limit', ha='center', fontsize=8, color='#795787')
axes[1].set_yticks([1, 0], [x[1] for x in comparisons])
axes[1].set(xlabel='Paired recall difference (percentage points)', xlim=(-17, 53), ylim=(-.62, 1.5))
axes[1].set_title('B. Gain passes; retention remains unresolved', loc='left', fontweight='bold')
fig.suptitle('Reference-query study: 16 independent reference + evaluation panels', fontsize=13)
fig.text(.5, .01, '119 candidate errors; 96 alarms per rule. Intervals: paired whole-panel bootstrap.\n'
         'Reference budgets vary; evaluation queries and historical calibration stay fixed.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .09, 1, .95), w_pad=2.5)
fig.savefig(HERE / 'QUERY_BUDGET.png', dpi=170)
