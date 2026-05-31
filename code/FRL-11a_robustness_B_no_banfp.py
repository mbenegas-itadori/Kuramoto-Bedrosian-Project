"""
FRL-11a · robustness_B_no_banfp.py
=====================================
Teste de robustez B: exclui BANFP do universo SHI
e re-simula P2 (top 5%, 10%, 15%) para verificar
se o retorno de 28.79% depende de um único ativo.

Compara:
  P2_full  : SHI top 5%  com BANFP (baseline)
  P3_full  : SHI top 10% com BANFP (baseline)
  P4_full  : SHI top 15% com BANFP (baseline)
  P2_noB   : SHI top 5%  sem BANFP
  P3_noB   : SHI top 10% sem BANFP
  P4_noB   : SHI top 15% sem BANFP

Inputs:
  - log_returns_342_final.csv (local)
  - phase_342_bk.csv (local)
  - bk_validation_meta.csv (Drive)
  - portfolio_returns.csv (Drive/frl/ — baseline)

Outputs (Drive/frl/robustness/):
  - robustness_B_metrics.csv
  - robustness_B_returns.csv
  - fig_robustness_B.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import os
import warnings
warnings.filterwarnings('ignore')
from google.colab import drive

drive.mount('/content/drive')

DRIVE_DIR  = '/content/drive/MyDrive/ste_matrices_corrected'
FRL_DIR    = os.path.join(DRIVE_DIR, 'frl')
ROB_DIR    = os.path.join(FRL_DIR, 'robustness')
os.makedirs(ROB_DIR, exist_ok=True)

FIG_DPI    = 300
REBAL_FREQ = 63
MIN_HIST   = 504
RF_ANNUAL  = 0.02
X_VALUES   = [0.05, 0.10, 0.15]
EXCLUDE    = ['BANFP']

CRISES = {
    'GFC':   ('2009-01-02', '2009-06-30'),
    'COVID': ('2020-02-19', '2020-03-23'),
}

COLORS = {
    'P2_full': '#AA2828',
    'P3_full': '#1E7840',
    'P4_full': '#AA5500',
    'P2_noB':  '#FF8080',
    'P3_noB':  '#80C880',
    'P4_noB':  '#FFB060',
}
LABELS = {
    'P2_full': 'SHI 5% (with BANFP)',
    'P3_full': 'SHI 10% (with BANFP)',
    'P4_full': 'SHI 15% (with BANFP)',
    'P2_noB':  'SHI 5% (excl. BANFP)',
    'P3_noB':  'SHI 10% (excl. BANFP)',
    'P4_noB':  'SHI 15% (excl. BANFP)',
}

# ─────────────────────────────────────────────────────────────
# 1. CARREGAR DADOS
# ─────────────────────────────────────────────────────────────
print("Carregando dados...")

lr = pd.read_csv('log_returns_342_final.csv',
                 index_col=0, parse_dates=True)
ph = pd.read_csv('phase_342_bk.csv',
                 index_col=0, parse_dates=True)
meta = pd.read_csv(
    os.path.join(DRIVE_DIR, 'bk_validation_meta.csv'))
meta['epsilon'] = meta['error_pct'] / 100
valid = meta['ticker'].tolist()
tickers = [t for t in valid
           if t in lr.columns and t in ph.columns]
N = len(tickers)

# Alinhar índices
common = lr.index.intersection(ph.index)
lr = lr.loc[common, tickers]
ph = ph.loc[common, tickers]

# Retornos simples com winsorize
ret = np.expm1(lr)
ret = ret.fillna(0.0).replace([np.inf, -np.inf], 0.0)
ret = ret.clip(-0.50, 0.50)

dates = ret.index
T     = len(dates)

# Universo sem BANFP
tickers_noB = [t for t in tickers if t not in EXCLUDE]
N_noB = len(tickers_noB)
noB_idx = [tickers.index(t) for t in tickers_noB]

print(f"  N completo : {N}")
print(f"  N sem BANFP: {N_noB}")

# ─────────────────────────────────────────────────────────────
# 2. PRÉ-COMPUTAR UNWRAP
# ─────────────────────────────────────────────────────────────
print("Pré-computando fases...")
phases_unwr = np.unwrap(ph.values, axis=0)
omega_inst  = np.diff(phases_unwr, axis=0)

ticker_to_idx = {t: i for i, t in enumerate(tickers)}

# ─────────────────────────────────────────────────────────────
# 3. CARREGAR BASELINE DO ARQUIVO SALVO
# ─────────────────────────────────────────────────────────────
print("Carregando baseline...")
baseline = pd.read_csv(
    os.path.join(FRL_DIR, 'portfolio_returns.csv'),
    index_col=0, parse_dates=True)

# ─────────────────────────────────────────────────────────────
# 4. SIMULAR PORTFÓLIOS SEM BANFP
# ─────────────────────────────────────────────────────────────
print("Simulando portfólios sem BANFP...")

rebal_points = list(range(MIN_HIST, T - 1, REBAL_FREQ))
port_daily   = {f'P{i}_noB': [] for i in [2, 3, 4]}
port_dates   = []

for step, t_idx in enumerate(rebal_points):
    if step % 10 == 0:
        print(f"  [{step+1:3d}/{len(rebal_points)}]"
              f" {dates[t_idx].date()}", end='\r')

    # SHI apenas sobre ativos sem BANFP
    omega_i   = omega_inst[:t_idx-1, noB_idx].mean(axis=0)
    omega_bar = omega_i.mean()
    shi       = np.abs(omega_i - omega_bar) / omega_bar
    shi_s     = pd.Series(shi, index=tickers_noB)

    # Pesos por threshold
    weights = {}
    for x in X_VALUES:
        n_sel = max(2, int(np.ceil(N_noB * x)))
        top   = shi_s.nlargest(n_sel).index
        w     = np.zeros(N)
        for tk in top:
            w[ticker_to_idx[tk]] = 1.0
        w /= w.sum()
        weights[x] = w

    next_t = rebal_points[step+1] \
        if step < len(rebal_points)-1 else T

    for day_idx in range(t_idx, next_t):
        day   = dates[day_idx]
        r_day = ret.iloc[day_idx].values
        port_dates.append(day)
        port_daily['P2_noB'].append(float(weights[0.05] @ r_day))
        port_daily['P3_noB'].append(float(weights[0.10] @ r_day))
        port_daily['P4_noB'].append(float(weights[0.15] @ r_day))

print(f"\n  Concluído.")

noB_df = pd.DataFrame(port_daily, index=port_dates)
noB_df.index = pd.to_datetime(noB_df.index)
noB_df = noB_df[~noB_df.index.duplicated(keep='first')]

# Combinar com baseline
start = noB_df.index[0]
base_sub = baseline.loc[start:, ['P2', 'P3', 'P4']].copy()
base_sub.columns = ['P2_full', 'P3_full', 'P4_full']
all_ret = base_sub.join(noB_df, how='inner')

# ─────────────────────────────────────────────────────────────
# 5. MÉTRICAS
# ─────────────────────────────────────────────────────────────
def metrics(s, label):
    cum    = (1 + s).cumprod()
    total  = cum.iloc[-1] - 1
    n      = len(s)
    ann_r  = (1+total)**(252/n) - 1
    ann_v  = s.std() * np.sqrt(252)
    sharpe = (ann_r - RF_ANNUAL) / ann_v \
             if ann_v > 0 else np.nan
    roll_max = cum.cummax()
    max_dd = ((cum - roll_max) / roll_max).min()
    rows = {'label': label,
            'ann_return': ann_r*100,
            'ann_vol': ann_v*100,
            'sharpe': sharpe,
            'max_drawdown': max_dd*100}
    for cr, (cs, ce) in CRISES.items():
        m  = (s.index >= cs) & (s.index <= ce)
        cr_s = s[m]
        if len(cr_s) > 0:
            c  = (1 + cr_s).cumprod()
            dd = ((c - c.cummax()) / c.cummax()).min()
            rows[f'{cr}_drawdown'] = dd * 100
        else:
            rows[f'{cr}_drawdown'] = np.nan
    return rows

rows = []
for k in ['P2_full','P3_full','P4_full',
          'P2_noB', 'P3_noB', 'P4_noB']:
    rows.append(metrics(all_ret[k], LABELS[k]))
met_df = pd.DataFrame(rows)
met_df.index = ['P2_full','P3_full','P4_full',
                'P2_noB', 'P3_noB', 'P4_noB']

print("\nTabela de robustez B:")
print(f"{'':35s} {'P2':>8} {'P3':>8} {'P4':>8}")
print(f"{'':35s} {'full':>8} {'full':>8} {'full':>8}")
print("-"*60)
for col in ['ann_return','ann_vol','sharpe',
            'max_drawdown','GFC_drawdown',
            'COVID_drawdown']:
    row = f"{col:35s}"
    for k in ['P2_full','P3_full','P4_full']:
        row += f"{met_df.loc[k,col]:>8.3f}"
    print(row)

print(f"\n{'':35s} {'P2':>8} {'P3':>8} {'P4':>8}")
print(f"{'':35s} {'no B':>8} {'no B':>8} {'no B':>8}")
print("-"*60)
for col in ['ann_return','ann_vol','sharpe',
            'max_drawdown','GFC_drawdown',
            'COVID_drawdown']:
    row = f"{col:35s}"
    for k in ['P2_noB','P3_noB','P4_noB']:
        row += f"{met_df.loc[k,col]:>8.3f}"
    print(row)

# ─────────────────────────────────────────────────────────────
# 6. FIGURA
# ─────────────────────────────────────────────────────────────
cum = (1 + all_ret).cumprod()
ymax = cum.max().max()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Robustness Test B: Excluding BANFP from SHI Universe\n'
    'Solid: with BANFP | Dashed: without BANFP',
    fontsize=11, fontweight='bold'
)

# Painel esquerdo: retorno acumulado
ax = axes[0]
for k, ls in [('P2_full','-'),('P3_full','-'),
               ('P4_full','-'),
               ('P2_noB','--'),('P3_noB','--'),
               ('P4_noB','--')]:
    ax.plot(cum.index, cum[k],
            color=COLORS[k], linewidth=1.5,
            linestyle=ls, alpha=0.85,
            label=LABELS[k])
for cr, (cs, ce) in CRISES.items():
    ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
               alpha=0.07, color='red'
               if cr == 'GFC' else 'orange')
    mid = pd.Timestamp(cs) + (
        pd.Timestamp(ce)-pd.Timestamp(cs))/2
    ax.text(mid, ymax*0.96, cr,
            ha='center', va='top',
            fontsize=7.5, color='gray')
ax.set_title('Cumulative return', fontsize=9)
ax.set_ylabel('Cumulative return (base = 1)',
              fontsize=9)
ax.legend(fontsize=6.5, loc='upper left')
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(3))
ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y'))
ax.tick_params(axis='x', rotation=30)

# Painel direito: Sharpe comparison
ax = axes[1]
labels_short = ['SHI 5%', 'SHI 10%', 'SHI 15%']
x = np.arange(3)
w = 0.35
sharpe_full = [met_df.loc[k,'sharpe']
               for k in ['P2_full','P3_full','P4_full']]
sharpe_noB  = [met_df.loc[k,'sharpe']
               for k in ['P2_noB','P3_noB','P4_noB']]
ax.bar(x - w/2, sharpe_full, w,
       color=['#AA2828','#1E7840','#AA5500'],
       alpha=0.8, label='With BANFP')
ax.bar(x + w/2, sharpe_noB, w,
       color=['#FF8080','#80C880','#FFB060'],
       alpha=0.8, label='Without BANFP')
for i, (vf, vn) in enumerate(
        zip(sharpe_full, sharpe_noB)):
    ax.text(i-w/2, vf+0.01, f'{vf:.3f}',
            ha='center', va='bottom', fontsize=8)
    ax.text(i+w/2, vn+0.01, f'{vn:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels_short)
ax.set_ylabel('Sharpe Ratio', fontsize=9)
ax.set_title('Sharpe ratio comparison', fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(ROB_DIR,
            'fig_robustness_B.png'),
            dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_robustness_B.png ✓")

# ─────────────────────────────────────────────────────────────
# 7. SALVAR
# ─────────────────────────────────────────────────────────────
met_df.to_csv(os.path.join(ROB_DIR,
              'robustness_B_metrics.csv'))
all_ret.to_csv(os.path.join(ROB_DIR,
               'robustness_B_returns.csv'))
print("  CSVs salvos ✓")

print("\n" + "="*55)
print("ROBUSTNESS B CONCLUÍDO")
print("="*55)
for k in ['P2_full','P2_noB']:
    print(f"  {LABELS[k]:35s} "
          f"Sharpe={met_df.loc[k,'sharpe']:.3f} "
          f"Ret={met_df.loc[k,'ann_return']:.2f}%")
print("="*55)
