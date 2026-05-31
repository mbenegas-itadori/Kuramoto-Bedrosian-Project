"""
FRL-11b · robustness_C_rolling_window.py
==========================================
Teste de robustez C: compara expanding window
(usado no paper) com rolling window de 3 anos
(756 dias) para estimação do SHI.

Hipótese: se o SHI é uma propriedade estrutural
estável, a escolha de expanding vs rolling não
deve mudar materialmente os resultados.

Compara:
  P2_exp : SHI top 5%  expanding (baseline)
  P3_exp : SHI top 10% expanding (baseline)
  P4_exp : SHI top 15% expanding (baseline)
  P2_rol : SHI top 5%  rolling 756d
  P3_rol : SHI top 10% rolling 756d
  P4_rol : SHI top 15% rolling 756d

Inputs:
  - log_returns_342_final.csv (local)
  - phase_342_bk.csv (local)
  - bk_validation_meta.csv (Drive)
  - portfolio_returns.csv (Drive/frl/ — baseline)

Outputs (Drive/frl/robustness/):
  - robustness_C_metrics.csv
  - robustness_C_returns.csv
  - fig_robustness_C.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
MIN_HIST   = 756    # rolling requer 756 dias mínimo
RF_ANNUAL  = 0.02
X_VALUES   = [0.05, 0.10, 0.15]
WIN_ROLL   = 756

CRISES = {
    'GFC':   ('2009-01-02', '2009-06-30'),
    'COVID': ('2020-02-19', '2020-03-23'),
}

COLORS = {
    'P2_exp': '#AA2828',
    'P3_exp': '#1E7840',
    'P4_exp': '#AA5500',
    'P2_rol': '#FF8080',
    'P3_rol': '#80C880',
    'P4_rol': '#FFB060',
}
LABELS = {
    'P2_exp': 'SHI 5% (expanding)',
    'P3_exp': 'SHI 10% (expanding)',
    'P4_exp': 'SHI 15% (expanding)',
    'P2_rol': 'SHI 5% (rolling 3yr)',
    'P3_rol': 'SHI 10% (rolling 3yr)',
    'P4_rol': 'SHI 15% (rolling 3yr)',
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
valid   = meta['ticker'].tolist()
tickers = [t for t in valid
           if t in lr.columns and t in ph.columns]
N = len(tickers)

common = lr.index.intersection(ph.index)
lr = lr.loc[common, tickers]
ph = ph.loc[common, tickers]

ret = np.expm1(lr)
ret = ret.fillna(0.0).replace([np.inf, -np.inf], 0.0)
ret = ret.clip(-0.50, 0.50)

dates = ret.index
T     = len(dates)

print(f"  N={N} | T={T}")

# ─────────────────────────────────────────────────────────────
# 2. PRÉ-COMPUTAR UNWRAP
# ─────────────────────────────────────────────────────────────
print("Pré-computando fases...")
phases_unwr = np.unwrap(ph.values, axis=0)
omega_inst  = np.diff(phases_unwr, axis=0)  # (T-1, N)

ticker_to_idx = {t: i for i, t in enumerate(tickers)}

# ─────────────────────────────────────────────────────────────
# 3. CARREGAR BASELINE
# ─────────────────────────────────────────────────────────────
print("Carregando baseline (expanding)...")
baseline = pd.read_csv(
    os.path.join(FRL_DIR, 'portfolio_returns.csv'),
    index_col=0, parse_dates=True)

# ─────────────────────────────────────────────────────────────
# 4. SIMULAR PORTFÓLIOS COM ROLLING WINDOW
# ─────────────────────────────────────────────────────────────
print(f"Simulando portfólios rolling {WIN_ROLL}d...")

rebal_points = list(range(MIN_HIST, T - 1, REBAL_FREQ))
port_daily   = {f'P{i}_rol': [] for i in [2, 3, 4]}
port_dates   = []

for step, t_idx in enumerate(rebal_points):
    if step % 10 == 0:
        print(f"  [{step+1:3d}/{len(rebal_points)}]"
              f" {dates[t_idx].date()}", end='\r')

    # SHI rolling: apenas os últimos WIN_ROLL dias
    win_start = max(0, t_idx - WIN_ROLL)
    omega_i   = omega_inst[win_start:t_idx-1,
                            :].mean(axis=0)
    omega_bar = omega_i.mean()
    shi       = np.abs(omega_i - omega_bar) / omega_bar
    shi_s     = pd.Series(shi, index=tickers)

    # Pesos por threshold
    weights = {}
    for x in X_VALUES:
        n_sel = max(2, int(np.ceil(N * x)))
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
        port_daily['P2_rol'].append(
            float(weights[0.05] @ r_day))
        port_daily['P3_rol'].append(
            float(weights[0.10] @ r_day))
        port_daily['P4_rol'].append(
            float(weights[0.15] @ r_day))

print(f"\n  Concluído.")

rol_df = pd.DataFrame(port_daily, index=port_dates)
rol_df.index = pd.to_datetime(rol_df.index)
rol_df = rol_df[~rol_df.index.duplicated(keep='first')]

# Combinar com baseline — alinhar no mesmo período
start   = rol_df.index[0]
base_s  = baseline.loc[start:,
                        ['P2','P3','P4']].copy()
base_s.columns = ['P2_exp','P3_exp','P4_exp']
all_ret = base_s.join(rol_df, how='inner')

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
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    rows   = {'label': label,
              'ann_return':  ann_r*100,
              'ann_vol':     ann_v*100,
              'sharpe':      sharpe,
              'max_drawdown':max_dd*100}
    for cr, (cs, ce) in CRISES.items():
        m    = (s.index >= cs) & (s.index <= ce)
        cr_s = s[m]
        if len(cr_s) > 0:
            c  = (1 + cr_s).cumprod()
            dd = ((c - c.cummax()) / c.cummax()).min()
            rows[f'{cr}_drawdown'] = dd * 100
        else:
            rows[f'{cr}_drawdown'] = np.nan
    return rows

rows = []
for k in ['P2_exp','P3_exp','P4_exp',
          'P2_rol','P3_rol','P4_rol']:
    rows.append(metrics(all_ret[k], LABELS[k]))
met_df = pd.DataFrame(rows)
met_df.index = ['P2_exp','P3_exp','P4_exp',
                'P2_rol','P3_rol','P4_rol']

print("\nTabela de robustez C:")
for suffix, tag in [('exp','Expanding'),
                     ('rol','Rolling 3yr')]:
    print(f"\n  {tag}:")
    print(f"  {'':30s} {'P2':>8} {'P3':>8} {'P4':>8}")
    print("  " + "-"*54)
    for col in ['ann_return','ann_vol','sharpe',
                'max_drawdown','GFC_drawdown',
                'COVID_drawdown']:
        row = f"  {col:30s}"
        for x in ['2','3','4']:
            row += f"{met_df.loc[f'P{x}_{suffix}',col]:>8.3f}"
        print(row)

# ─────────────────────────────────────────────────────────────
# 6. FIGURA
# ─────────────────────────────────────────────────────────────
cum  = (1 + all_ret).cumprod()
ymax = cum[['P2_exp','P3_exp','P4_exp',
            'P2_rol','P3_rol','P4_rol']].max().max()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Robustness Test C: Expanding vs. Rolling '
    '(756-day) Window for SHI Estimation\n'
    'Solid: expanding | Dashed: rolling 3yr',
    fontsize=11, fontweight='bold'
)

# Painel esquerdo: retorno acumulado P2
ax = axes[0]
ax.set_title('SHI top 5% — cumulative return',
             fontsize=9)
for k, ls in [('P2_exp','-'),('P2_rol','--')]:
    ax.plot(cum.index, cum[k],
            color=COLORS[k], linewidth=1.8,
            linestyle=ls, alpha=0.9,
            label=LABELS[k])
for cr, (cs, ce) in CRISES.items():
    ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
               alpha=0.07,
               color='red' if cr=='GFC' else 'orange')
    mid = pd.Timestamp(cs) + (
        pd.Timestamp(ce)-pd.Timestamp(cs))/2
    ax.text(mid, ymax*0.96, cr,
            ha='center', va='top',
            fontsize=7.5, color='gray')
ax.set_ylabel('Cumulative return (base = 1)',
              fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(3))
ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y'))
ax.tick_params(axis='x', rotation=30)

# Painel direito: Sharpe lado a lado
ax = axes[1]
ax.set_title('Sharpe ratio: expanding vs rolling',
             fontsize=9)
labels_s = ['SHI 5%', 'SHI 10%', 'SHI 15%']
x = np.arange(3)
w = 0.35
sh_exp = [met_df.loc[f'P{i}_exp','sharpe']
          for i in [2,3,4]]
sh_rol = [met_df.loc[f'P{i}_rol','sharpe']
          for i in [2,3,4]]
colors_exp = ['#AA2828','#1E7840','#AA5500']
colors_rol = ['#FF8080','#80C880','#FFB060']
ax.bar(x-w/2, sh_exp, w, color=colors_exp,
       alpha=0.8, label='Expanding')
ax.bar(x+w/2, sh_rol, w, color=colors_rol,
       alpha=0.8, label='Rolling 3yr')
for i, (ve, vr) in enumerate(zip(sh_exp, sh_rol)):
    ax.text(i-w/2, ve+0.01, f'{ve:.3f}',
            ha='center', va='bottom', fontsize=8)
    ax.text(i+w/2, vr+0.01, f'{vr:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels_s)
ax.set_ylabel('Sharpe Ratio', fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(ROB_DIR,
            'fig_robustness_C.png'),
            dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_robustness_C.png ✓")

# ─────────────────────────────────────────────────────────────
# 7. SALVAR
# ─────────────────────────────────────────────────────────────
met_df.to_csv(os.path.join(ROB_DIR,
              'robustness_C_metrics.csv'))
all_ret.to_csv(os.path.join(ROB_DIR,
               'robustness_C_returns.csv'))
print("  CSVs salvos ✓")

print("\n" + "="*55)
print("ROBUSTNESS C CONCLUÍDO")
print("="*55)
for x in ['2','3','4']:
    e = met_df.loc[f'P{x}_exp','sharpe']
    r = met_df.loc[f'P{x}_rol','sharpe']
    diff = r - e
    print(f"  P{x}: expanding={e:.3f} "
          f"rolling={r:.3f} diff={diff:+.3f}")
print("="*55)
