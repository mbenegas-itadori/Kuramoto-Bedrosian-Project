"""
FRL-10 · portfolio_simulation.py
==================================
Simulação de portfólios para comparação de performance:
  P0: Equal-weight benchmark (todos os 316 ativos)
  P1: Markowitz mínima variância (todos os 316 ativos)
  P2: SHI-filtered X=5%  + equal-weight
  P3: SHI-filtered X=10% + equal-weight
  P4: SHI-filtered X=15% + equal-weight

Design:
  - Período: Jan 2007 – Jan 2026
  - Rebalanceamento: a cada 63 dias (trimestral)
  - SHI: calculado sobre todo o histórico disponível
    até cada ponto de rebalanceamento (expanding window)
  - Sem look-ahead: SHI calculado apenas com dados
    disponíveis até t antes de alocar em t+1

Inputs:
  - log_returns_342_final.csv
  - phase_342_bk.csv
  - bk_validation_meta.csv

Outputs (Drive/frl/):
  - portfolio_returns.csv
  - portfolio_metrics.csv
  - portfolio_shi_history.csv
  - fig_FRL_portfolio_cumret.png
  - fig_FRL_portfolio_drawdown.png
  - fig_FRL_portfolio_metrics.png
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

DRIVE_DIR = '/content/drive/MyDrive/ste_matrices_corrected'
FRL_DIR   = os.path.join(DRIVE_DIR, 'frl')
os.makedirs(FRL_DIR, exist_ok=True)

FIG_DPI     = 300
REBAL_FREQ  = 63       # dias úteis (~trimestral)
X_VALUES    = [0.05, 0.10, 0.15]
MIN_HIST    = 504      # mínimo de história (~2 anos) — inicia ~Jan 2009
RF_ANNUAL   = 0.02     # taxa livre de risco anual aproximada

# Períodos de crise para análise específica
# GFC: pico da crise após Lehman (dentro da janela efetiva)
# COVID: queda abrupta fev-mar 2020
CRISES = {
    'GFC':   ('2009-01-02', '2009-06-30'),
    'COVID': ('2020-02-19', '2020-03-23'),
}

# Paleta
COLORS = {
    'P0': '#888888',   # equal-weight
    'P1': '#1464A0',   # Markowitz
    'P2': '#AA2828',   # SHI 5%
    'P3': '#1E7840',   # SHI 10%
    'P4': '#AA5500',   # SHI 15%
}
LABELS = {
    'P0': 'Equal-weight (benchmark)',
    'P1': 'Min-variance (Markowitz)',
    'P2': 'SHI-filtered (top 5%)',
    'P3': 'SHI-filtered (top 10%)',
    'P4': 'SHI-filtered (top 15%)',
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

# Alinhar universo: apenas tickers presentes em ambos
valid = meta['ticker'].tolist()
tickers = [t for t in valid if t in lr.columns
           and t in ph.columns]
N = len(tickers)

# Alinhar índices: usar apenas datas presentes em ambos
common_dates = lr.index.intersection(ph.index)
lr  = lr.loc[common_dates, tickers]
ph  = ph.loc[common_dates, tickers]

# Converter log-retornos para retornos simples
ret = np.expm1(lr)  # r = e^(log_ret) - 1

# ── CONTROLE DE QUALIDADE DOS DADOS ──────────────────────────
print("\n  Verificando qualidade dos dados...")

# 1. NaN
n_nan = ret.isna().sum().sum()
print(f"  NaN em ret       : {n_nan}")
if n_nan > 0:
    ret = ret.fillna(0.0)
    print(f"  → NaN substituídos por 0")

# 2. Inf
n_inf = np.isinf(ret.values).sum()
print(f"  Inf em ret       : {n_inf}")
if n_inf > 0:
    ret = ret.replace([np.inf, -np.inf], 0.0)
    print(f"  → Inf substituídos por 0")

# 3. Outliers extremos — winsorize a 1% e 99% por ativo
#    Retornos diários > |50%| são quase certamente erros de dados
OUTLIER_CAP = 0.50
n_outliers = (ret.abs() > OUTLIER_CAP).sum().sum()
print(f"  |ret| > {OUTLIER_CAP:.0%}    : {n_outliers} observações")
if n_outliers > 0:
    ret = ret.clip(lower=-OUTLIER_CAP, upper=OUTLIER_CAP)
    print(f"  → Winsorizado em ±{OUTLIER_CAP:.0%}")

# 4. Relatório por ativo (top 5 mais problemáticos)
vol_by_asset = ret.std() * np.sqrt(252)
top5_vol = vol_by_asset.nlargest(5)
print(f"\n  Top 5 ativos por volatilidade anualizada:")
for tk, v in top5_vol.items():
    print(f"    {tk:8s}: {v:.1%}")

# 5. Verificar consistência P0 (equal-weight)
ew_ret = ret.mean(axis=1)
print(f"\n  P0 equal-weight stats (pré-simulação):")
print(f"    Vol anualizada  : {ew_ret.std()*np.sqrt(252):.1%}")
print(f"    Max retorno dia : {ew_ret.max():.4f}")
print(f"    Min retorno dia : {ew_ret.min():.4f}")

dates = ret.index
T     = len(dates)

print(f"  N ativos  : {N}")
print(f"  T dias    : {T}")
print(f"  Período   : {dates[0].date()} → {dates[-1].date()}")

# ─────────────────────────────────────────────────────────────
# 2. FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────

def compute_shi_expanding(omega_inst_full, end_idx):
    """
    Calcula SHI para cada ativo usando dados de [0, end_idx].
    Expanding window — sem look-ahead.
    omega_inst_full: pré-computado fora do loop (T-1, N)
    """
    omega_i  = omega_inst_full[:end_idx-1, :].mean(axis=0)
    omega_bar = omega_i.mean()
    shi       = np.abs(omega_i - omega_bar) / omega_bar
    return shi, omega_bar


def markowitz_min_var(returns_window, max_weight=0.10):
    """
    Portfólio de mínima variância com restrições:
      - pesos somam 1
      - cada peso entre 0 e max_weight (long-only)
      - sem short selling
    Retorna vetor de pesos.
    """
    n = returns_window.shape[1]
    cov = returns_window.cov().values
    # Regularização para evitar matriz singular
    cov += np.eye(n) * 1e-8

    def portfolio_var(w):
        return w @ cov @ w

    constraints = [{'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(portfolio_var, w0,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'maxiter': 1000,
                               'ftol': 1e-10})

    if result.success:
        w = np.maximum(result.x, 0)
        return w / w.sum()
    else:
        return np.ones(n) / n


def compute_metrics(port_ret, label=''):
    """Métricas de performance de uma série de retornos diários."""
    cum    = (1 + port_ret).cumprod()
    total  = cum.iloc[-1] - 1
    n_days = len(port_ret)
    ann_ret = (1 + total) ** (252 / n_days) - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe  = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan

    # Maximum drawdown
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd   = drawdown.min()

    return {
        'label':         label,
        'total_return':  total * 100,
        'ann_return':    ann_ret * 100,
        'ann_vol':       ann_vol * 100,
        'sharpe':        sharpe,
        'max_drawdown':  max_dd * 100,
    }


def crisis_metrics(port_ret, crisis_name, start, end):
    """Métricas durante período de crise.
    Retorna NaN explícito se o período não estiver na série.
    """
    mask = (port_ret.index >= start) & (port_ret.index <= end)
    cr   = port_ret[mask]
    if len(cr) == 0:
        return {
            f'{crisis_name}_return':   np.nan,
            f'{crisis_name}_vol':      np.nan,
            f'{crisis_name}_drawdown': np.nan,
        }
    cum   = (1 + cr).cumprod()
    total = cum.iloc[-1] - 1
    vol   = cr.std() * np.sqrt(252)
    dd    = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        f'{crisis_name}_return':   total * 100,
        f'{crisis_name}_vol':      vol * 100,
        f'{crisis_name}_drawdown': dd * 100,
    }

# ─────────────────────────────────────────────────────────────
# 3. SIMULAÇÃO ROLLING
# ─────────────────────────────────────────────────────────────
print(f"\nSimulando portfólios (rebalanceamento a cada "
      f"{REBAL_FREQ} dias)...")

# Pré-computar unwrap completo uma vez (fix ponto 4)
print("  Pré-computando fases unwrapped...")
phases_unwr_full = np.unwrap(ph.values, axis=0)
omega_inst_full  = np.diff(phases_unwr_full, axis=0)  # (T-1, N)

# Dicionário de lookup ticker→índice (fix ponto 1)
ticker_to_idx = {t: i for i, t in enumerate(tickers)}

# Identificar pontos de rebalanceamento
rebal_points = list(range(MIN_HIST, T - 1, REBAL_FREQ))
print(f"  {len(rebal_points)} pontos de rebalanceamento")
print(f"  Primeiro: {dates[rebal_points[0]].date()}")
print(f"  Último  : {dates[rebal_points[-1]].date()}")

# Acumular retornos diários no loop (fix ponto 2 — evita ~1.5GB RAM)
port_daily  = {k: [] for k in ['P0','P1','P2','P3','P4']}
port_dates  = []

# Armazenar SHI por data de rebalanceamento
shi_history = pd.DataFrame(index=[dates[r]
                                   for r in rebal_points],
                            columns=tickers, dtype=float)

for step, t_idx in enumerate(rebal_points):
    t_date = dates[t_idx]

    if step % 5 == 0:
        print(f"  [{step+1:3d}/{len(rebal_points)}] "
              f"{t_date.date()}", end='\r')

    # SHI expandido até t_idx (fix ponto 4: usa array pré-computado)
    shi, omega_bar = compute_shi_expanding(
        omega_inst_full, t_idx)
    shi_series = pd.Series(shi, index=tickers)
    shi_history.loc[t_date] = shi

    # Janela de retornos para Markowitz
    ret_window = ret.iloc[max(0, t_idx-756):t_idx]

    # ── P0: Equal-weight ─────────────────────────────────
    w_ew = np.ones(N) / N

    # ── P1: Markowitz mínima variância ───────────────────
    w_mv = markowitz_min_var(ret_window)

    # ── P2, P3, P4: SHI-filtered equal-weight ────────────
    shi_weights = {}
    for x in X_VALUES:
        n_select = max(2, int(np.ceil(N * x)))
        top_tickers = shi_series.nlargest(n_select).index
        w_shi = np.zeros(N)
        for tk in top_tickers:
            w_shi[ticker_to_idx[tk]] = 1.0  # fix ponto 1
        w_shi /= w_shi.sum()
        shi_weights[x] = w_shi

    # Período de aplicação: até próximo rebalanceamento
    if step < len(rebal_points) - 1:
        next_t = rebal_points[step + 1]
    else:
        next_t = T

    # Calcular retornos diários e acumular (fix ponto 2)
    weights_map = {
        'P0': w_ew,
        'P1': w_mv,
        'P2': shi_weights[0.05],
        'P3': shi_weights[0.10],
        'P4': shi_weights[0.15],
    }
    for day_idx in range(t_idx, next_t):
        day = dates[day_idx]
        r_day = ret.iloc[day_idx].values
        port_dates.append(day)
        for k, w in weights_map.items():
            port_daily[k].append(float(w @ r_day))

print(f"\n  Simulação concluída.")

# ─────────────────────────────────────────────────────────────
# 4. CALCULAR RETORNOS DOS PORTFÓLIOS
# ─────────────────────────────────────────────────────────────
print("\nCalculando retornos dos portfólios...")

# Construir DataFrame a partir das listas acumuladas no loop
port_df = pd.DataFrame(port_daily, index=port_dates)
port_df.index = pd.to_datetime(port_df.index)

# Verificar duplicatas (fix ponto E)
n_dup = port_df.index.duplicated().sum()
if n_dup > 0:
    print(f"  ⚠ {n_dup} datas duplicadas — removendo primeiras ocorrências")
port_df = port_df[~port_df.index.duplicated(keep='first')]

start_date = port_df.index[0]
print(f"  Período efetivo: {start_date.date()} → "
      f"{port_df.index[-1].date()}")
for key in ['P0','P1','P2','P3','P4']:
    print(f"  {key}: {LABELS[key][:35]}")

# ─────────────────────────────────────────────────────────────
# 5. MÉTRICAS DE PERFORMANCE
# ─────────────────────────────────────────────────────────────
print("\nCalculando métricas...")

metrics_list = []
for key in ['P0', 'P1', 'P2', 'P3', 'P4']:
    m = compute_metrics(port_df[key], LABELS[key])
    # Métricas de crise
    for crisis, (cs, ce) in CRISES.items():
        cm = crisis_metrics(port_df[key], crisis, cs, ce)
        m.update(cm)
    metrics_list.append(m)

metrics_df = pd.DataFrame(metrics_list)
metrics_df.index = ['P0', 'P1', 'P2', 'P3', 'P4']

print("\nTabela de performance:")
print(f"{'':30s} {'P0':>8} {'P1':>8} "
      f"{'P2':>8} {'P3':>8} {'P4':>8}")
print("-" * 70)
for col, fmt in [
    ('ann_return',       '{:>8.2f}'),
    ('ann_vol',          '{:>8.2f}'),
    ('sharpe',           '{:>8.3f}'),
    ('max_drawdown',     '{:>8.2f}'),
    ('GFC_drawdown',     '{:>8.2f}'),
    ('COVID_drawdown',   '{:>8.2f}'),
]:
    row = f"{col:30s}"
    for key in ['P0','P1','P2','P3','P4']:
        val = metrics_df.loc[key, col]
        row += fmt.format(val) if pd.notna(val) else '     N/A'
    print(row)

# ─────────────────────────────────────────────────────────────
# 6. SALVAR CSVs
# ─────────────────────────────────────────────────────────────
print("\nSalvando CSVs...")

port_df.to_csv(
    os.path.join(FRL_DIR, 'portfolio_returns.csv'))
print("  portfolio_returns.csv ✓")

metrics_df.to_csv(
    os.path.join(FRL_DIR, 'portfolio_metrics.csv'))
print("  portfolio_metrics.csv ✓")

shi_history.to_csv(
    os.path.join(FRL_DIR, 'portfolio_shi_history.csv'))
print("  portfolio_shi_history.csv ✓")

# ─────────────────────────────────────────────────────────────
# 7. FIGURAS
# ─────────────────────────────────────────────────────────────
print("\nGerando figuras...")

cum_ret = (1 + port_df).cumprod()

# ── Fig 1: Retorno acumulado — dois painéis ──────────────────
# Painel A: todos os portfólios em escala log
# Painel B: sem P0 em escala linear (para legibilidade)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    'Cumulative Portfolio Performance\n'
    'SHI-filtered vs. traditional strategies '
    '(S\\&P 500 universe, 2009--2026)',
    fontsize=11, fontweight='bold'
)

# Painel A — escala log, todos os portfólios
ax1.set_title('(A) All portfolios — log scale', fontsize=9)
for key in ['P0', 'P1', 'P2', 'P3', 'P4']:
    lw = 2.0 if 'SHI' in LABELS[key] else 1.5
    ax1.plot(cum_ret.index, cum_ret[key],
             color=COLORS[key], linewidth=lw,
             alpha=0.9, label=LABELS[key])
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative return (log scale)', fontsize=9)
ax1.legend(fontsize=7.5, loc='upper left')
ax1.grid(alpha=0.3, which='both')
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Painel B — escala linear, sem P0
ax2.set_title('(B) Excluding equal-weight — linear scale',
              fontsize=9)
for key in ['P1', 'P2', 'P3', 'P4']:
    lw = 2.0 if 'SHI' in LABELS[key] else 1.5
    ax2.plot(cum_ret.index, cum_ret[key],
             color=COLORS[key], linewidth=lw,
             alpha=0.9, label=LABELS[key])
ax2.set_ylabel('Cumulative return (base = 1)', fontsize=9)
ax2.legend(fontsize=7.5, loc='upper left')
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Sombrear crises em ambos os painéis
# ymax calculado sobre os dados — robusto independente do backend
ymax_log = cum_ret[['P0','P1','P2','P3','P4']].max().max()
ymax_lin = cum_ret[['P1','P2','P3','P4']].max().max()

for ax, ymax in [(ax1, ymax_log), (ax2, ymax_lin)]:
    for crisis, (cs, ce) in CRISES.items():
        color = 'red' if crisis == 'GFC' else 'orange'
        ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
                   alpha=0.08, color=color)
        mid = pd.Timestamp(cs) + (
            pd.Timestamp(ce) - pd.Timestamp(cs)) / 2
        ax.text(mid, ymax * 0.92, crisis,
                ha='center', va='top',
                fontsize=7.5, color='gray')
    ax.set_xlabel('Date', fontsize=9)
    ax.tick_params(axis='x', rotation=30, labelsize=8)

plt.tight_layout()
fig.savefig(
    os.path.join(FRL_DIR, 'fig_FRL_portfolio_cumret.png'),
    dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_FRL_portfolio_cumret.png ✓")

# ── Fig 2: Drawdown ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

for key in ['P0', 'P1', 'P2', 'P3', 'P4']:
    roll_max = cum_ret[key].cummax()
    dd = (cum_ret[key] - roll_max) / roll_max * 100
    lw = 2.0 if 'SHI' in LABELS[key] else 1.2
    ax.plot(dd.index, dd,
            color=COLORS[key], linewidth=lw,
            alpha=0.85, label=LABELS[key])

for crisis, (cs, ce) in CRISES.items():
    color = 'red' if crisis == 'GFC' else 'orange'
    ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
               alpha=0.08, color=color)

# Anotar crises — ymin calculado sobre os dados
dd_all = pd.DataFrame({
    key: (cum_ret[key] - cum_ret[key].cummax())
         / cum_ret[key].cummax() * 100
    for key in ['P0','P1','P2','P3','P4']
})
ymin_dd = dd_all.min().min()

for crisis, (cs, ce) in CRISES.items():
    mid = pd.Timestamp(cs) + (pd.Timestamp(ce) -
                               pd.Timestamp(cs)) / 2
    ax.text(mid, ymin_dd * 0.08, crisis,
            ha='center', va='bottom',
            fontsize=8, color='gray')

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Drawdown (%)', fontsize=11)
ax.set_title(
    'Portfolio Drawdown\n'
    'SHI-filtered vs. traditional strategies',
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
fig.savefig(
    os.path.join(FRL_DIR, 'fig_FRL_portfolio_drawdown.png'),
    dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_FRL_portfolio_drawdown.png ✓")

# ── Fig 3: Tabela de métricas visual ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

rows = [
    ('Ann. Return (%)',     'ann_return',    '{:.2f}'),
    ('Ann. Volatility (%)', 'ann_vol',       '{:.2f}'),
    ('Sharpe Ratio',        'sharpe',        '{:.3f}'),
    ('Max Drawdown (%)',    'max_drawdown',  '{:.2f}'),
    ('GFC Drawdown (%)',    'GFC_drawdown',  '{:.2f}'),
    ('COVID Drawdown (%)',  'COVID_drawdown','{:.2f}'),
]

col_labels = ['Metric'] + [LABELS[k] for k in
                            ['P0','P1','P2','P3','P4']]
table_data = []
for row_label, col, fmt in rows:
    row = [row_label]
    for key in ['P0','P1','P2','P3','P4']:
        val = metrics_df.loc[key, col]
        row.append(fmt.format(val) if pd.notna(val)
                   else 'N/A')
    table_data.append(row)

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.8)

# Colorir cabeçalhos
for j, key in enumerate(['P0','P1','P2','P3','P4']):
    table[0, j+1].set_facecolor(COLORS[key])
    table[0, j+1].set_text_props(color='white',
                                  fontweight='bold')

ax.set_title(
    'Portfolio Performance Metrics\n'
    'Full period and crisis episodes',
    fontsize=11, fontweight='bold', pad=20
)
plt.tight_layout()
fig.savefig(
    os.path.join(FRL_DIR, 'fig_FRL_portfolio_metrics.png'),
    dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_FRL_portfolio_metrics.png ✓")

# ─────────────────────────────────────────────────────────────
# 8. RESUMO FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FRL-10 · PORTFOLIO SIMULATION CONCLUÍDA")
print("="*55)
print(f"  Período    : {port_df.index[0].date()} → "
      f"{port_df.index[-1].date()}")
print(f"  Rebalanceamentos: {len(rebal_points)}")
print(f"\n  Sharpe ratios:")
for key in ['P0','P1','P2','P3','P4']:
    print(f"    {key} ({LABELS[key][:30]:30s}): "
          f"{metrics_df.loc[key,'sharpe']:.3f}")
print(f"\n  GFC drawdown:")
for key in ['P0','P1','P2','P3','P4']:
    print(f"    {key}: {metrics_df.loc[key,'GFC_drawdown']:.2f}%")
print(f"\n  COVID drawdown:")
for key in ['P0','P1','P2','P3','P4']:
    print(f"    {key}: "
          f"{metrics_df.loc[key,'COVID_drawdown']:.2f}%")
print(f"\n  Outputs em: {FRL_DIR}")
for f in ['portfolio_returns.csv',
          'portfolio_metrics.csv',
          'portfolio_shi_history.csv',
          'fig_FRL_portfolio_cumret.png',
          'fig_FRL_portfolio_drawdown.png',
          'fig_FRL_portfolio_metrics.png']:
    print(f"    - {f}")
print("="*55)
