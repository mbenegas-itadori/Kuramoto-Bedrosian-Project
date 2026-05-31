"""
FRL-03 · epsilon_rolling.py
=============================
Análise rolling de εᵢ e bootstrap para a note FRL.

Análise 1 — εᵢ rolling:
  Janela: 756 dias (~3 anos), step: 63 dias (~1 trimestre)
  ω̄(t) calculado na mesma janela rolling (Opção A)
  Produz série temporal εᵢ(t) por ativo

Análise 2 — Bootstrap:
  Reamostras com reposição de 316 ativos
  Distribução do número esperado de ativos com εᵢ > 0.20
  Extrapolação para S&P 500 completo e Russell 1000

Inputs:
  - phase_342_bk.csv
  - bk_validation_meta.csv (εᵢ período completo)
  - gics_sectors.json

Outputs (no Drive/frl/):
  - epsilon_rolling.csv      : εᵢ(t) por ativo e janela
  - epsilon_rolling_meta.csv : estatísticas por ativo
  - bootstrap_results.csv    : distribuição bootstrap
  - fig_FRL4_rolling_heatmap.png
  - fig_FRL5_rolling_selected.png
  - fig_FRL6_bootstrap.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
from google.colab import drive

# ─────────────────────────────────────────────────────────────
# 0. CONFIGURAÇÕES
# ─────────────────────────────────────────────────────────────
drive.mount('/content/drive')

DRIVE_DIR  = '/content/drive/MyDrive/ste_matrices_corrected'
FRL_DIR    = os.path.join(DRIVE_DIR, 'frl')
os.makedirs(FRL_DIR, exist_ok=True)

THRESHOLD  = 0.20
WIN        = 756    # ~3 anos de dias úteis
STEP       = 63     # ~1 trimestre
N_BOOT     = 2000   # amostras bootstrap
SEED       = 42
FIG_DPI    = 300

np.random.seed(SEED)

COLOR_HIGH = '#AA2828'
COLOR_LOW  = '#AAAAAA'
COLOR_DYNA = '#AA5500'
COLOR_P1   = '#1464A0'
COLOR_P2   = '#1E7840'

# Ativos de interesse para análise detalhada
FOCUS_ASSETS = ['BANFP', 'BCAL', 'ARTNA', 'AIFF', 'AN',
                'BHRB', 'ASG', 'AXON', 'BLFS', 'BCDA']

INTERPRETATIONS = {
    'BANFP': 'Preferred stock',
    'BCAL':  'Regional bank',
    'ARTNA': 'Regulated utility',
    'AIFF':  'Neurotechnology',
    'AN':    'Auto dealer',
    'BLFS':  'Small-cap biotech',
    'BCDA':  'Small-cap biotech',
    'AXON':  'Secular growth',
    'BHRB':  'Digital transition (T2)',
    'ASG':   'Rate regime shift (T2)',
}

# ─────────────────────────────────────────────────────────────
# 1. CARREGAR DADOS
# ─────────────────────────────────────────────────────────────
print("Carregando dados...")

df_phases = pd.read_csv(
    'phase_342_bk.csv',
    index_col=0, parse_dates=True
)
phases  = df_phases.values
dates   = df_phases.index
tickers = df_phases.columns.tolist()
N, T    = len(tickers), len(dates)

meta = pd.read_csv(
    os.path.join(DRIVE_DIR, 'bk_validation_meta.csv')
)
meta['epsilon'] = meta['error_pct'] / 100

with open(os.path.join(DRIVE_DIR, 'gics_sectors.json')) as f:
    gics = json.load(f)
meta['sector'] = meta['ticker'].map(gics).fillna('Unknown')

print(f"  N={N} ativos | T={T} dias")
print(f"  Janela rolling: {WIN} dias | step: {STEP} dias")

# ─────────────────────────────────────────────────────────────
# 2. ANÁLISE 1 — εᵢ ROLLING (Opção A: ω̄ local)
# ─────────────────────────────────────────────────────────────
print("\n--- ANÁLISE 1: εᵢ ROLLING ---")

# Definir janelas
window_starts = list(range(0, T - WIN, STEP))
window_ends   = [s + WIN for s in window_starts]
window_dates  = [dates[e - 1] for e in window_ends]
n_windows     = len(window_starts)

print(f"  {n_windows} janelas rolling")
print(f"  De: {window_dates[0].date()} → {window_dates[-1].date()}")

# Pré-computar fases unwrapped (vetorizado)
print("  Unwrapping fases (vetorizado)...")
phases_unwr = np.unwrap(phases, axis=0)  # (T, N)
omega_inst  = np.diff(phases_unwr, axis=0)  # (T-1, N)

# Calcular εᵢ(t) para cada janela
print("  Calculando εᵢ rolling...")
eps_rolling = np.zeros((n_windows, N))

for w, (s, e) in enumerate(zip(window_starts, window_ends)):
    # omega médio de cada ativo na janela [s, e)
    omega_w   = omega_inst[s:e-1, :].mean(axis=0)  # (N,)
    # ω̄ local (Opção A)
    omega_bar = omega_w.mean()
    # εᵢ local
    eps_rolling[w, :] = np.abs(omega_w - omega_bar) / omega_bar

    if w % 10 == 0:
        n_above = (eps_rolling[w, :] > THRESHOLD).sum()
        print(f"  [{w+1:3d}/{n_windows}] {window_dates[w].date()} "
              f"— ε>{THRESHOLD}: {n_above} ativos "
              f"| ω̄={omega_bar:.5f}")

print(f"\n  Concluído. Shape: {eps_rolling.shape}")

# ─────────────────────────────────────────────────────────────
# 3. ESTATÍSTICAS POR ATIVO
# ─────────────────────────────────────────────────────────────
print("\n--- ESTATÍSTICAS POR ATIVO ---")

rolling_meta = []
for i, ticker in enumerate(tickers):
    eps_i    = eps_rolling[:, i]
    pct_high = (eps_i > THRESHOLD).mean() * 100
    rolling_meta.append({
        'ticker':      ticker,
        'eps_full':    meta.set_index('ticker').loc[ticker, 'epsilon']
                       if ticker in meta['ticker'].values else np.nan,
        'eps_roll_mean': eps_i.mean(),
        'eps_roll_std':  eps_i.std(),
        'eps_roll_max':  eps_i.max(),
        'pct_above':     pct_high,
        'sector':        gics.get(ticker, 'Unknown'),
        'interpretation': INTERPRETATIONS.get(ticker, ''),
    })

rolling_meta_df = pd.DataFrame(rolling_meta)\
    .sort_values('pct_above', ascending=False)

print(f"\nTop 15 ativos por % do tempo com ε>{THRESHOLD}:")
print(f"{'Ticker':8s} {'ε_full':>8} {'ε_mean':>8} "
      f"{'ε_max':>8} {'pct>0.20':>10}")
print("-"*48)
for _, r in rolling_meta_df.head(15).iterrows():
    print(f"{r['ticker']:8s} {r['eps_full']:>8.3f} "
          f"{r['eps_roll_mean']:>8.3f} "
          f"{r['eps_roll_max']:>8.3f} "
          f"{r['pct_above']:>9.1f}%")

# ─────────────────────────────────────────────────────────────
# 4. ANÁLISE 2 — BOOTSTRAP
# ─────────────────────────────────────────────────────────────
print(f"\n--- ANÁLISE 2: BOOTSTRAP (N_BOOT={N_BOOT}) ---")

# εᵢ do período completo para bootstrap
eps_full = meta.set_index('ticker').loc[
    tickers, 'epsilon'].values

# Bootstrap: resample de N ativos com reposição
n_above_boot = np.zeros(N_BOOT, dtype=int)
pct_above_boot = np.zeros(N_BOOT)

for b in range(N_BOOT):
    idx          = np.random.choice(N, N, replace=True)
    eps_boot     = eps_full[idx]
    n_above_boot[b]  = (eps_boot > THRESHOLD).sum()
    pct_above_boot[b] = (eps_boot > THRESHOLD).mean()

# Estatísticas bootstrap
n_obs       = (eps_full > THRESHOLD).sum()
pct_obs     = n_obs / N * 100
pct_mean    = pct_above_boot.mean() * 100
pct_ci_low  = np.percentile(pct_above_boot, 2.5) * 100
pct_ci_high = np.percentile(pct_above_boot, 97.5) * 100

print(f"  Observado       : {n_obs}/{N} ({pct_obs:.2f}%)")
print(f"  Bootstrap média : {pct_mean:.2f}%")
print(f"  IC 95%          : [{pct_ci_low:.2f}%, {pct_ci_high:.2f}%]")

# Extrapolação para diferentes universos
print(f"\n  Extrapolação (prevalência = {pct_mean:.2f}%):")
universes = [
    ('S&P 500 completo',  500),
    ('Russell 1000',     1000),
    ('Russell 3000',     3000),
    ('US equity market', 8000),
]
for label, size in universes:
    n_exp    = pct_mean / 100 * size
    n_lo     = pct_ci_low / 100 * size
    n_hi     = pct_ci_high / 100 * size
    print(f"    {label:22s} ({size:5d}): "
          f"~{n_exp:.0f} [{n_lo:.0f}–{n_hi:.0f}]")

# ─────────────────────────────────────────────────────────────
# 5. SALVAR CSVs
# ─────────────────────────────────────────────────────────────
print("\nSalvando CSVs...")

# epsilon_rolling.csv — matriz completa (n_windows × N)
rolling_df = pd.DataFrame(
    eps_rolling,
    index=[d.strftime('%Y-%m-%d') for d in window_dates],
    columns=tickers
)
rolling_df.to_csv(os.path.join(FRL_DIR, 'epsilon_rolling.csv'))
print("  epsilon_rolling.csv ✓")

# epsilon_rolling_meta.csv
rolling_meta_df.to_csv(
    os.path.join(FRL_DIR, 'epsilon_rolling_meta.csv'),
    index=False)
print("  epsilon_rolling_meta.csv ✓")

# bootstrap_results.csv
boot_df = pd.DataFrame({
    'n_above':   n_above_boot,
    'pct_above': pct_above_boot * 100,
})
boot_df.to_csv(
    os.path.join(FRL_DIR, 'bootstrap_results.csv'),
    index=False)
print("  bootstrap_results.csv ✓")

# ─────────────────────────────────────────────────────────────
# 6. FIGURAS
# ─────────────────────────────────────────────────────────────
print("\nGerando figuras...")

dates_roll = pd.to_datetime(
    [d.strftime('%Y-%m-%d') for d in window_dates])

# ── FIG FRL-4: Line plot εᵢ(t) top ativos ───────────────────
# Selecionar top 8 por pct_above para o line plot
top8 = rolling_meta_df.head(8)['ticker'].tolist()

fig, ax = plt.subplots(figsize=(14, 6))

# Paleta discreta para 8 linhas
palette = [
    '#AA2828', '#1464A0', '#1E7840', '#AA5500',
    '#6600AA', '#008080', '#8B4513', '#444444'
]

for i, ticker in enumerate(top8):
    idx   = tickers.index(ticker)
    eps_t = eps_rolling[:, idx]
    pct   = (eps_t > THRESHOLD).mean() * 100
    label = (f"{ticker} — "
             f"{INTERPRETATIONS.get(ticker, '')[:22]} "
             f"({pct:.0f}%)")
    color = COLOR_DYNA if ticker in ['BHRB','ASG'] \
            else palette[i]
    lw    = 2.0 if pct >= 80 else 1.2
    alpha = 0.9 if pct >= 80 else 0.7

    ax.plot(dates_roll, eps_t,
            color=color, linewidth=lw,
            alpha=alpha, label=label)

# Threshold
ax.axhline(THRESHOLD, color='black', linestyle='--',
           linewidth=1.2, alpha=0.7,
           label=f'Threshold $\\varepsilon^*={THRESHOLD}$')

# Área acima do threshold (fundo suave)
ax.axhspan(THRESHOLD, eps_rolling[:, [
    tickers.index(t) for t in top8
]].max() + 0.05,
           alpha=0.04, color='red')

# Eventos históricos
events = {
    'GFC 2008–09': '2009-06-01',
    'COVID 2020':  '2020-06-01',
}
ymax = eps_rolling[:, [tickers.index(t)
                        for t in top8]].max() + 0.05
for label, date in events.items():
    target = pd.Timestamp(date)
    ax.axvline(target, color='gray', linestyle=':',
               linewidth=1, alpha=0.7)
    ax.text(target, ymax * 0.97, label,
            fontsize=7, color='gray',
            ha='center', va='top')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('$\\varepsilon_i(t)$', fontsize=11)
ax.set_title(
    'Rolling Spectral Heterogeneity Index — Top 8 Assets\n'
    f'Window={WIN}d (~3yr), Step={STEP}d | '
    f'Reference: local $\\bar{{\\omega}}(t)$ (Option A)\n'
    f'% in parentheses = time above threshold $\\varepsilon^*={THRESHOLD}$',
    fontsize=10
)
ax.legend(fontsize=8, loc='upper left',
          framealpha=0.85, ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(-0.02, ymax)
plt.tight_layout()
fig.savefig(os.path.join(FRL_DIR, 'fig_FRL4_rolling_lineplot.png'),
            dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_FRL4_rolling_lineplot.png ✓")

# ── FIG FRL-5: εᵢ(t) para ativos selecionados ────────────────
focus_in_data = [t for t in FOCUS_ASSETS if t in tickers]
n_focus = len(focus_in_data)
ncols   = 2
nrows   = (n_focus + 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(14, 2.8 * nrows),
                          sharex=True)
axes = axes.flatten()
fig.suptitle(
    'Rolling $\\varepsilon_i(t)$ — Selected Assets\n'
    f'Window={WIN}d, Step={STEP}d | '
    f'Reference: local $\\bar{{\\omega}}(t)$ (Option A)',
    fontsize=10, fontweight='bold'
)

for ax, ticker in zip(axes, focus_in_data):
    idx   = tickers.index(ticker)
    eps_t = eps_rolling[:, idx]
    color = COLOR_HIGH if ticker not in ['BHRB','ASG'] \
            else COLOR_DYNA

    ax.plot(dates_roll, eps_t,
            color=color, linewidth=1.2, alpha=0.85)
    ax.fill_between(dates_roll, THRESHOLD, eps_t,
                    where=eps_t > THRESHOLD,
                    alpha=0.2, color=color)
    ax.axhline(THRESHOLD, color='gray', linestyle='--',
               linewidth=1, alpha=0.7)

    # % do tempo acima do threshold
    pct = (eps_t > THRESHOLD).mean() * 100
    ax.set_title(
        f'{ticker} — {INTERPRETATIONS.get(ticker,"")} '
        f'({pct:.0f}% above threshold)',
        fontsize=7.5
    )
    ax.set_ylabel('$\\varepsilon_i(t)$', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, max(eps_t.max() + 0.05, 0.35))

# Ocultar eixos extras
for ax in axes[n_focus:]:
    ax.set_visible(False)

# Formatar eixo X
for ax in axes[max(0, n_focus-ncols):]:
    ax.tick_params(axis='x', rotation=30, labelsize=7)

plt.tight_layout()
fig.savefig(os.path.join(FRL_DIR, 'fig_FRL5_rolling_selected.png'),
            dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_FRL5_rolling_selected.png ✓")

# ── FIG FRL-6: Bootstrap ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'Bootstrap Estimation of Structurally Decoupled Assets\n'
    f'$N_{{boot}}={N_BOOT}$ resamples | '
    f'Threshold $\\varepsilon^*={THRESHOLD}$',
    fontsize=11, fontweight='bold'
)

# Painel 1: distribuição bootstrap de n_above
ax = axes[0]
ax.hist(n_above_boot, bins=range(n_above_boot.min(),
                                  n_above_boot.max()+2),
        color=COLOR_HIGH, alpha=0.7, edgecolor='white')
ax.axvline(n_obs, color='black', linestyle='--',
           linewidth=1.5, label=f'Observed: {n_obs}')
ax.axvline(np.percentile(n_above_boot, 2.5),
           color='gray', linestyle=':', linewidth=1)
ax.axvline(np.percentile(n_above_boot, 97.5),
           color='gray', linestyle=':', linewidth=1,
           label=f'95% CI: [{np.percentile(n_above_boot, 2.5):.0f},'
                 f'{np.percentile(n_above_boot, 97.5):.0f}]')
ax.set_xlabel(f'Number of assets with $\\varepsilon_i > {THRESHOLD}$',
              fontsize=10)
ax.set_ylabel('Bootstrap frequency', fontsize=10)
ax.set_title('Bootstrap distribution\n(N=316 resampled with replacement)',
             fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Painel 2: extrapolação
ax = axes[1]
sizes  = [316, 500, 1000, 3000, 8000]
labels = ['This\nstudy\n(316)',
          'S&P 500\n(500)',
          'Russell\n1000',
          'Russell\n3000',
          'US equity\n(~8000)']
n_exp  = [pct_mean/100 * s for s in sizes]
n_lo   = [pct_ci_low/100 * s for s in sizes]
n_hi   = [pct_ci_high/100 * s for s in sizes]
err_lo = [n_exp[i] - n_lo[i] for i in range(len(sizes))]
err_hi = [n_hi[i] - n_exp[i] for i in range(len(sizes))]

colors = [COLOR_HIGH if i == 0 else COLOR_LOW
          for i in range(len(sizes))]
ax.bar(range(len(sizes)), n_exp,
       color=colors, alpha=0.7, edgecolor='white')
ax.errorbar(range(len(sizes)), n_exp,
            yerr=[err_lo, err_hi],
            fmt='none', color='black',
            capsize=5, linewidth=1.5)
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Expected number of decoupled assets', fontsize=10)
ax.set_title(
    f'Extrapolation (prevalence={pct_mean:.1f}%)\n'
    f'Error bars: 95% CI',
    fontsize=9
)
ax.grid(alpha=0.3, axis='y')

# Anotar valores
for i, (n, lo, hi) in enumerate(zip(n_exp, n_lo, n_hi)):
    ax.text(i, hi + 1, f'{n:.0f}',
            ha='center', va='bottom',
            fontsize=8, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FRL_DIR, 'fig_FRL6_bootstrap.png'),
            dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_FRL6_bootstrap.png ✓")

# ─────────────────────────────────────────────────────────────
# 7. RESUMO FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FRL-03 · EPSILON ROLLING CONCLUÍDO")
print("="*55)
print(f"  Janelas rolling     : {n_windows}")
print(f"  Ativos > {THRESHOLD} sempre: "
      f"{(rolling_meta_df['pct_above']==100).sum()}")
print(f"  Ativos > {THRESHOLD} nunca : "
      f"{(rolling_meta_df['pct_above']==0).sum()}")
print(f"\n  Bootstrap:")
print(f"  Prevalência obs.    : {pct_obs:.2f}%")
print(f"  IC 95%              : [{pct_ci_low:.2f}%, "
      f"{pct_ci_high:.2f}%]")
print(f"\n  Extrapolação:")
for label, size in universes:
    n_e = pct_mean/100 * size
    n_l = pct_ci_low/100 * size
    n_h = pct_ci_high/100 * size
    print(f"    {label:22s}: ~{n_e:.0f} [{n_l:.0f}–{n_h:.0f}]")
print(f"\n  Outputs em: {FRL_DIR}")
for f in ['epsilon_rolling.csv', 'epsilon_rolling_meta.csv',
          'bootstrap_results.csv',
          'fig_FRL4_rolling_heatmap.png',
          'fig_FRL5_rolling_selected.png',
          'fig_FRL6_bootstrap.png']:
    print(f"    - {f}")
print("="*55)
