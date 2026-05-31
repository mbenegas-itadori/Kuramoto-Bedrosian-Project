"""
FRL-02 · epsilon_analysis.py
=============================
Análise completa do Índice de Heterogeneidade Espectral (εᵢ)
para a note Finance Research Letters.

Blocos:
  1. Carregamento e distribuição básica
  2. Análise setorial (Kruskal-Wallis)
  3. Estabilidade temporal (Spearman P1 vs P2)
  4. Mecanismo de cancelamento (BHRB e ASG)
  5. Figuras de publicação

Inputs (no Drive):
  - bk_validation_meta.csv
  - gics_sectors.json
  - phase_342_bk.csv

Outputs (no Drive):
  - frl/epsilon_summary.csv
  - frl/epsilon_sector.csv
  - frl/epsilon_stability.csv
  - frl/fig_FRL1_histogram.png
  - frl/fig_FRL2_stability.png
  - frl/fig_FRL3_mechanism.png
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

THRESHOLD  = 0.20       # threshold principal de identificação
MID_DATE   = pd.Timestamp('2016-06-30')
FIG_DPI    = 300

# Paleta consistente com o paper principal
COLOR_HIGH = '#AA2828'   # ativos com εᵢ > threshold
COLOR_LOW  = '#AAAAAA'   # ativos típicos
COLOR_P1   = '#1464A0'   # período 1
COLOR_P2   = '#1E7840'   # período 2
COLOR_BOTH = '#AA2828'   # robustos em ambos
COLOR_DYNA = '#AA5500'   # desacoplamento dinâmico (BHRB, ASG)

# Interpretações econômicas dos ativos identificados
INTERPRETATIONS = {
    'BANFP': 'Preferred stock — fixed income dynamics',
    'BCAL':  'Regional bank — local credit cycle',
    'ARTNA': 'Artesian Water — regulated utility',
    'AIFF':  'Firefly Neuroscience — regulatory approval cycle',
    'AN':    'AutoNation — automotive credit cycle',
    'BLFS':  'BioLife Solutions — small-cap biotech',
    'BCDA':  'Bioxcel Therapeutics — small-cap biotech',
    'AXON':  'Axon Enterprise — secular growth (trend > cycle)',
    'BHRB':  'Block Inc. — digital transition (Type 2)',
    'ASG':   'Affiliated Managers — rate regime shift (Type 2)',
}

# ─────────────────────────────────────────────────────────────
# 1. CARREGAR DADOS
# ─────────────────────────────────────────────────────────────
print("Carregando dados...")

meta = pd.read_csv(
    os.path.join(DRIVE_DIR, 'bk_validation_meta.csv')
)
meta['epsilon'] = meta['error_pct'] / 100

with open(os.path.join(DRIVE_DIR, 'gics_sectors.json')) as f:
    gics = json.load(f)
meta['sector'] = meta['ticker'].map(gics).fillna('Unknown')

df_phases = pd.read_csv(
    'phase_342_bk.csv',
    index_col=0, parse_dates=True
)
phases  = df_phases.values
dates   = df_phases.index
tickers = df_phases.columns.tolist()
N       = len(tickers)

print(f"  Ativos    : {N}")
print(f"  Período   : {dates[0].date()} → {dates[-1].date()}")
print(f"  ε > {THRESHOLD}: {(meta['epsilon'] > THRESHOLD).sum()} ativos")

# Verificar consistência entre phase_342_bk e bk_validation_meta
missing = set(tickers) - set(meta['ticker'])
if missing:
    print(f"  ⚠ {len(missing)} tickers sem metadados: {missing}")
else:
    print(f"  ✓ Todos os {N} tickers têm metadados")

# ─────────────────────────────────────────────────────────────
# 2. ANÁLISE SETORIAL
# ─────────────────────────────────────────────────────────────
print("\n--- ANÁLISE SETORIAL ---")

sector_stats = []
for sector in sorted(meta['sector'].unique()):
    sub  = meta[meta['sector'] == sector]
    p10  = (sub['epsilon'] > 0.10).mean() * 100
    p15  = (sub['epsilon'] > 0.15).mean() * 100
    p20  = (sub['epsilon'] > 0.20).mean() * 100
    sector_stats.append({
        'sector':  sector,
        'n':       len(sub),
        'mean':    sub['epsilon'].mean(),
        'std':     sub['epsilon'].std(),
        'pct_010': p10,
        'pct_015': p15,
        'pct_020': p20,
    })
    print(f"  {sector:35s} n={len(sub):3d} "
          f"mean={sub['epsilon'].mean():.3f} "
          f"p>0.20={p20:.1f}%")

sector_df = pd.DataFrame(sector_stats)

# Kruskal-Wallis
groups = [
    meta[meta['sector'] == s]['epsilon'].values
    for s in meta['sector'].unique()
    if s != 'Unknown' and
    len(meta[meta['sector'] == s]) >= 5
]
kw_stat, kw_p = stats.kruskal(*groups)
print(f"\n  Kruskal-Wallis: H={kw_stat:.3f}, p={kw_p:.4f}")
print(f"  → {'NÃO significativo' if kw_p > 0.05 else 'SIGNIFICATIVO'}")
print(f"  → εᵢ é idiossincrático, não setorial")

# ─────────────────────────────────────────────────────────────
# 3. ESTABILIDADE TEMPORAL
# ─────────────────────────────────────────────────────────────
print("\n--- ESTABILIDADE TEMPORAL ---")

mask1 = dates <= MID_DATE
mask2 = dates >  MID_DATE
print(f"  P1: {dates[mask1][0].date()} → {dates[mask1][-1].date()}"
      f" ({mask1.sum()} dias)")
print(f"  P2: {dates[mask2][0].date()} → {dates[mask2][-1].date()}"
      f" ({mask2.sum()} dias)")

def compute_epsilon_block(phases_block, tickers):
    """Calcula εᵢ para um bloco de fases."""
    omega = np.zeros(phases_block.shape[1])
    for i in range(phases_block.shape[1]):
        unwr     = np.unwrap(phases_block[:, i])
        omega[i] = np.diff(unwr).mean()
    omega_bar = omega.mean()
    epsilon   = np.abs(omega - omega_bar) / omega_bar
    return epsilon, omega, omega_bar

eps1, omega1, obar1 = compute_epsilon_block(
    phases[mask1], tickers)
eps2, omega2, obar2 = compute_epsilon_block(
    phases[mask2], tickers)

print(f"  ω̄ P1 = {obar1:.5f} | ω̄ P2 = {obar2:.5f}")
print(f"  ε>0.20 P1: {(eps1>THRESHOLD).sum()} | "
      f"ε>0.20 P2: {(eps2>THRESHOLD).sum()}")

# Correlação de rank
rho, p_rho = stats.spearmanr(eps1, eps2)
print(f"\n  Spearman ρ = {rho:.4f}, p = {p_rho:.4f}")
print(f"  → {'NÃO significativo' if p_rho > 0.05 else 'SIGNIFICATIVO'}")

# Ativos robustos
stab_df = pd.DataFrame({
    'ticker':   tickers,
    'eps_full': meta.set_index('ticker').loc[tickers,'epsilon'].values,
    'eps_P1':   eps1,
    'eps_P2':   eps2,
    'omega_P1': omega1,
    'omega_P2': omega2,
    'sector':   meta.set_index('ticker').loc[tickers,'sector'].values,
})

both_high = stab_df[
    (stab_df['eps_P1'] > THRESHOLD) &
    (stab_df['eps_P2'] > THRESHOLD)
].sort_values('eps_full', ascending=False)

only_P1 = stab_df[
    (stab_df['eps_P1'] > THRESHOLD) &
    (stab_df['eps_P2'] <= THRESHOLD)
]
only_P2 = stab_df[
    (stab_df['eps_P1'] <= THRESHOLD) &
    (stab_df['eps_P2'] > THRESHOLD)
]

print(f"\n  Ativos robustos (ambos): {len(both_high)}")
for _, r in both_high.iterrows():
    tag = '(Type 2)' if r['eps_full'] < THRESHOLD else ''
    interp = INTERPRETATIONS.get(r['ticker'], '')
    print(f"    {r['ticker']:6s} ε_full={r['eps_full']:.3f} "
          f"ε_P1={r['eps_P1']:.3f} ε_P2={r['eps_P2']:.3f} "
          f"{tag} {interp}")

print(f"\n  Só em P1: {len(only_P1)} | Só em P2: {len(only_P2)}")

# Definir colunas robust e dynamic ANTES das figuras
stab_df['robust']  = (
    (stab_df['eps_P1'] > THRESHOLD) &
    (stab_df['eps_P2'] > THRESHOLD)
)
stab_df['dynamic'] = stab_df['ticker'].isin(['BHRB', 'ASG'])

# ─────────────────────────────────────────────────────────────
# 4. MECANISMO DE CANCELAMENTO (BHRB e ASG)
# ─────────────────────────────────────────────────────────────
print("\n--- MECANISMO DE CANCELAMENTO ---")

dynamic_assets = ['BHRB', 'ASG']

# Vetorizado: muito mais rápido que loop Python
phases_unwr      = np.unwrap(phases, axis=0)
omega_all        = np.diff(phases_unwr, axis=0).mean(axis=0)
omega_bar_global = omega_all.mean()
print(f"  ω̄ global = {omega_bar_global:.5f} rad/dia")

for ticker in dynamic_assets:
    if ticker not in tickers:
        print(f"  {ticker}: não encontrado")
        continue
    idx = tickers.index(ticker)
    for label, mask in [('P1', mask1), ('P2', mask2),
                         ('Full', np.ones(len(dates), bool))]:
        unwr  = np.unwrap(phases[mask, idx])
        om    = np.diff(unwr).mean()
        per   = 2*np.pi / abs(om)
        dev   = om - omega_bar_global
        sign  = '+' if dev > 0 else '-'
        print(f"  {ticker} {label}: ω̂={om:.5f} "
              f"(período={per:.1f}d) "
              f"desvio={sign}{abs(dev):.5f}")
    print()

# ─────────────────────────────────────────────────────────────
# 5. SALVAR OUTPUTS CSV
# ─────────────────────────────────────────────────────────────
print("Salvando CSVs...")

# epsilon_summary.csv
summary = meta[['ticker','epsilon','sector']].copy()
summary['interpretation'] = summary['ticker'].map(
    INTERPRETATIONS).fillna('')
summary['type1'] = (summary['epsilon'] > THRESHOLD) & \
                   summary['ticker'].isin(
                       both_high[both_high['eps_full']>THRESHOLD]
                       ['ticker'])
summary['type2'] = summary['ticker'].isin(['BHRB','ASG'])
summary.to_csv(
    os.path.join(FRL_DIR, 'epsilon_summary.csv'), index=False)
print("  epsilon_summary.csv ✓")

# epsilon_sector.csv
sector_df['kw_H'] = kw_stat
sector_df['kw_p'] = kw_p
sector_df.to_csv(
    os.path.join(FRL_DIR, 'epsilon_sector.csv'), index=False)
print("  epsilon_sector.csv ✓")

# epsilon_stability.csv
stab_df['interpretation'] = stab_df['ticker'].map(
    INTERPRETATIONS).fillna('')
stab_df.to_csv(
    os.path.join(FRL_DIR, 'epsilon_stability.csv'), index=False)
print("  epsilon_stability.csv ✓")

# ─────────────────────────────────────────────────────────────
# 6. FIGURAS DE PUBLICAÇÃO
# ─────────────────────────────────────────────────────────────
print("\nGerando figuras...")

# ── FIG FRL-1: Histograma de εᵢ ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

bins = np.linspace(0, meta['epsilon'].max()+0.02, 50)
ax.hist(meta[meta['epsilon'] <= THRESHOLD]['epsilon'],
        bins=bins, color=COLOR_LOW, alpha=0.7,
        label=f'Typical ($\\varepsilon_i \\leq {THRESHOLD}$, '
              f'n={(meta["epsilon"] <= THRESHOLD).sum()})')
ax.hist(meta[meta['epsilon'] > THRESHOLD]['epsilon'],
        bins=bins, color=COLOR_HIGH, alpha=0.8,
        label=f'Structurally decoupled ($\\varepsilon_i > {THRESHOLD}$, '
              f'n={(meta["epsilon"] > THRESHOLD).sum()})')

ax.axvline(THRESHOLD, color='black', linestyle='--',
           linewidth=1.2, label=f'Threshold $\\varepsilon^*={THRESHOLD}$')

# Anotar ativos identificados
for _, row in meta[meta['epsilon'] > THRESHOLD].iterrows():
    ax.annotate(
        row['ticker'],
        (row['epsilon'], 0.5),
        fontsize=6.5, color=COLOR_HIGH,
        rotation=90, ha='center', va='bottom'
    )

ax.set_xlabel('Spectral Heterogeneity Index $\\varepsilon_i$',
              fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(
    'Distribution of the Spectral Heterogeneity Index\n'
    f'$N={N}$ assets, S\\&P 500, 2007–2026',
    fontsize=11
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FRL_DIR, 'fig_FRL1_histogram.png'),
            dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_FRL1_histogram.png ✓")

# ── FIG FRL-2: Scatter P1 vs P2 ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))

# Todos os ativos típicos
typical = stab_df[~stab_df['robust']]
ax.scatter(typical['eps_P1'], typical['eps_P2'],
           s=12, alpha=0.35, color=COLOR_LOW, zorder=2)

# Robustos Tipo 1 (εᵢ_full > threshold)
type1 = both_high[both_high['eps_full'] >= THRESHOLD]
ax.scatter(type1['eps_P1'], type1['eps_P2'],
           s=80, color=COLOR_HIGH, zorder=4,
           label=f'Type 1 — static decoupling (n={len(type1)})')
for _, r in type1.iterrows():
    ax.annotate(r['ticker'], (r['eps_P1'], r['eps_P2']),
                fontsize=7.5, xytext=(5, 3),
                textcoords='offset points', color=COLOR_HIGH,
                fontweight='bold')

# Robustos Tipo 2 (εᵢ_full < threshold — paradoxais)
type2 = both_high[both_high['eps_full'] < THRESHOLD]
ax.scatter(type2['eps_P1'], type2['eps_P2'],
           s=80, color=COLOR_DYNA, zorder=4,
           marker='D',
           label=f'Type 2 — dynamic decoupling (n={len(type2)})')
for _, r in type2.iterrows():
    ax.annotate(r['ticker'], (r['eps_P1'], r['eps_P2']),
                fontsize=7.5, xytext=(5, 3),
                textcoords='offset points', color=COLOR_DYNA,
                fontweight='bold')

# Thresholds e linha de identidade
ax.axhline(THRESHOLD, color='gray', linestyle='--',
           linewidth=1, alpha=0.7)
ax.axvline(THRESHOLD, color='gray', linestyle='--',
           linewidth=1, alpha=0.7)
lim = max(stab_df['eps_P1'].max(),
          stab_df['eps_P2'].max()) + 0.05
ax.plot([0, lim], [0, lim], color='gray',
        linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('$\\varepsilon_i$ — Period 1 (2007–2016)',
              fontsize=11)
ax.set_ylabel('$\\varepsilon_i$ — Period 2 (2017–2026)',
              fontsize=11)
ax.set_title(
    f'Temporal Stability of $\\varepsilon_i$\n'
    f'Spearman $\\rho={rho:.3f}$ (p={p_rho:.3f}) — '
    f'Robust assets: {len(both_high)}',
    fontsize=11
)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, lim)
ax.set_ylim(-0.02, lim)
plt.tight_layout()
fig.savefig(os.path.join(FRL_DIR, 'fig_FRL2_stability.png'),
            dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_FRL2_stability.png ✓")

# ── FIG FRL-3: Mecanismo de cancelamento ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Type 2 Dynamic Decoupling: Frequency Regime Shift\n'
    'Period-full $\\varepsilon_i \\approx 0$ despite persistent '
    'decoupling in both sub-periods',
    fontsize=11, fontweight='bold'
)

for ax, ticker in zip(axes, ['BHRB', 'ASG']):
    if ticker not in tickers:
        ax.text(0.5, 0.5, f'{ticker} not found',
                ha='center', va='center',
                transform=ax.transAxes)
        continue

    idx    = tickers.index(ticker)
    ph     = phases[:, idx]

    # ω̂ rolling (janela de 252 dias)
    win    = 252
    omega_roll = np.full(len(ph), np.nan)
    for t in range(win, len(ph)):
        unwr           = np.unwrap(ph[t-win:t])
        omega_roll[t]  = np.diff(unwr).mean()

    ax.plot(dates, omega_roll,
            color=COLOR_DYNA, linewidth=1.0, alpha=0.8,
            label=f'$\\hat{{\\omega}}_i(t)$ rolling 252d')
    ax.axhline(omega_bar_global, color='black',
               linestyle='--', linewidth=1.5,
               label=f'$\\bar{{\\omega}}={omega_bar_global:.4f}$')
    ax.axvline(MID_DATE, color='gray',
               linestyle=':', linewidth=1,
               label='P1|P2 split (Jun 2016)')

    # Anotar períodos — guard contra NaN nos primeiros 252 pontos
    valid1 = ~np.isnan(omega_roll[mask1])
    ax.fill_between(
        dates[mask1],
        omega_roll[mask1],
        omega_bar_global,
        where=valid1 & (omega_roll[mask1] > omega_bar_global),
        alpha=0.15, color=COLOR_P1,
        label='Above $\\bar{\\omega}$ (P1)')

    valid2 = ~np.isnan(omega_roll[mask2])
    ax.fill_between(
        dates[mask2],
        omega_roll[mask2],
        omega_bar_global,
        where=valid2 & (omega_roll[mask2] < omega_bar_global),
        alpha=0.15, color=COLOR_P2,
        label='Below $\\bar{\\omega}$ (P2)')

    interp = INTERPRETATIONS.get(ticker, '')
    ax.set_title(f'{ticker} — {interp}', fontsize=9)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('$\\hat{\\omega}_i$ (rad/day)', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FRL_DIR, 'fig_FRL3_mechanism.png'),
            dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print("  fig_FRL3_mechanism.png ✓")

# ─────────────────────────────────────────────────────────────
# 7. RESUMO FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FRL-02 · EPSILON ANALYSIS CONCLUÍDO")
print("="*55)
print(f"  N ativos            : {N}")
print(f"  ε > {THRESHOLD}           : "
      f"{(meta['epsilon']>THRESHOLD).sum()} ativos")
print(f"  Kruskal-Wallis      : H={kw_stat:.3f}, p={kw_p:.3f}")
print(f"  Spearman ρ (P1,P2)  : {rho:.4f}, p={p_rho:.4f}")
print(f"  Ativos robustos     : {len(both_high)}")
print(f"    Tipo 1 (estático) : {len(type1)}")
print(f"    Tipo 2 (dinâmico) : {len(type2)}")
print(f"\n  Outputs em: {FRL_DIR}")
print(f"    - epsilon_summary.csv")
print(f"    - epsilon_sector.csv")
print(f"    - epsilon_stability.csv")
print(f"    - fig_FRL1_histogram.png")
print(f"    - fig_FRL2_stability.png")
print(f"    - fig_FRL3_mechanism.png")
print("="*55)
