"""
FRL-14 · longin_solnik_robust.py
=====================================
Testes robustos para o resultado Longin-Solnik:

  1. Mann-Whitney (referência — já no paper)
  2. Teste de permutação (10.000 draws)
     H₀: selecionar 8 ativos aleatoriamente produz
         Δρ tão baixo quanto o observado
  3. Bootstrap IC 95% para a diferença de Δρ médio
     entre grupo alto SHI e típicos
  4. Kolmogorov-Smirnov (distribuições completas)

Inputs:
  - longin_solnik_test.csv (Drive/frl/)

Outputs (Drive/frl/):
  - longin_solnik_robust_results.csv
  - fig_longin_solnik_robust.png
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from google.colab import drive

drive.mount('/content/drive')

DRIVE_DIR  = '/content/drive/MyDrive/ste_matrices_corrected'
FRL_DIR    = os.path.join(DRIVE_DIR, 'frl')
FIG_DPI    = 300
N_PERM     = 10_000
N_BOOT     = 10_000
SEED       = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────
# 1. CARREGAR DADOS
# ─────────────────────────────────────────────────────────────
print("Carregando dados...")
ls = pd.read_csv(os.path.join(FRL_DIR,
                 'longin_solnik_test.csv'),
                 index_col=0)

delta   = ls['delta_bear'].values.astype(float)
high    = ls['high_eps'].astype(bool).values
typical = ~high

delta_hi  = delta[high]
delta_typ = delta[typical]

n_hi  = high.sum()
n_typ = typical.sum()
n_tot = len(delta)

print(f"  N total  : {n_tot}")
print(f"  Alto SHI : {n_hi}")
print(f"  Típicos  : {n_typ}")
print(f"\n  Δρ médio alto SHI : {delta_hi.mean():.4f}")
print(f"  Δρ médio típicos  : {delta_typ.mean():.4f}")
print(f"  Diferença observada: "
      f"{delta_hi.mean() - delta_typ.mean():.4f}")

# ─────────────────────────────────────────────────────────────
# 2. MANN-WHITNEY (referência)
# ─────────────────────────────────────────────────────────────
print("\n=== Teste 1: Mann-Whitney ===")
mw_stat, mw_p = stats.mannwhitneyu(
    delta_hi, delta_typ,
    alternative='less')
print(f"  U = {mw_stat:.0f}")
print(f"  p = {mw_p:.4f}")
print(f"  {'✓ significativo' if mw_p < 0.05 else '✗ não significativo'} (α=0.05)")

# ─────────────────────────────────────────────────────────────
# 3. TESTE DE PERMUTAÇÃO
# ─────────────────────────────────────────────────────────────
print(f"\n=== Teste 2: Permutação ({N_PERM:,} draws) ===")
print("  H₀: selecionar 8 ativos aleatoriamente")
print("      produz Δρ médio tão baixo quanto observado")

obs_mean_hi = delta_hi.mean()
obs_diff    = delta_hi.mean() - delta_typ.mean()

# Estatística: média de Δρ do grupo de 8 selecionado
perm_means = np.zeros(N_PERM)
perm_diffs = np.zeros(N_PERM)

for i in range(N_PERM):
    idx = np.random.choice(n_tot, n_hi,
                           replace=False)
    perm_means[i] = delta[idx].mean()
    perm_diffs[i] = (delta[idx].mean() -
                     delta[~np.isin(
                         np.arange(n_tot), idx
                     )].mean())

# p-valor: fração de permutações com média ≤ observada
perm_p_mean = (perm_means <= obs_mean_hi).mean()
perm_p_diff = (perm_diffs <= obs_diff).mean()

print(f"\n  Estatística: média de Δρ do grupo de {n_hi}")
print(f"  Observado  : {obs_mean_hi:.4f}")
print(f"  Média null : {perm_means.mean():.4f}")
print(f"  Std null   : {perm_means.std():.4f}")
print(f"  p-valor    : {perm_p_mean:.4f}")
print(f"  {'✓ significativo' if perm_p_mean < 0.05 else '✗ não significativo'} (α=0.05)")

print(f"\n  Estatística alternativa: diferença de médias")
print(f"  Observado  : {obs_diff:.4f}")
print(f"  p-valor    : {perm_p_diff:.4f}")
print(f"  {'✓ significativo' if perm_p_diff < 0.05 else '✗ não significativo'} (α=0.05)")

# ─────────────────────────────────────────────────────────────
# 4. BOOTSTRAP IC 95%
# ─────────────────────────────────────────────────────────────
print(f"\n=== Teste 3: Bootstrap IC 95% ({N_BOOT:,} draws) ===")

boot_diff = np.zeros(N_BOOT)
boot_mean_hi = np.zeros(N_BOOT)

for i in range(N_BOOT):
    # Reamostrar com reposição dentro de cada grupo
    b_hi  = np.random.choice(delta_hi,
                              n_hi, replace=True)
    b_typ = np.random.choice(delta_typ,
                              n_typ, replace=True)
    boot_diff[i]    = b_hi.mean() - b_typ.mean()
    boot_mean_hi[i] = b_hi.mean()

# IC percentil
ci_diff = np.percentile(boot_diff, [2.5, 97.5])
ci_hi   = np.percentile(boot_mean_hi, [2.5, 97.5])

print(f"\n  Diferença de médias (alto SHI - típicos):")
print(f"  Observada : {obs_diff:.4f}")
print(f"  IC 95%    : [{ci_diff[0]:.4f}, "
      f"{ci_diff[1]:.4f}]")
print(f"  {'✓ IC exclui zero' if ci_diff[1] < 0 else '⚠ IC inclui zero'}")

print(f"\n  Média Δρ alto SHI:")
print(f"  Observada : {obs_mean_hi:.4f}")
print(f"  IC 95%    : [{ci_hi[0]:.4f}, "
      f"{ci_hi[1]:.4f}]")

# Bootstrap p-valor (shift para H₀)
boot_diff_shifted = boot_diff - boot_diff.mean()
boot_p = (boot_diff_shifted <= obs_diff).mean()
print(f"  p-valor bootstrap: {boot_p:.4f}")
print(f"  {'✓ significativo' if boot_p < 0.05 else '✗ não significativo'} (α=0.05)")

# ─────────────────────────────────────────────────────────────
# 5. KOLMOGOROV-SMIRNOV
# ─────────────────────────────────────────────────────────────
print(f"\n=== Teste 4: Kolmogorov-Smirnov ===")
print("  H₀: distribuições de Δρ são idênticas")

ks_stat, ks_p = stats.ks_2samp(
    delta_hi, delta_typ,
    alternative='less')
print(f"  KS stat = {ks_stat:.4f}")
print(f"  p       = {ks_p:.4f}")
print(f"  {'✓ significativo' if ks_p < 0.05 else '✗ não significativo'} (α=0.05)")

# ─────────────────────────────────────────────────────────────
# 6. TABELA RESUMO
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TABELA RESUMO — TESTES DE ROBUSTEZ")
print("="*60)
print(f"{'Teste':35s} {'Estatística':>12} {'p-valor':>10}")
print("-"*60)
print(f"{'Mann-Whitney (U)':35s} "
      f"{mw_stat:>12.0f} {mw_p:>10.4f}")
print(f"{'Permutação (média Δρ)':35s} "
      f"{obs_mean_hi:>12.4f} {perm_p_mean:>10.4f}")
print(f"{'Permutação (diferença)':35s} "
      f"{obs_diff:>12.4f} {perm_p_diff:>10.4f}")
print(f"{'Bootstrap (p-valor)':35s} "
      f"{'IC=['+f'{ci_diff[0]:.3f},{ci_diff[1]:.3f}'+']':>12} "
      f"{boot_p:>10.4f}")
print(f"{'Kolmogorov-Smirnov':35s} "
      f"{ks_stat:>12.4f} {ks_p:>10.4f}")
print("="*60)
print(f"\nDiferença observada de Δρ médio:")
print(f"  Alto SHI : {delta_hi.mean():.4f} "
      f"(n={n_hi})")
print(f"  Típicos  : {delta_typ.mean():.4f} "
      f"(n={n_typ})")
print(f"  Diferença: {obs_diff:.4f} "
      f"(alto SHI tem Δρ menor)")
print(f"  IC 95% bootstrap: "
      f"[{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")

# ─────────────────────────────────────────────────────────────
# 7. FIGURAS
# ─────────────────────────────────────────────────────────────
print("\nGerando figuras...")

c_hi  = '#C44E52'
c_typ = '#4C72B0'
c_nul = '#888888'

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'Robustness of the Longin-Solnik Test\n'
    'High-SHI assets vs. typical assets: '
    'crisis correlation increase (Δρ)',
    fontsize=11, fontweight='bold'
)

# ── Painel A: distribuição nula permutação ────────────────────
ax = axes[0]
ax.hist(perm_means, bins=60,
        color=c_nul, alpha=0.6,
        density=True, edgecolor='white',
        linewidth=0.3,
        label=f'Null distribution\n'
              f'({N_PERM:,} permutations)')
ax.axvline(obs_mean_hi, color=c_hi,
           linewidth=2.2, linestyle='--',
           label=f'Observed mean\n'
                 f'Δρ = {obs_mean_hi:.3f}')
ax.axvline(perm_means.mean(),
           color=c_nul, linewidth=1.2,
           linestyle=':',
           label=f'Null mean = '
                 f'{perm_means.mean():.3f}')

# Área de rejeição
x_crit = np.percentile(perm_means, 5)
x_range = np.linspace(perm_means.min(),
                       x_crit, 100)
from scipy.stats import gaussian_kde
kde_perm = gaussian_kde(perm_means,
                         bw_method=0.15)
ax.fill_between(x_range,
                kde_perm(x_range),
                alpha=0.3, color=c_hi,
                label=f'Rejection region\n'
                      f'p={perm_p_mean:.4f}')

ax.set_xlabel('Mean Δρ of random group of 8',
              fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title('(A) Permutation test\n'
             'Null distribution of mean Δρ',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right')
ax.grid(alpha=0.25)

# ── Painel B: bootstrap da diferença ─────────────────────────
ax = axes[1]
ax.hist(boot_diff, bins=60,
        color=c_typ, alpha=0.6,
        density=True, edgecolor='white',
        linewidth=0.3,
        label=f'Bootstrap distribution\n'
              f'({N_BOOT:,} draws)')
ax.axvline(obs_diff, color=c_hi,
           linewidth=2.2, linestyle='--',
           label=f'Observed diff\n'
                 f'{obs_diff:.3f}')
ax.axvline(ci_diff[0], color=c_hi,
           linewidth=1.2, linestyle=':',
           alpha=0.8)
ax.axvline(ci_diff[1], color=c_hi,
           linewidth=1.2, linestyle=':',
           alpha=0.8,
           label=f'95% CI\n'
                 f'[{ci_diff[0]:.3f}, '
                 f'{ci_diff[1]:.3f}]')
ax.axvline(0, color='black',
           linewidth=1.0, alpha=0.5,
           linestyle='-')

ax.set_xlabel('Difference in mean Δρ\n'
              '(high-SHI minus typical)',
              fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title('(B) Bootstrap CI\n'
             'Difference in mean Δρ',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right')
ax.grid(alpha=0.25)

# ── Painel C: distribuições empíricas Δρ ─────────────────────
ax = axes[2]

# KDE das distribuições reais
x_d = np.linspace(-0.3, 0.55, 300)
kde_hi  = gaussian_kde(delta_hi,
                        bw_method=0.35)
kde_typ = gaussian_kde(delta_typ,
                        bw_method=0.20)

ax.fill_between(x_d, kde_typ(x_d),
                alpha=0.30, color=c_typ,
                label=f'Typical (n={n_typ})\n'
                      f'mean={delta_typ.mean():.3f}')
ax.plot(x_d, kde_typ(x_d),
        color=c_typ, linewidth=1.8)

ax.fill_between(x_d, kde_hi(x_d),
                alpha=0.45, color=c_hi,
                label=f'High-SHI (n={n_hi})\n'
                      f'mean={delta_hi.mean():.3f}')
ax.plot(x_d, kde_hi(x_d),
        color=c_hi, linewidth=2.0)

# Rug plots
ax.plot(delta_typ,
        np.full_like(delta_typ, -0.15),
        '|', color=c_typ, alpha=0.20,
        markersize=5, markeredgewidth=0.8)
ax.plot(delta_hi,
        np.full_like(delta_hi, -0.35),
        '|', color=c_hi, alpha=1.0,
        markersize=10, markeredgewidth=2.0)

# Médias verticais
ax.axvline(delta_typ.mean(),
           color=c_typ, linewidth=1.2,
           linestyle=':', alpha=0.8)
ax.axvline(delta_hi.mean(),
           color=c_hi, linewidth=1.5,
           linestyle='--', alpha=0.9)

# Anotações dos 8 ativos
labels_hi = ['BANFP','BCAL','ARTNA',
             'AIFF','BLFS','AXON',
             'AN','BCDA']
for j, (tk, dv) in enumerate(
        zip(labels_hi, delta_hi)):
    ax.annotate(tk, (dv, -0.35),
                xytext=(dv, 0.3 + j*0.25),
                fontsize=6.5, color='darkred',
                ha='center',
                arrowprops=dict(
                    arrowstyle='-',
                    color='darkred',
                    alpha=0.3, lw=0.6))

# Estatísticas no painel
textstr = (f'Mann-Whitney p={mw_p:.3f}\n'
           f'Permutation p={perm_p_mean:.3f}\n'
           f'KS p={ks_p:.3f}')
ax.text(0.97, 0.97, textstr,
        transform=ax.transAxes,
        ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='lightyellow',
                  edgecolor='gray', alpha=0.8))

ax.set_xlabel('Δρ (crisis correlation increase)',
              fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title('(C) Empirical distributions of Δρ\n'
             'High-SHI vs. typical assets',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper left')
ax.set_ylim(-0.6)
ax.grid(alpha=0.25)

plt.tight_layout()
fig_path = os.path.join(FRL_DIR,
           'fig_longin_solnik_robust.png')
fig.savefig(fig_path, dpi=FIG_DPI,
            bbox_inches='tight')
print("  fig_longin_solnik_robust.png ✓")

# ─────────────────────────────────────────────────────────────
# 8. SALVAR RESULTADOS
# ─────────────────────────────────────────────────────────────
results = pd.DataFrame([
    {'test': 'Mann-Whitney',
     'statistic': mw_stat,
     'p_value': mw_p,
     'note': 'U statistic, one-sided (less)'},
    {'test': 'Permutation (mean Δρ)',
     'statistic': obs_mean_hi,
     'p_value': perm_p_mean,
     'note': f'{N_PERM} permutations, '
             f'null mean={perm_means.mean():.4f}'},
    {'test': 'Permutation (difference)',
     'statistic': obs_diff,
     'p_value': perm_p_diff,
     'note': f'observed diff={obs_diff:.4f}'},
    {'test': 'Bootstrap CI 95%',
     'statistic': obs_diff,
     'p_value': boot_p,
     'note': f'IC=[{ci_diff[0]:.4f},'
             f'{ci_diff[1]:.4f}]'},
    {'test': 'Kolmogorov-Smirnov',
     'statistic': ks_stat,
     'p_value': ks_p,
     'note': 'one-sided (less)'},
])
results.to_csv(
    os.path.join(FRL_DIR,
    'longin_solnik_robust_results.csv'),
    index=False)
print("  longin_solnik_robust_results.csv ✓")

print("\n" + "="*60)
print("FRL-14 · TESTES ROBUSTOS CONCLUÍDOS")
print("="*60)
