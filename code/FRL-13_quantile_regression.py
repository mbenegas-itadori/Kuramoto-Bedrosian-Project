"""
FRL-13 · quantile_regression.py
=====================================
Regressão incremental padronizada:
  Spec 1: OLS com variáveis padronizadas (z-scores)
  Spec 2: Regressão quantílica padronizada
          τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}

Modelo:
  Δρᵢ* = α + β₁·SHIᵢ* + β₂·σᵢ* + β₃·βᵢ* + εᵢ

Onde * denota z-score: (x - μ) / σ

Inputs:
  - log_returns_342_final.csv (local)
  - longin_solnik_test.csv (Drive/frl/)

Outputs (Drive/frl/):
  - quantile_regression_results.csv
  - fig_quantile_regression.png
  - fig_quantile_coefficients.png
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')
from google.colab import drive

drive.mount('/content/drive')

DRIVE_DIR = '/content/drive/MyDrive/ste_matrices_corrected'
FRL_DIR   = os.path.join(DRIVE_DIR, 'frl')
FIG_DPI   = 300
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# ─────────────────────────────────────────────────────────────
# 1. CARREGAR DADOS
# ─────────────────────────────────────────────────────────────
print("Carregando dados...")

lr = pd.read_csv('log_returns_342_final.csv',
                 index_col=0, parse_dates=True)
ls = pd.read_csv(os.path.join(FRL_DIR,
                 'longin_solnik_test.csv'),
                 index_col=0)

lr = lr.apply(pd.to_numeric, errors='coerce').fillna(0)
lr = lr.clip(-0.5, 0.5)

print(f"  Log-retornos: {lr.shape}")
print(f"  Longin-Solnik: {ls.shape}")

# ─────────────────────────────────────────────────────────────
# 2. CALCULAR CONTROLES WIDEBAND
# ─────────────────────────────────────────────────────────────
print("\nCalculando controles...")

mkt_ret = lr.mean(axis=1)
controls = []

for ticker in ls.index:
    if ticker not in lr.columns:
        continue
    r = lr[ticker]
    m = mkt_ret.reindex(r.index)
    mask = (r != 0) & r.notna() & m.notna()
    r = r[mask].values.astype(float)
    m = m[mask].values.astype(float)
    if len(r) < 252:
        continue
    if np.std(r) < 1e-8 or np.std(m) < 1e-8:
        continue
    try:
        X = np.column_stack([np.ones(len(m)), m])
        res = sm.OLS(r, X).fit()
        beta = float(res.params[1])
    except:
        beta = float(np.corrcoef(r, m)[0,1] *
                     np.std(r) / np.std(m))
    vol = float(np.std(r) * np.sqrt(252))
    controls.append({
        'ticker':   ticker,
        'beta_mkt': beta,
        'vol':      vol,
    })

ctrl_df = pd.DataFrame(controls).set_index('ticker')
print(f"  Controles: {len(ctrl_df)} ativos")

# ─────────────────────────────────────────────────────────────
# 3. MONTAR DATASET
# ─────────────────────────────────────────────────────────────
print("\nMontando dataset...")

reg_df = ls[['epsilon','delta_bear',
             'high_eps']].join(
    ctrl_df[['beta_mkt','vol']], how='inner')
reg_df = reg_df.dropna()

print(f"  N final: {len(reg_df)} ativos")
print(f"  Alto SHI: {reg_df['high_eps'].sum()}")

# ─────────────────────────────────────────────────────────────
# 4. PADRONIZAR (Z-SCORES)
# ─────────────────────────────────────────────────────────────
def standardize(s):
    return (s - s.mean()) / s.std()

reg_df['shi_z']      = standardize(reg_df['epsilon'])
reg_df['vol_z']      = standardize(reg_df['vol'])
reg_df['beta_z']     = standardize(reg_df['beta_mkt'])
reg_df['delta_z']    = standardize(reg_df['delta_bear'])

# Estatísticas descritivas
print("\nEstatísticas das variáveis originais:")
for col, label in [('epsilon','SHI'),
                   ('vol','Volatility'),
                   ('beta_mkt','Beta'),
                   ('delta_bear','Δρ')]:
    print(f"  {label:15s}: "
          f"μ={reg_df[col].mean():.4f}  "
          f"σ={reg_df[col].std():.4f}  "
          f"min={reg_df[col].min():.4f}  "
          f"max={reg_df[col].max():.4f}")

print("\nCorrelações entre variáveis padronizadas:")
corr = reg_df[['shi_z','vol_z',
               'beta_z','delta_z']].corr()
print(corr.round(3))

# ─────────────────────────────────────────────────────────────
# 5. OLS PADRONIZADO — 5 ESPECIFICAÇÕES
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("OLS PADRONIZADO")
print("Variável dependente: Δρ* (z-score)")
print("="*65)

y_z = reg_df['delta_z'].values.astype(float)

specs_ols = {
    'M1': ['shi_z'],
    'M2': ['shi_z', 'beta_z'],
    'M3': ['shi_z', 'vol_z'],
    'M4': ['shi_z', 'beta_z', 'vol_z'],
}

ols_models = {}
for name, cols in specs_ols.items():
    X = sm.add_constant(
        reg_df[cols].values.astype(float))
    mod = sm.OLS(y_z, X).fit(cov_type='HC3')
    ols_models[name] = (mod, cols)

# Tabela OLS
print(f"\n{'':22s} {'M1':>10} {'M2':>10} "
      f"{'M3':>10} {'M4':>10}")
print("-"*62)

for var, label in [
    ('shi_z',  'SHI* (β₁)'),
    ('beta_z', 'Beta* (β₂)'),
    ('vol_z',  'Vol* (β₃)'),
    ('const',  'Constant'),
]:
    row_c = f"  {label:20s}"
    row_t = f"  {'':20s}"
    for name, (mod, cols) in ols_models.items():
        all_v = ['const'] + cols
        if var in all_v:
            idx = all_v.index(var)
            c = mod.params[idx]
            t = mod.tvalues[idx]
            p = mod.pvalues[idx]
            sig = ('***' if p<0.01 else
                   '**'  if p<0.05 else
                   '*'   if p<0.10 else '')
            row_c += f"  {c:>7.4f}{sig:<3}"
            row_t += f"  ({t:>5.2f})  "
        else:
            row_c += f"  {'---':>10}"
            row_t += f"  {'':>10}"
    print(row_c)
    print(row_t)

print("-"*62)
row_r2 = f"  {'Adj. R²':20s}"
row_n  = f"  {'N':20s}"
for name, (mod, cols) in ols_models.items():
    row_r2 += f"  {mod.rsquared_adj:>10.4f}"
    row_n  += f"  {int(mod.nobs):>10}"
print(row_r2)
print(row_n)
print("="*65)
print("*** p<0.01, ** p<0.05, * p<0.10 | HC3 robust SE")
print("Coeficientes interpretados como: 1 SD de X → β SD de Δρ")

# ─────────────────────────────────────────────────────────────
# 6. REGRESSÃO QUANTÍLICA PADRONIZADA
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("REGRESSÃO QUANTÍLICA PADRONIZADA")
print("Variável dependente: Δρ* (z-score)")
print(f"Quantis: {QUANTILES}")
print("="*65)

# Usar statsmodels QuantReg
from statsmodels.regression.quantile_regression \
    import QuantReg

y_q = reg_df['delta_z'].values.astype(float)
X_q = sm.add_constant(
    reg_df[['shi_z','vol_z',
            'beta_z']].values.astype(float))

qr_models = {}
for tau in QUANTILES:
    mod = QuantReg(y_q, X_q).fit(q=tau,
                                  max_iter=2000)
    qr_models[tau] = mod

# Tabela quantílica
vars_q = ['const','SHI*','Vol*','Beta*']
print(f"\n{'':18s}", end='')
for tau in QUANTILES:
    print(f"  τ={tau:.2f}  ", end='')
print()
print("-"*75)

for i, var in enumerate(vars_q):
    row_c = f"  {var:16s}"
    row_t = f"  {'':16s}"
    for tau in QUANTILES:
        mod = qr_models[tau]
        c = mod.params[i]
        t = mod.tvalues[i]
        p = mod.pvalues[i]
        sig = ('***' if p<0.01 else
               '**'  if p<0.05 else
               '*'   if p<0.10 else '')
        row_c += f"  {c:>6.4f}{sig:<3}"
        row_t += f"  ({t:>5.2f})  "
    print(row_c)
    print(row_t)

print("-"*75)
print("*** p<0.01, ** p<0.05, * p<0.10")

# Pseudo R² de Koenker-Machado
print("\nPseudo R² (Koenker-Machado):")
for tau in QUANTILES:
    mod = qr_models[tau]
    print(f"  τ={tau:.2f}: {mod.prsquared:.4f}")

# ─────────────────────────────────────────────────────────────
# 7. SALVAR RESULTADOS
# ─────────────────────────────────────────────────────────────
rows = []

# OLS
for name, (mod, cols) in ols_models.items():
    all_v = ['const'] + cols
    for i, var in enumerate(all_v):
        rows.append({
            'type': 'OLS', 'model': name,
            'quantile': np.nan,
            'variable': var,
            'coef': mod.params[i],
            'tstat': mod.tvalues[i],
            'pvalue': mod.pvalues[i],
            'r2': mod.rsquared_adj,
            'n': mod.nobs
        })

# QR
for tau, mod in qr_models.items():
    for i, var in enumerate(vars_q):
        rows.append({
            'type': 'QR', 'model': f'tau={tau}',
            'quantile': tau,
            'variable': var,
            'coef': mod.params[i],
            'tstat': mod.tvalues[i],
            'pvalue': mod.pvalues[i],
            'r2': mod.prsquared,
            'n': mod.nobs
        })

res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(FRL_DIR,
              'quantile_regression_results.csv'),
              index=False)
print("\n✓ quantile_regression_results.csv salvo")

# ─────────────────────────────────────────────────────────────
# 8. FIGURAS
# ─────────────────────────────────────────────────────────────
print("\nGerando figuras...")

c_shi = '#C44E52'
c_vol = '#4C72B0'
c_bet = '#2CA02C'
c_ols = '#888888'

# ── Fig 1: Coeficientes por quantil ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    'Quantile Regression: SHI Incremental Power\n'
    'Standardized coefficients by quantile of Δρ*',
    fontsize=11, fontweight='bold'
)

var_info = [
    (1, 'SHI* (β₁)', c_shi),
    (2, 'Volatility* (β₂)', c_vol),
    (3, 'Market beta* (β₃)', c_bet),
]

# OLS M4 coeficientes para referência
ols_m4 = ols_models['M4'][0]
ols_coefs = {
    1: ols_m4.params[1],
    2: ols_m4.params[2],
    3: ols_m4.params[3],
}
ols_ci = {
    1: 1.96 * ols_m4.bse[1],
    2: 1.96 * ols_m4.bse[2],
    3: 1.96 * ols_m4.bse[3],
}

for ax, (idx, label, color) in zip(axes, var_info):
    coefs = [qr_models[tau].params[idx]
             for tau in QUANTILES]
    cis   = [1.96 * qr_models[tau].bse[idx]
             for tau in QUANTILES]
    pvals = [qr_models[tau].pvalues[idx]
             for tau in QUANTILES]

    # Linha dos coeficientes QR
    ax.plot(QUANTILES, coefs,
            color=color, linewidth=2.2,
            marker='o', markersize=7,
            zorder=5, label='Quantile reg.')

    # IC 95%
    ax.fill_between(
        QUANTILES,
        [c-e for c,e in zip(coefs,cis)],
        [c+e for c,e in zip(coefs,cis)],
        alpha=0.20, color=color)

    # Marcar significância
    for tau, c, p in zip(QUANTILES, coefs, pvals):
        if p < 0.01:
            ax.plot(tau, c, '*', color='black',
                    markersize=10, zorder=6)
        elif p < 0.05:
            ax.plot(tau, c, 'o', color='black',
                    markersize=6, zorder=6,
                    markerfacecolor='none',
                    markeredgewidth=1.5)

    # Linha OLS como referência
    ax.axhline(ols_coefs[idx],
               color=c_ols, linewidth=1.5,
               linestyle='--', alpha=0.7,
               label=f'OLS M4')
    ax.fill_between(
        [0.05, 0.95],
        [ols_coefs[idx]-ols_ci[idx]]*2,
        [ols_coefs[idx]+ols_ci[idx]]*2,
        alpha=0.10, color=c_ols)

    ax.axhline(0, color='black',
               linewidth=0.8, alpha=0.5)
    ax.set_xticks(QUANTILES)
    ax.set_xticklabels(
        [f'τ={t}' for t in QUANTILES],
        fontsize=8, rotation=30)
    ax.set_xlabel('Quantile of Δρ*', fontsize=9)
    ax.set_ylabel('Standardized coefficient',
                  fontsize=9)
    ax.set_title(label, fontsize=10,
                 fontweight='bold', color=color)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Anotação: * = p<0.01, circle = p<0.05
    ax.text(0.02, 0.02,
            '★ p<0.01  ○ p<0.05',
            transform=ax.transAxes,
            fontsize=7, color='black',
            alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(FRL_DIR,
            'fig_quantile_coefficients.png'),
            dpi=FIG_DPI, bbox_inches='tight')
plt.savefig('/tmp/fig_qr_coefs.png',
            dpi=150, bbox_inches='tight')
print("  fig_quantile_coefficients.png ✓")

# ── Fig 2: Scatter Δρ × SHI por quantil ──────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(
    'SHI and Crisis Correlation Increase\n'
    'OLS vs. Quantile Regression fits',
    fontsize=11, fontweight='bold'
)

# Painel A: scatter com fits quantílicos
ax = axes2[0]
typical = reg_df[~reg_df['high_eps']]
high    = reg_df[reg_df['high_eps']]

ax.scatter(typical['shi_z'],
           typical['delta_z'],
           color=c_vol, alpha=0.20, s=15,
           label='Typical (n=308)', zorder=2)
ax.scatter(high['shi_z'],
           high['delta_z'],
           color=c_shi, s=80, zorder=5,
           edgecolors='darkred', linewidth=0.8,
           label='High-SHI (n=8)')

x_fit = np.linspace(reg_df['shi_z'].min(),
                    reg_df['shi_z'].max(), 100)

# OLS fit
ols_m1 = ols_models['M1'][0]
y_ols = (ols_m1.params[0]
         + ols_m1.params[1] * x_fit)
ax.plot(x_fit, y_ols, '--',
        color=c_ols, linewidth=1.8,
        label='OLS fit', zorder=4)

# QR fits para τ selecionados
qr_colors = {0.10: '#d62728',
             0.25: '#ff7f0e',
             0.50: '#2ca02c',
             0.75: '#1f77b4',
             0.90: '#9467bd'}

for tau in [0.10, 0.25, 0.50, 0.75, 0.90]:
    mod = qr_models[tau]
    # Apenas intercepto + coef SHI (idx 1)
    y_qr = mod.params[0] + mod.params[1] * x_fit
    ax.plot(x_fit, y_qr,
            color=qr_colors[tau],
            linewidth=1.4, alpha=0.85,
            label=f'QR τ={tau}')

ax.axhline(0, color='black',
           linewidth=0.6, alpha=0.4)
ax.axvline(0, color='black',
           linewidth=0.6, alpha=0.4)
ax.set_xlabel('SHI* (z-score)', fontsize=9)
ax.set_ylabel('Δρ* (z-score)', fontsize=9)
ax.set_title('(A) SHI* vs. Δρ*\nwith QR fits by quantile',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7, loc='upper left',
          ncol=2)
ax.grid(alpha=0.25)

# Painel B: coeficiente SHI por quantil
# (versão limpa para o paper)
ax = axes2[1]
coefs_shi = [qr_models[tau].params[1]
             for tau in QUANTILES]
cis_shi   = [1.96 * qr_models[tau].bse[1]
             for tau in QUANTILES]
pvals_shi = [qr_models[tau].pvalues[1]
             for tau in QUANTILES]

bars = ax.bar(
    [f'τ={t}' for t in QUANTILES],
    coefs_shi,
    color=[c_shi if p < 0.05 else '#DDAAAA'
           for p in pvals_shi],
    alpha=0.85, edgecolor='darkred',
    linewidth=0.8)

ax.errorbar(
    range(len(QUANTILES)),
    coefs_shi, yerr=cis_shi,
    fmt='none', color='black',
    capsize=4, linewidth=1.2)

# OLS M1 como referência
ax.axhline(ols_m1.params[1],
           color=c_ols, linewidth=1.8,
           linestyle='--', alpha=0.8,
           label=f'OLS = {ols_m1.params[1]:.4f}')

ax.axhline(0, color='black',
           linewidth=0.8, alpha=0.5)

# Anotar p-valores
for i, (c, p) in enumerate(
        zip(coefs_shi, pvals_shi)):
    sig = ('***' if p<0.01 else
           '**'  if p<0.05 else
           '*'   if p<0.10 else 'n.s.')
    ax.text(i, c + (0.02 if c >= 0 else -0.05),
            sig, ha='center', va='bottom',
            fontsize=8.5, fontweight='bold',
            color='darkred')

ax.set_ylabel('Standardized coefficient on SHI*',
              fontsize=9)
ax.set_title('(B) SHI* coefficient across quantiles\n'
             'Dark red = significant at 5%',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis='y')

plt.tight_layout()
fig2.savefig(os.path.join(FRL_DIR,
             'fig_quantile_regression.png'),
             dpi=FIG_DPI, bbox_inches='tight')
fig2.savefig('/tmp/fig_qr_main.png',
             dpi=150, bbox_inches='tight')
print("  fig_quantile_regression.png ✓")

# ─────────────────────────────────────────────────────────────
# 9. RESUMO FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("RESUMO — REGRESSÃO QUANTÍLICA")
print("="*65)
print(f"\n  Coeficiente SHI* por quantil:")
for tau, c, p in zip(QUANTILES,
                      coefs_shi, pvals_shi):
    sig = ('***' if p<0.01 else
           '**'  if p<0.05 else
           '*'   if p<0.10 else 'n.s.')
    print(f"    τ={tau:.2f}: β={c:>8.4f}  "
          f"p={p:.4f}  {sig}")
print(f"\n  OLS M1 (referência): "
      f"β={ols_m1.params[1]:.4f}  "
      f"p={ols_m1.pvalues[1]:.4f}")
print(f"\n  Padrão esperado se SHI é estrutural:")
print(f"    β cresce em magnitude nos quantis baixos")
print(f"    → SHI mais relevante onde Δρ é menor")
print("="*65)
