"""
setup_quantile_regression.py
Setup completo para rodar quantile_regression.py no Colab.
"""

import os
import shutil
import pandas as pd
import psutil
from google.colab import drive

drive.mount('/content/drive')

DRIVE_DIR = '/content/drive/MyDrive/ste_matrices_corrected'
FRL_DIR   = os.path.join(DRIVE_DIR, 'frl')

# ─────────────────────────────────────────────────────────────
# 1. Arquivo local necessário
# ─────────────────────────────────────────────────────────────
print("=== Arquivos locais ===")
f = 'log_returns_342_final.csv'
if os.path.exists(f):
    size = os.path.getsize(f) / 1e6
    print(f"  ✓ {f} ({size:.1f} MB)")
else:
    src = os.path.join(DRIVE_DIR, f)
    if os.path.exists(src):
        shutil.copy(src, f)
        size = os.path.getsize(f) / 1e6
        print(f"  ✓ {f} copiado ({size:.1f} MB)")
    else:
        print(f"  ✗ {f} — não encontrado")

# ─────────────────────────────────────────────────────────────
# 2. Arquivo no Drive
# ─────────────────────────────────────────────────────────────
print("\n=== Arquivos no Drive ===")
f_drive = os.path.join(FRL_DIR,
                        'longin_solnik_test.csv')
if os.path.exists(f_drive):
    df = pd.read_csv(f_drive, index_col=0)
    print(f"  ✓ longin_solnik_test.csv: "
          f"{len(df)} ativos, "
          f"colunas={df.columns.tolist()}")
    # Verificar colunas necessárias
    needed = ['epsilon','delta_bear','high_eps']
    missing = [c for c in needed
               if c not in df.columns]
    if missing:
        print(f"  ⚠ Colunas faltando: {missing}")
    else:
        print(f"  ✓ Todas as colunas necessárias presentes")
else:
    print(f"  ✗ longin_solnik_test.csv não encontrado")
    print(f"    Caminho: {f_drive}")

# ─────────────────────────────────────────────────────────────
# 3. Verificar statsmodels (necessário para QuantReg)
# ─────────────────────────────────────────────────────────────
print("\n=== Dependências ===")
try:
    from statsmodels.regression.quantile_regression \
        import QuantReg
    import statsmodels
    print(f"  ✓ statsmodels {statsmodels.__version__}"
          f" — QuantReg disponível")
except ImportError:
    print("  ✗ statsmodels não disponível")
    print("    Instalando...")
    os.system('pip install statsmodels '
              '--break-system-packages -q')
    print("  ✓ instalado")

# ─────────────────────────────────────────────────────────────
# 4. Verificar pasta de output
# ─────────────────────────────────────────────────────────────
print("\n=== Pasta de output ===")
print(f"  ✓ Drive/frl/: {FRL_DIR}")
os.makedirs(FRL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 5. RAM
# ─────────────────────────────────────────────────────────────
print("\n=== RAM ===")
ram = psutil.virtual_memory()
print(f"  Disponível: {ram.available/1e9:.1f} GB "
      f"de {ram.total/1e9:.1f} GB")
status = ('✓ suficiente'
          if ram.available > 2e9
          else '⚠ considere reiniciar o runtime')
print(f"  {status}")

# ─────────────────────────────────────────────────────────────
# 6. Pronto
# ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("TUDO PRONTO. Para rodar:")
print("="*50)
print("  exec(open('quantile_regression.py').read())")
print("="*50)
print("\nTempo estimado: 3-5 minutos")
print("Outputs em Drive/frl/:")
print("  - quantile_regression_results.csv")
print("  - fig_quantile_coefficients.png")
print("  - fig_quantile_regression.png")
