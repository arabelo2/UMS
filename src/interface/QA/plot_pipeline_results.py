#!/usr/bin/env python3
"""
Enhanced 2D Pipeline Results Plotter
Combines axial and lateral analysis for Digital Twin vs TFM comparison
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Para servidores headless
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import json
import csv
import warnings
from fwhm_methods_addon import estimate_all_fwhm_methods

# Suprimir avisos de divisões por zero
warnings.filterwarnings("ignore")

# Configurações globais de plotagem
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 18
})

def ensure_dir(path):
    """Garante que o diretório existe"""
    os.makedirs(path, exist_ok=True)

def classify_result(value, reference, is_axial=True):
    """
    Classifica resultado baseado em erro relativo
    Incorporated color scheme from plot_pipeline_results_FWHM.py
    
    Args:
        value: Valor medido
        reference: Valor de referência (teórico para axial, DT para lateral)
        is_axial: Se True, usa limites mais rígidos para axial
    """
    if reference is None or reference <= 0 or value is None or np.isnan(value):
        return '#7f8c8d', 'Unknown'  # Cinza
    
    error_pct = abs(value - reference) / reference * 100
    
    # Usando o esquema de cores do plot_pipeline_results_FWHM.py
    if is_axial:
        # Limites mais rígidos para axial (ajustados para 4 categorias)
        if error_pct < 10:
            return '#2ecc71', 'Good (<10% error)'       # Verde
        elif error_pct < 20:
            return '#f39c12', 'Moderate (10-20% error)' # Amarelo/Laranja
        elif error_pct < 40:
            return '#e67e22', 'Poor (20-40% error)'     # Laranja escuro
        else:
            return '#e74c3c', 'Very Poor (>40% error)'  # Vermelho
    else:
        # Limites para lateral usando o esquema FWHM
        if error_pct < 15:
            return '#2ecc71', 'Good (<15% error)'       # Verde
        elif error_pct < 50:
            return '#f39c12', 'Approximation (15-50% error)'  # Amarelo/Laranja
        else:
            return '#e74c3c', 'Poor (>50% error)'       # Vermelho

def create_legend_elements(include_theoretical=True, include_patterns=True):
    """
    Cria elementos de legenda consistentes com plot_pipeline_results_FWHM.py
    
    Args:
        include_theoretical: Se True, inclui linha teórica na legenda
        include_patterns: Se True, inclui padrões de hachura
    """
    legend_elements = []
    
    # Seção de Qualidade (baseado no classify_result)
    if include_theoretical:
        legend_elements.append(
            Line2D([0], [0], color='black', ls='--', lw=3, label='Theoretical Baseline')
        )
    
    # Seção de Metodologia (padrões)
    if include_patterns:
        legend_elements.extend([
            Patch(facecolor='gray', hatch='///', alpha=0.7, edgecolor='black', 
                  label='Digital Twin (Hatched)'),
            Patch(facecolor='gray', alpha=0.9, edgecolor='black', 
                  label='TFM (Solid)')
        ])
    
    # Seção de Categorias de Qualidade
    legend_elements.extend([
        Patch(facecolor='#2ecc71', label='Good (<15% error)'),
        Patch(facecolor='#f39c12', label='Approximation (15-50% error)'),
        Patch(facecolor='#e74c3c', label='Poor (>50% error)'),
        Patch(facecolor='#7f8c8d', label='Unknown/Invalid')
    ])
    
    return legend_elements

def load_data_2d(out_root):
    """
    Carrega todos os dados 2D do pipeline
    
    Returns:
        dict: Dados carregados ou None em caso de erro
    """
    try:
        # Digital Twin
        dt_dir = os.path.join(out_root, "digital_twin")
        z_dt = np.loadtxt(os.path.join(dt_dir, "field_z_vals.csv"), delimiter=',')
        x_dt = np.loadtxt(os.path.join(dt_dir, "field_x_vals.csv"), delimiter=',')
        p_field_raw = np.loadtxt(os.path.join(dt_dir, "field_p_field.csv"), delimiter=',')
        
        # Verificar dimensões e reorganizar se necessário
        if p_field_raw.shape[0] == len(z_dt) and p_field_raw.shape[1] == len(x_dt):
            p_field = p_field_raw  # Já está no formato correto (z, x)
        elif p_field_raw.shape[0] == len(x_dt) and p_field_raw.shape[1] == len(z_dt):
            p_field = p_field_raw.T  # Transpor para (z, x)
        else:
            print(f"[WARNING] Unexpected shape: p_field_raw {p_field_raw.shape}, x_dt({len(x_dt)}), z_dt({len(z_dt)})")
            p_field = p_field_raw
        
        # TFM
        tfm_dir = os.path.join(out_root, "fmc_tfm")
        z_tfm = np.loadtxt(os.path.join(tfm_dir, "results_z_vals.csv"), delimiter=',')
        x_tfm = np.loadtxt(os.path.join(tfm_dir, "results_x_vals.csv"), delimiter=',')
        envelope_2d = np.loadtxt(os.path.join(tfm_dir, "results_envelope_2d.csv"), delimiter=',')
        
        # Verificar dimensões - envelope_2d deve ser (x_tfm, z_tfm)
        if envelope_2d.shape != (len(x_tfm), len(z_tfm)):
            print(f"[WARNING] Shape mismatch: envelope_2d {envelope_2d.shape} vs x_tfm({len(x_tfm)}), z_tfm({len(z_tfm)})")
            # Tentar transpor se necessário
            if envelope_2d.shape == (len(z_tfm), len(x_tfm)):
                envelope_2d = envelope_2d.T  # Transpor para (x, z)
        
        # Extrair perfis centrais
        ix_center_dt = np.argmin(np.abs(x_dt - 0.0))
        iz_center_dt = np.argmin(np.abs(z_dt - 0.0))
        
        ix_center_tfm = np.argmin(np.abs(x_tfm - 0.0))
        
        return {
            # Digital Twin
            'z_dt': z_dt, 'x_dt': x_dt, 'p_field': p_field,
            'dt_axial': p_field[:, ix_center_dt],  # Perfil axial (z) em x=0
            'dt_lateral': p_field[iz_center_dt, :],  # Perfil lateral (x) em z=0
            
            # TFM
            'z_tfm': z_tfm, 'x_tfm': x_tfm, 'envelope_2d': envelope_2d,
            'tfm_axial': envelope_2d[ix_center_tfm, :],  # Perfil axial (z) em x=0
            # 'tfm_lateral' será calculado nas funções de plotagem
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_field_maps_2d(data, params, output_dir="plots"):
    """
    Plota mapas de campo 2D para Digital Twin e TFM
    """
    print(f"[PLOT] Generating 2D Field Maps...")
    
    Dt0 = params.get('Dt0', 59.2)
    target_z = Dt0 + params.get('DF', 23.6)
    
    # Criar figura com 4 subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Digital Twin (Full)
    ax1 = plt.subplot(2, 3, 1)
    p_db = 20 * np.log10(data['p_field'] / np.max(data['p_field']) + 1e-6)
    im1 = ax1.pcolormesh(data['x_dt'], data['z_dt'], p_db, 
                        cmap='jet', vmin=-40, vmax=0, shading='auto')
    ax1.set_title("Digital Twin Field Map (dB)")
    ax1.axhline(Dt0, color='white', ls='--', alpha=0.7, label='Interface')
    ax1.axhline(target_z, color='cyan', ls='--', alpha=0.7, label='Focus')
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("z (mm)")
    plt.colorbar(im1, ax=ax1, label='dB')
    ax1.legend(loc='upper right', fontsize=10)
    
    # 2. Digital Twin (Zoom)
    ax2 = plt.subplot(2, 3, 2)
    z_mask = (data['z_dt'] > target_z - 30) & (data['z_dt'] < target_z + 30)
    if np.any(z_mask):
        im2 = ax2.pcolormesh(data['x_dt'], data['z_dt'][z_mask], 
                            p_db[z_mask, :], cmap='jet', vmin=-20, vmax=0, shading='auto')
        ax2.set_title("Digital Twin - Focal Zone")
        ax2.axvline(0, color='white', ls='--', alpha=0.5)
        ax2.axhline(target_z, color='cyan', ls='--', alpha=0.7)
        ax2.set_xlabel("x (mm)")
        ax2.set_ylabel("z (mm)")
    
    # 3. TFM Reconstruction (Full)
    ax3 = plt.subplot(2, 3, 3)
    # Normalizar envelope para [0, 1]
    envelope_norm = data['envelope_2d'] / np.max(data['envelope_2d'])
    # Note: envelope_2d tem shape (31, 151) -> (x_tfm, z_tfm)
    im3 = ax3.pcolormesh(data['z_tfm'], data['x_tfm'], envelope_norm,
                        cmap='hot', vmin=0, vmax=1, shading='auto')
    ax3.set_title("TFM Reconstruction (Normalized)")
    ax3.axvline(Dt0, color='cyan', ls='--', alpha=0.7, label='Interface')
    ax3.axvline(target_z, color='white', ls='--', alpha=0.7, label='Focus')
    ax3.set_xlabel("z (mm)")
    ax3.set_ylabel("x (mm)")
    plt.colorbar(im3, ax=ax3, label='Normalized Amplitude')
    ax3.legend(loc='upper right', fontsize=10)
    
    # 4. TFM (Zoom)
    ax4 = plt.subplot(2, 3, 4)
    z_mask_tfm = (data['z_tfm'] > target_z - 30) & (data['z_tfm'] < target_z + 30)
    if np.any(z_mask_tfm):
        # envelope_norm tem shape (31, 151), precisamos cortar as colunas (z)
        zoom_envelope = envelope_norm[:, z_mask_tfm]
        im4 = ax4.pcolormesh(data['z_tfm'][z_mask_tfm], data['x_tfm'], 
                            zoom_envelope, cmap='hot', vmin=0, vmax=1, shading='auto')
        ax4.set_title("TFM - Focal Zone")
        ax4.axvline(target_z, color='white', ls='--', alpha=0.7)
        ax4.set_xlabel("z (mm)")
        ax4.set_ylabel("x (mm)")
    
    # 5. Diferença entre DT e TFM (na região focal)
    ax5 = plt.subplot(2, 3, 5)
    # Interpolar Digital Twin para mesma grade do TFM
    from scipy.interpolate import griddata
    
    # Digital Twin: p_db tem shape (151, 41) -> (z_dt, x_dt)
    # TFM: envelope_norm tem shape (31, 151) -> (x_tfm, z_tfm)
    
    # Criar grade do Digital Twin (usando indexing='xy' para compatibilidade)
    X_dt, Z_dt = np.meshgrid(data['x_dt'], data['z_dt'], indexing='xy')  # (151, 41)
    points_dt = np.column_stack((X_dt.ravel(), Z_dt.ravel()))  # (151*41, 2)
    values_dt = p_db.ravel()  # (151*41,)
    
    # Criar grade do TFM (usando indexing='ij' para compatibilidade)
    X_tfm, Z_tfm = np.meshgrid(data['x_tfm'], data['z_tfm'], indexing='ij')  # (31, 151)
    points_tfm = np.column_stack((X_tfm.ravel(), Z_tfm.ravel()))  # (31*151, 2)
    
    # Interpolar Digital Twin para a grade do TFM
    p_db_interp_flat = griddata(points_dt, values_dt, points_tfm, method='linear', fill_value=np.nan)
    p_db_interp = p_db_interp_flat.reshape(X_tfm.shape)  # (31, 151)
    
    # Calcular diferença (apenas na região onde temos dados)
    mask = ~np.isnan(p_db_interp)
    if np.any(mask):
        # Convert envelope_norm to dB for comparison
        envelope_db = 20 * np.log10(envelope_norm + 1e-6)
        diff = np.zeros_like(p_db_interp)
        diff[mask] = p_db_interp[mask] - envelope_db[mask]
        
        im5 = ax5.pcolormesh(data['z_tfm'], data['x_tfm'], diff, 
                            cmap='RdBu_r', vmin=-20, vmax=20, shading='auto')
        ax5.set_title("Difference: DT(dB) - TFM(dB)")
        ax5.set_xlabel("z (mm)")
        ax5.set_ylabel("x (mm)")
        plt.colorbar(im5, ax=ax5, label='dB Difference')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "field_maps_2d_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_profiles_comparison(data, params, output_dir="plots"):
    """
    Plota comparação de perfis axiais e laterais
    """
    print(f"[PLOT] Generating Profile Comparisons...")
    
    Dt0 = params.get('Dt0', 59.2)
    target_z = Dt0 + params.get('DF', 23.6)
    
    # Calcular índice focal para TFM
    iz_focus = np.argmin(np.abs(data['z_tfm'] - target_z))
    data['tfm_lateral'] = data['envelope_2d'][:, iz_focus]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Normalizar perfis
    dt_axial_norm = data['dt_axial'] / np.max(data['dt_axial'])
    dt_lateral_norm = data['dt_lateral'] / np.max(data['dt_lateral'])
    tfm_axial_norm = data['tfm_axial'] / np.max(data['tfm_axial'])
    tfm_lateral_norm = data['tfm_lateral'] / np.max(data['tfm_lateral'])
    
    # 1. Perfis Axiais (DT vs TFM)
    ax1 = axes[0, 0]
    ax1.plot(data['z_dt'], dt_axial_norm, 'b-', linewidth=2, label='Digital Twin')
    ax1.plot(data['z_tfm'], tfm_axial_norm, 'r--', linewidth=2, label='TFM Reconstruction')
    ax1.axvline(Dt0, color='gray', linestyle=':', alpha=0.7, label='Interface')
    ax1.axvline(target_z, color='green', linestyle='--', alpha=0.7, label='Target Focus')
    ax1.set_xlabel('Depth z (mm)')
    ax1.set_ylabel('Normalized Amplitude')
    ax1.set_title('Axial Profiles Comparison (x=0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Perfis Laterais (DT vs TFM)
    ax2 = axes[0, 1]
    ax2.plot(data['x_dt'], dt_lateral_norm, 'b-', linewidth=2, label='Digital Twin')
    ax2.plot(data['x_tfm'], tfm_lateral_norm, 'r--', linewidth=2, label='TFM Reconstruction')
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.7, label='Center')
    ax2.set_xlabel('Lateral Position x (mm)')
    ax2.set_ylabel('Normalized Amplitude')
    ax2.set_title(f'Lateral Profiles at Focus (z={target_z:.1f}mm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Zoom do Perfil Axial (região focal)
    ax3 = axes[1, 0]
    z_zoom_min = max(target_z - 20, data['z_dt'].min())
    z_zoom_max = min(target_z + 20, data['z_dt'].max())
    
    z_mask_dt = (data['z_dt'] >= z_zoom_min) & (data['z_dt'] <= z_zoom_max)
    z_mask_tfm = (data['z_tfm'] >= z_zoom_min) & (data['z_tfm'] <= z_zoom_max)
    
    if np.any(z_mask_dt) and np.any(z_mask_tfm):
        ax3.plot(data['z_dt'][z_mask_dt], dt_axial_norm[z_mask_dt], 'b-', linewidth=2, label='Digital Twin')
        ax3.plot(data['z_tfm'][z_mask_tfm], tfm_axial_norm[z_mask_tfm], 'r--', linewidth=2, label='TFM')
        ax3.axvline(target_z, color='green', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Depth z (mm)')
        ax3.set_ylabel('Normalized Amplitude')
        ax3.set_title('Axial Profiles - Focal Region Zoom')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Resíduos Axiais
    ax4 = axes[1, 1]
    # Interpolar TFM na grade do DT para calcular resíduos
    from scipy.interpolate import interp1d
    if len(data['z_tfm']) > 1 and len(data['tfm_axial']) > 1:
        tfm_interp = interp1d(data['z_tfm'], tfm_axial_norm, kind='linear', 
                             bounds_error=False, fill_value=0)
        tfm_on_dt_grid = tfm_interp(data['z_dt'])
        
        residuals = dt_axial_norm - tfm_on_dt_grid
        ax4.plot(data['z_dt'], residuals, 'k-', linewidth=1.5)
        ax4.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax4.fill_between(data['z_dt'], 0, residuals, where=residuals>=0, 
                        alpha=0.3, color='blue', label='DT > TFM')
        ax4.fill_between(data['z_dt'], 0, residuals, where=residuals<0, 
                        alpha=0.3, color='red', label='DT < TFM')
        ax4.set_xlabel('Depth z (mm)')
        ax4.set_ylabel('Residual (DT - TFM)')
        ax4.set_title('Axial Profiles Residuals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profiles_comparison_2d.png"), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_and_plot_fwhm_comparison(data, params, output_dir="plots"):
    """
    Calcula e plota comparação de FWHM axial e lateral
    COM LEGENDAS CONSISTENTES COM plot_pipeline_results_FWHM.py
    """
    print(f"[PLOT] Generating FWHM Comparison Charts with Enhanced Legend...")
    
    Dt0 = params.get('Dt0', 59.2)
    target_z = Dt0 + params.get('DF', 23.6)
    theoretical_fwhm = params.get('theoretical_fwhm_mm', 0.0)
    
    # Normalizar perfis para cálculo de FWHM
    dt_axial_norm = data['dt_axial'] / np.max(data['dt_axial'])
    dt_lateral_norm = data['dt_lateral'] / np.max(data['dt_lateral'])
    tfm_axial_norm = data['tfm_axial'] / np.max(data['tfm_axial'])
    tfm_lateral_norm = data['tfm_lateral'] / np.max(data['tfm_lateral'])
    
    # Calcular FWHM para todos os métodos
    # Axial - Digital Twin (na profundidade focal)
    iz_focus_dt = np.argmin(np.abs(data['z_dt'] - target_z))
    dt_axial_results = estimate_all_fwhm_methods(
        data['z_dt'], dt_axial_norm, 
        theoretical_fwhm=theoretical_fwhm, 
        show_plot=False
    )
    
    # Axial - TFM
    search_mask = (data['z_tfm'] > Dt0 + 5.0)
    peak_idx = np.where(search_mask, tfm_axial_norm, -np.inf).argmax()
    fwhm_mask = (data['z_tfm'] >= data['z_tfm'][peak_idx] - 25) & (data['z_tfm'] <= data['z_tfm'][peak_idx] + 25)
    tfm_axial_results = estimate_all_fwhm_methods(
        data['z_tfm'][fwhm_mask], tfm_axial_norm[fwhm_mask],
        theoretical_fwhm=None, 
        show_plot=False
    )
    
    # Lateral - Digital Twin
    dt_lateral_results = estimate_all_fwhm_methods(
        data['x_dt'], dt_lateral_norm,
        theoretical_fwhm=None,
        show_plot=False
    )
    
    # Lateral - TFM
    tfm_lateral_results = estimate_all_fwhm_methods(
        data['x_tfm'], tfm_lateral_norm,
        theoretical_fwhm=None,
        show_plot=False
    )
    
    methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
    
    # 1. Gráfico de barras comparativo Axial vs Lateral COM LEGENDA MELHORADA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    width = 0.35
    x = np.arange(len(methods))
    
    # Axial Comparison - COM CORES DO PLOT_FWHM
    dt_axial_vals = [dt_axial_results.get(m, 0.0) for m in methods]
    tfm_axial_vals = [tfm_axial_results.get(m, 0.0) for m in methods]
    
    # Cores baseadas na precisão (axial usa referência teórica)
    dt_axial_colors = [classify_result(v, theoretical_fwhm, is_axial=True)[0] for v in dt_axial_vals]
    tfm_axial_colors = [classify_result(v, theoretical_fwhm, is_axial=True)[0] for v in tfm_axial_vals]
    
    bars1a = ax1.bar(x - width/2, dt_axial_vals, width, 
                     color=dt_axial_colors, hatch='///', 
                     edgecolor='black', alpha=0.7, label='Digital Twin')
    bars1b = ax1.bar(x + width/2, tfm_axial_vals, width,
                     color=tfm_axial_colors,
                     edgecolor='black', alpha=0.9, label='TFM')
    
    # Linha teórica (Preta, como no FWHM plot)
    if theoretical_fwhm > 0:
        ax1.axhline(theoretical_fwhm, color='black', ls='-', lw=3, 
                   label=f'Theoretical Baseline ({theoretical_fwhm:.2f} mm)')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel('FWHM (mm)', fontweight='bold')
    ax1.set_title(f'Axial FWHM: Deviation from Theoretical Value', fontsize=20, fontweight='bold')
    
    # Anotar diferenças absolutas (como no FWHM plot)
    for i, (v_dt, v_tfm) in enumerate(zip(dt_axial_vals, tfm_axial_vals)):
        if theoretical_fwhm > 0 and v_dt > 0 and v_tfm > 0:
            # Diferença absoluta como no plot FWHM
            diff_dt = v_dt - theoretical_fwhm
            diff_tfm = v_tfm - theoretical_fwhm
            ax1.text(i - width/2, v_dt + 0.1, f'{diff_dt:+.1f}', 
                    ha='center', fontsize=16, fontweight='bold')
            ax1.text(i + width/2, v_tfm + 0.1, f'{diff_tfm:+.1f}', 
                    ha='center', fontsize=16, fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # LEGENDA DO LADO DIREITO (como no FWHM plot)
    # Remover legenda automática e adicionar legenda customizada
    ax1.legend().remove()
    legend_elements_axial = create_legend_elements(include_theoretical=True, include_patterns=True)
    ax1.legend(handles=legend_elements_axial, loc='upper left', bbox_to_anchor=(1.05, 1), 
              fontsize=14, frameon=True, shadow=True)
    
    # Lateral Comparison - COM CORES DO PLOT_FWHM
    dt_lateral_vals = [dt_lateral_results.get(m, 0.0) for m in methods]
    tfm_lateral_vals = [tfm_lateral_results.get(m, 0.0) for m in methods]
    
    # Para lateral, comparar com Digital Twin como referência
    dt_lateral_colors = ['steelblue'] * len(methods)  # Azul padrão para DT como no FWHM plot
    tfm_lateral_colors = []
    for i, v_tfm in enumerate(tfm_lateral_vals):
        v_dt_ref = dt_lateral_vals[i] if dt_lateral_vals[i] > 0 else None
        color, _ = classify_result(v_tfm, v_dt_ref, is_axial=False)
        tfm_lateral_colors.append(color)
    
    bars2a = ax2.bar(x - width/2, dt_lateral_vals, width,
                     color=dt_lateral_colors, hatch='///',
                     edgecolor='black', alpha=0.7, label='Digital Twin')
    bars2b = ax2.bar(x + width/2, tfm_lateral_vals, width,
                     color=tfm_lateral_colors,
                     edgecolor='black', alpha=0.9, label='TFM')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('FWHM (mm)', fontweight='bold')
    ax2.set_title('Lateral FWHM: TFM vs Digital Twin Reference', fontsize=20, fontweight='bold')
    
    # Anotar diferenças percentuais
    for i, (v_dt, v_tfm) in enumerate(zip(dt_lateral_vals, tfm_lateral_vals)):
        if v_dt > 0 and v_tfm > 0:
            diff_pct = ((v_tfm - v_dt) / v_dt) * 100
            ax2.text(i, max(v_dt, v_tfm) + 0.5, f'{diff_pct:+.1f}%', 
                    ha='center', fontweight='bold', color='darkred', fontsize=16)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # LEGENDA DO LADO DIREITO (sem linha teórica para lateral)
    ax2.legend().remove()
    legend_elements_lateral = create_legend_elements(include_theoretical=False, include_patterns=True)
    ax2.legend(handles=legend_elements_lateral, loc='upper left', bbox_to_anchor=(1.05, 1), 
              fontsize=14, frameon=True, shadow=True)
    
    # Ajustar layout para acomodar legendas
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Deixar espaço à direita para legendas
    
    plt.savefig(os.path.join(output_dir, "fwhm_axial_vs_lateral_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico de radar/spider para mostrar todas as métricas
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Converter para coordenadas polares
    angles = np.linspace(0, 2*np.pi, len(methods), endpoint=False).tolist()
    angles += angles[:1]  # Fechar o polígono
    
    # Valores normalizados (0-1) para melhor visualização
    def normalize_vals(vals):
        max_val = max(v for v in vals if v > 0)
        return [v/max_val if v > 0 else 0 for v in vals]
    
    dt_axial_norm = normalize_vals(dt_axial_vals)
    tfm_axial_norm = normalize_vals(tfm_axial_vals)
    dt_lateral_norm = normalize_vals(dt_lateral_vals)
    tfm_lateral_norm = normalize_vals(tfm_lateral_vals)
    
    # Fechar polígonos
    dt_axial_norm += dt_axial_norm[:1]
    tfm_axial_norm += tfm_axial_norm[:1]
    dt_lateral_norm += dt_lateral_norm[:1]
    tfm_lateral_norm += tfm_lateral_norm[:1]
    
    # Plotar
    ax.plot(angles, dt_axial_norm, 'b-', linewidth=2, label='DT Axial')
    ax.fill(angles, dt_axial_norm, 'b', alpha=0.1)
    
    ax.plot(angles, tfm_axial_norm, 'r-', linewidth=2, label='TFM Axial')
    ax.fill(angles, tfm_axial_norm, 'r', alpha=0.1)
    
    ax.plot(angles, dt_lateral_norm, 'g--', linewidth=2, label='DT Lateral')
    ax.fill(angles, dt_lateral_norm, 'g', alpha=0.1)
    
    ax.plot(angles, tfm_lateral_norm, 'm--', linewidth=2, label='TFM Lateral')
    ax.fill(angles, tfm_lateral_norm, 'm', alpha=0.1)
    
    # Configurações
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.2)
    ax.set_title('FWHM Methods Radar Chart (Normalized)', fontsize=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fwhm_methods_radar_chart.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'dt_axial': dt_axial_results,
        'tfm_axial': tfm_axial_results,
        'dt_lateral': dt_lateral_results,
        'tfm_lateral': tfm_lateral_results
    }

def generate_comprehensive_report(fwhm_results, params, data, output_dir="plots"):
    """
    Gera relatório CSV completo com todas as métricas
    """
    print(f"[REPORT] Generating Comprehensive CSV Report...")
    
    Dt0 = params.get('Dt0', 59.2)
    target_z = Dt0 + params.get('DF', 23.6)
    theoretical_fwhm = params.get('theoretical_fwhm_mm', 0.0)
    
    methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
    
    report_path = os.path.join(output_dir, "comprehensive_analysis_report.csv")
    
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Cabeçalho
        writer.writerow(['Comprehensive Pipeline Analysis Report'])
        writer.writerow(['Generated by: Enhanced 2D Pipeline Plotter'])
        writer.writerow([''])
        writer.writerow(['Pipeline Parameters'])
        writer.writerow(['Parameter', 'Value', 'Units'])
        for key, value in params.items():
            if isinstance(value, (int, float, str)):
                writer.writerow([key, value, ''])
        writer.writerow([''])
        
        # Seção de métricas gerais
        writer.writerow(['General Metrics'])
        writer.writerow(['Metric', 'Value', 'Units', 'Notes'])
        
        # Encontrar pico no TFM
        search_mask = (data['z_tfm'] > Dt0 + 5.0)
        peak_idx = np.where(search_mask, data['tfm_axial'], -np.inf).argmax()
        peak_z = data['z_tfm'][peak_idx]
        peak_value = data['tfm_axial'][peak_idx]
        
        writer.writerow(['TFM Peak Depth', f'{peak_z:.2f}', 'mm', 
                        f'Target was {target_z:.1f} mm'])
        writer.writerow(['TFM Peak Value', f'{peak_value:.4f}', '', ''])
        writer.writerow(['Theoretical FWHM', f'{theoretical_fwhm:.4f}', 'mm', ''])
        writer.writerow(['Digital Twin FWHM (F1)', 
                        f"{fwhm_results['dt_axial'].get('F1', 0.0):.4f}", 
                        'mm', 'Axial at focus'])
        writer.writerow([''])
        
        # Seção de FWHM detalhada
        writer.writerow(['FWHM Analysis'])
        writer.writerow(['Method', 'DT Axial (mm)', 'TFM Axial (mm)', 
                        'DT Lateral (mm)', 'TFM Lateral (mm)', 
                        'Axial Error vs Theory (%)', 'Lateral Error vs DT (%)',
                        'Axial Quality', 'Lateral Quality'])
        
        for method in methods:
            dt_axial = fwhm_results['dt_axial'].get(method, 0.0)
            tfm_axial = fwhm_results['tfm_axial'].get(method, 0.0)
            dt_lateral = fwhm_results['dt_lateral'].get(method, 0.0)
            tfm_lateral = fwhm_results['tfm_lateral'].get(method, 0.0)
            
            # Calcular erros
            axial_error = 0.0
            if theoretical_fwhm > 0 and tfm_axial > 0:
                axial_error = ((tfm_axial - theoretical_fwhm) / theoretical_fwhm) * 100
            
            lateral_error = 0.0
            if dt_lateral > 0 and tfm_lateral > 0:
                lateral_error = ((tfm_lateral - dt_lateral) / dt_lateral) * 100
            
            # Classificar qualidade usando o mesmo esquema
            axial_color, axial_quality = classify_result(tfm_axial, theoretical_fwhm, is_axial=True)
            lateral_color, lateral_quality = classify_result(tfm_lateral, dt_lateral, is_axial=False)
            
            writer.writerow([
                method,
                f'{dt_axial:.4f}',
                f'{tfm_axial:.4f}',
                f'{dt_lateral:.4f}',
                f'{tfm_lateral:.4f}',
                f'{axial_error:.2f}',
                f'{lateral_error:.2f}',
                axial_quality,
                lateral_quality
            ])
        
        writer.writerow([''])
        
        # Resumo estatístico
        writer.writerow(['Statistical Summary'])
        writer.writerow(['Metric', 'Mean', 'Std Dev', 'Min', 'Max', 'Units'])
        
        # Calcular estatísticas para TFM Axial
        tfm_axial_vals = [fwhm_results['tfm_axial'].get(m, 0.0) for m in methods]
        tfm_axial_vals = [v for v in tfm_axial_vals if v > 0]
        
        if tfm_axial_vals:
            writer.writerow([
                'TFM Axial FWHM',
                f'{np.mean(tfm_axial_vals):.4f}',
                f'{np.std(tfm_axial_vals):.4f}',
                f'{np.min(tfm_axial_vals):.4f}',
                f'{np.max(tfm_axial_vals):.4f}',
                'mm'
            ])
        
        # Calcular estatísticas para TFM Lateral
        tfm_lateral_vals = [fwhm_results['tfm_lateral'].get(m, 0.0) for m in methods]
        tfm_lateral_vals = [v for v in tfm_lateral_vals if v > 0]
        
        if tfm_lateral_vals:
            writer.writerow([
                'TFM Lateral FWHM',
                f'{np.mean(tfm_lateral_vals):.4f}',
                f'{np.std(tfm_lateral_vals):.4f}',
                f'{np.min(tfm_lateral_vals):.4f}',
                f'{np.max(tfm_lateral_vals):.4f}',
                'mm'
            ])
        
        # Razão Lateral/Axial
        if tfm_axial_vals and tfm_lateral_vals:
            ratio_mean = np.mean(tfm_lateral_vals) / np.mean(tfm_axial_vals) if np.mean(tfm_axial_vals) > 0 else 0
            writer.writerow([
                'Lateral/Axial Ratio',
                f'{ratio_mean:.4f}',
                '', '', '',
                ''
            ])
    
    print(f"✅ Comprehensive report saved to: {report_path}")

def main():
    """Função principal"""
    
    # 1. Localizar arquivo de parâmetros
    possible_paths = [
        os.path.join("results/run12", "run_params.json"),
        "run_params.json"
    ]
    
    param_file = next((p for p in possible_paths if os.path.exists(p)), None)
    if not param_file:
        print("[ERROR] run_params.json not found in any expected location.")
        print("Searched in:", possible_paths)
        return
    
    print(f"[INFO] Using parameters from: {param_file}")
    
    # 2. Carregar parâmetros
    try:
        with open(param_file, "r", encoding='utf-8') as f:
            params = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load parameters: {e}")
        return
    
    out_root = os.path.dirname(param_file)
    print(f"[INFO] Output root directory: {out_root}")
    
    # 3. Preparar diretório de saída
    ensure_dir("plots")
    
    # 4. Carregar dados
    data = load_data_2d(out_root)
    if data is None:
        print("[ERROR] Failed to load data. Exiting.")
        return
    
    print("[INFO] Data loaded successfully:")
    print(f"  Digital Twin: {data['p_field'].shape[1]} x {data['p_field'].shape[0]} points")
    print(f"  TFM: {data['envelope_2d'].shape[0]} x {data['envelope_2d'].shape[1]} points")
    
    # 5. Gerar todos os plots
    plot_field_maps_2d(data, params)
    plot_profiles_comparison(data, params)
    fwhm_results = calculate_and_plot_fwhm_comparison(data, params)
    
    # 6. Gerar relatório
    generate_comprehensive_report(fwhm_results, params, data)
    
    # 7. Plot adicional: Resumo em uma página
    print("[PLOT] Generating One-Page Summary...")
    create_one_page_summary(data, params, fwhm_results)
    
    print(f"\n{'='*60}")
    print("✅ PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print("Generated files in 'plots/' directory:")
    print("  1. field_maps_2d_comparison.png - Full 2D field maps")
    print("  2. profiles_comparison_2d.png - Axial/Lateral profiles")
    print("  3. fwhm_axial_vs_lateral_comparison.png - FWHM comparison")
    print("  4. fwhm_methods_radar_chart.png - Radar chart of methods")
    print("  5. comprehensive_analysis_report.csv - Full analysis report")
    print("  6. one_page_summary.png - Quick overview")
    print(f"{'='*60}")

def create_one_page_summary(data, params, fwhm_results):
    """
    Cria um resumo de uma página com os resultados mais importantes
    """
    fig = plt.figure(figsize=(20, 15))
    
    Dt0 = params.get('Dt0', 59.2)
    target_z = Dt0 + params.get('DF', 23.6)
    theoretical_fwhm = params.get('theoretical_fwhm_mm', 0.0)
    
    # 1. TFM 2D Map (top left)
    ax1 = plt.subplot(3, 3, 1)
    envelope_norm = data['envelope_2d'] / np.max(data['envelope_2d'])
    im1 = ax1.pcolormesh(data['z_tfm'], data['x_tfm'], envelope_norm,
                        cmap='hot', vmin=0, vmax=1, shading='auto')
    ax1.axvline(Dt0, color='cyan', ls='--', alpha=0.7, label='Interface')
    ax1.axvline(target_z, color='white', ls='--', alpha=0.7, label='Target')
    ax1.set_xlabel('Depth (mm)')
    ax1.set_ylabel('Lateral (mm)')
    ax1.set_title('TFM Reconstruction')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.legend(fontsize=9, loc='upper right')
    
    # 2. Axial Profile Comparison (top middle)
    ax2 = plt.subplot(3, 3, 2)
    dt_axial_norm = data['dt_axial'] / np.max(data['dt_axial'])
    tfm_axial_norm = data['tfm_axial'] / np.max(data['tfm_axial'])
    
    ax2.plot(data['z_dt'], dt_axial_norm, 'b-', linewidth=2, label='Digital Twin')
    ax2.plot(data['z_tfm'], tfm_axial_norm, 'r--', linewidth=2, label='TFM')
    ax2.axvline(target_z, color='green', ls=':', alpha=0.7, label='Target')
    ax2.set_xlabel('Depth (mm)')
    ax2.set_ylabel('Normalized Amplitude')
    ax2.set_title('Axial Profiles (x=0)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Lateral Profile Comparison (top right)
    ax3 = plt.subplot(3, 3, 3)
    dt_lateral_norm = data['dt_lateral'] / np.max(data['dt_lateral'])
    tfm_lateral_norm = data['tfm_lateral'] / np.max(data['tfm_lateral'])
    
    ax3.plot(data['x_dt'], dt_lateral_norm, 'b-', linewidth=2, label='Digital Twin')
    ax3.plot(data['x_tfm'], tfm_lateral_norm, 'r--', linewidth=2, label='TFM')
    ax3.axvline(0, color='gray', ls=':', alpha=0.5, label='Center')
    ax3.set_xlabel('Lateral Position (mm)')
    ax3.set_ylabel('Normalized Amplitude')
    ax3.set_title(f'Lateral Profiles (z={target_z:.1f}mm)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. FWHM Comparison Bar Chart (middle left)
    ax4 = plt.subplot(3, 3, 4)
    methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
    
    dt_axial_vals = [fwhm_results['dt_axial'].get(m, 0.0) for m in methods]
    tfm_axial_vals = [fwhm_results['tfm_axial'].get(m, 0.0) for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax4.bar(x - width/2, dt_axial_vals, width, color='steelblue', alpha=0.6, label='DT')
    ax4.bar(x + width/2, tfm_axial_vals, width, color='seagreen', alpha=0.6, label='TFM')
    
    if theoretical_fwhm > 0:
        ax4.axhline(theoretical_fwhm, color='black', ls='--', 
                   linewidth=2, label=f'Theory: {theoretical_fwhm:.2f}mm')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, rotation=45)
    ax4.set_ylabel('FWHM (mm)')
    ax4.set_title('Axial FWHM Methods Comparison')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Lateral FWHM Comparison (middle middle)
    ax5 = plt.subplot(3, 3, 5)
    dt_lateral_vals = [fwhm_results['dt_lateral'].get(m, 0.0) for m in methods]
    tfm_lateral_vals = [fwhm_results['tfm_lateral'].get(m, 0.0) for m in methods]
    
    ax5.bar(x - width/2, dt_lateral_vals, width, color='steelblue', alpha=0.6, label='DT')
    ax5.bar(x + width/2, tfm_lateral_vals, width, color='seagreen', alpha=0.6, label='TFM')
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(methods, rotation=45)
    ax5.set_ylabel('FWHM (mm)')
    ax5.set_title('Lateral FWHM Methods Comparison')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. FWHM Ratio Lateral/Axial (middle right)
    ax6 = plt.subplot(3, 3, 6)
    
    ratios = []
    for i in range(len(methods)):
        if tfm_axial_vals[i] > 0:
            ratio = tfm_lateral_vals[i] / tfm_axial_vals[i]
            ratios.append(ratio)
        else:
            ratios.append(0)
    
    bars = ax6.bar(x, ratios, color='purple', alpha=0.7)
    ax6.axhline(1.0, color='red', ls='--', alpha=0.5, label='Ideal (1:1)')
    
    # Colorir barras baseado no desvio do ideal
    for i, bar in enumerate(bars):
        if ratios[i] > 1.5:
            bar.set_color('red')
        elif ratios[i] > 1.2:
            bar.set_color('orange')
        elif ratios[i] >= 0.8:
            bar.set_color('green')
        else:
            bar.set_color('blue')
    
    ax6.set_xticks(x)
    ax6.set_xticklabels(methods, rotation=45)
    ax6.set_ylabel('Ratio (Lateral/Axial)')
    ax6.set_title('FWHM Lateral/Axial Ratio')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Text Summary (bottom row, spanning 3 columns)
    ax7 = plt.subplot(3, 1, 3)
    ax7.axis('off')
    
    # Encontrar métricas chave
    search_mask = (data['z_tfm'] > Dt0 + 5.0)
    peak_idx = np.where(search_mask, data['tfm_axial'], -np.inf).argmax()
    peak_z = data['z_tfm'][peak_idx]
    peak_value = data['tfm_axial'][peak_idx]
    
    # FWHM médios
    tfm_axial_vals_all = [fwhm_results['tfm_axial'].get(m, 0.0) for m in methods]
    tfm_axial_vals_valid = [v for v in tfm_axial_vals_all if v > 0]
    tfm_axial_mean = np.mean(tfm_axial_vals_valid) if tfm_axial_vals_valid else 0
    
    tfm_lateral_vals_all = [fwhm_results['tfm_lateral'].get(m, 0.0) for m in methods]
    tfm_lateral_vals_valid = [v for v in tfm_lateral_vals_all if v > 0]
    tfm_lateral_mean = np.mean(tfm_lateral_vals_valid) if tfm_lateral_vals_valid else 0
    
    # Calcular erros
    depth_error = abs(peak_z - target_z) / target_z * 100 if target_z > 0 else 0
    fwhm_error = abs(tfm_axial_mean - theoretical_fwhm) / theoretical_fwhm * 100 if theoretical_fwhm > 0 else 0
    
    # Calcular razão lateral/axial (com verificação de divisão por zero)
    if tfm_axial_mean > 0:
        lateral_axial_ratio = tfm_lateral_mean / tfm_axial_mean
    else:
        lateral_axial_ratio = 0
    
    # Criar texto de resumo
    summary_text = (
        f"PIPELINE ANALYSIS SUMMARY\n\n"
        f"Target Parameters:\n"
        f"  • Focus Depth: {target_z:.1f} mm\n"
        f"  • Theoretical FWHM: {theoretical_fwhm:.2f} mm\n\n"
        f"TFM Results:\n"
        f"  • Peak Depth: {peak_z:.1f} mm ({depth_error:.1f}% error)\n"
        f"  • Axial FWHM: {tfm_axial_mean:.2f} mm ({fwhm_error:.1f}% error)\n"
        f"  • Lateral FWHM: {tfm_lateral_mean:.2f} mm\n"
        f"  • Lateral/Axial Ratio: {lateral_axial_ratio:.2f}\n\n"
        f"Quality Assessment:\n"
    )
    
    # Adicionar avaliação qualitativa usando o mesmo esquema de cores
    if depth_error < 5 and fwhm_error < 15:
        summary_text += "  • ✅ EXCELLENT: Both depth and FWHM within specifications\n"
    elif depth_error < 10 and fwhm_error < 25:
        summary_text += "  • ⚠️ GOOD: Minor deviations from target\n"
    elif depth_error < 20 or fwhm_error < 40:
        summary_text += "  • ⚠️ MODERATE: Significant deviations detected\n"
    else:
        summary_text += "  • ❌ POOR: Large deviations from target\n"
    
    ax7.text(0.02, 0.95, summary_text, fontsize=14, 
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Enhanced 2D Pipeline Analysis Summary', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plots/one_page_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()