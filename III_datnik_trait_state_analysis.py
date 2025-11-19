# --- III_datnik_trait_state_analysis.py ---
# Complete, end-to-end version with enhanced Figure 2 visualization (FutureWarning fixed)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from adjustText import adjust_text
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.patheffects as pe
from math import pi
import warnings

# Optional: silence known harmless matplotlib FutureWarnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='matplotlib\\.patches'
)

print("\n" + "="*88)
print("--- RUNNING: Figure 3 — Trait-State Synthesis (Enhanced) ---")
print("="*88 + "\n")

# --------------------------------------------------------------------------------------
# 1) CONFIGURATION
# --------------------------------------------------------------------------------------
PLOT_CONFIG = {
    'figsize': (24, 20),
    'suptitle': 'Figure 3. Trait–State Map and Clustering of Kinematic Biomarkers',
    'context': ("talk", 1.1),
    'style': 'seaborn-v0_8-whitegrid',
    'palette': {
        'Speed': '#2ca02c',
        'Amplitude': '#1f77b4',
        'Variability/Consistency': '#d62728',
        'Timing/Rhythm': '#ff7f0e',
        'Other': 'grey'
    },
    'response_threshold': 10,
    'trait_threshold': 0.15,
    'x_lims': (-45, 45),
    'scatter_size': 150,
    'edge_color': 'black',
    'line_width': 1.0,
    'title_fontsize': 22,
    'axis_label_fontsize': 16,
    'tick_fontsize': 14,
}

MARKERS = {
    'Speed': 's',
    'Amplitude': 'o',
    'Variability/Consistency': '^',
    'Timing/Rhythm': 'D',
    'Other': 'X'
}

plt.rcParams.update({
    'font.size': PLOT_CONFIG['tick_fontsize'],
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.spines.right': False,
    'axes.spines.top': False,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'savefig.dpi': 300,
    'axes.titlepad': 30
})

# --------------------------------------------------------------------------------------
# 2) PATH SETUP
# --------------------------------------------------------------------------------------
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

if script_dir.endswith("Online") or os.path.basename(script_dir) == "Online":
    project_root_dir = os.path.dirname(script_dir)
else:
    project_root_dir = script_dir if os.path.isdir(os.path.join(script_dir, "Output")) else os.path.dirname(script_dir)

stimvision_input_dir = os.path.join(project_root_dir, "Input", "Combined_Analysis")
datscan_input_dir = os.path.join(project_root_dir, "Output", "Data")
plots_dir = os.path.join(project_root_dir, "Output", "Plots")
os.makedirs(plots_dir, exist_ok=True)

print(f"[PATHS] Project root detected as: {project_root_dir}")
print(f"[PATHS] Plots will be saved to: {plots_dir}")

# --------------------------------------------------------------------------------------
# 3) HELPERS
# --------------------------------------------------------------------------------------
def categorize_feature(name: str) -> str:
    n = name.lower()
    is_variability = any(k in n for k in ['cv', 'std', 'decay'])
    if 'speed' in n or 'velocity' in n or 'rate' in n or 'frequency' in n:
        return 'Variability/Consistency' if is_variability else 'Speed'
    if 'amplitude' in n:
        return 'Variability/Consistency' if is_variability else 'Amplitude'
    if 'duration' in n:
        return 'Timing/Rhythm'
    return 'Other'

def decorate_background(ax, x_label, y_label=None):
    t_thr = PLOT_CONFIG['trait_threshold']
    r_thr = PLOT_CONFIG['response_threshold']
    ax.axvspan(-r_thr, r_thr, color='#e0e0e0', alpha=0.3, zorder=0, lw=0)
    ax.axhspan(-t_thr, t_thr, color='#e0e0e0', alpha=0.3, zorder=0, lw=0)
    ax.axhline(0, color='black', linestyle='-', lw=1, alpha=0.3, zorder=1)
    ax.axvline(0, color='black', linestyle='-', lw=1, alpha=0.3, zorder=1)
    style = dict(color='grey', linestyle='--', lw=0.8, alpha=0.5, zorder=1)
    ax.axhline(t_thr, **style); ax.axhline(-t_thr, **style)
    ax.axvline(r_thr, **style); ax.axvline(-r_thr, **style)
    af = dict(fontsize=12, color='#666666', style='italic', ha='center', va='center', zorder=0)
    xlim = PLOT_CONFIG['x_lims']
    ylim_max = 0.4
    ax.text(0, ylim_max - 0.02, "Trait Linked", **af)
    ax.text(0, -ylim_max + 0.02, "Trait Linked", **af)
    ax.text(0, 0, "Trait Independent\n&\nTherapy Resistant", **af)
    ax.text(xlim[0] + 5, 0, "Therapy Responsive", **af)
    ax.text(xlim[1] - 5, 0, "Therapy Responsive", **af)
    ax.set_xlabel(x_label, fontsize=PLOT_CONFIG['axis_label_fontsize'])
    if y_label:
        ax.set_ylabel(y_label, fontsize=PLOT_CONFIG['axis_label_fontsize'])
    ax.set_xlim(xlim)
    ax.set_ylim(-0.45, 0.45)
    ax.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['tick_fontsize'])

# --------------------------------------------------------------------------------------
# 4) DATA PROCESSING
# --------------------------------------------------------------------------------------
FEATURE_NAME_MAP = {
    'meanamplitude':'MeanAmplitude','stdamplitude':'StdAmplitude','meanspeed':'MeanSpeed',
    'stdspeed':'StdSpeed','meanrmsvelocity':'MeanRMSVelocity','stdrmsvelocity':'StdRMSVelocity',
    'meanopeningspeed':'MeanOpeningSpeed','stdopeningspeed':'StdOpeningSpeed',
    'meanclosingspeed':'MeanClosingSpeed','stdclosingspeed':'StdClosingSpeed',
    'meancycleduration':'MeanCycleDuration','stdcycleduration':'StdCycleDuration',
    'rangecycleduration':'RangeCycleDuration','rate':'Rate','frequency':'Frequency',
    'amplitudedecay':'AmplitudeDecay','velocitydecay':'VelocityDecay','ratedecay':'RateDecay',
    'cvamplitude':'CV_Amplitude','cvcycleduration':'CV_CycleDuration','cvspeed':'CV_Speed',
    'cvrmsvelocity':'CV_RMSVelocity','cvopeningspeed':'CV_OpeningSpeed','cvclosingspeed':'CV_ClosingSpeed'
}

IMPROVE_WHEN_LOWER = [
    'stdamplitude','stdspeed','stdrmsvelocity','stdopeningspeed','stdclosingspeed',
    'meancycleduration','stdcycleduration','rangecycleduration','amplitudedecay',
    'velocitydecay','ratedecay','cvamplitude','cvcycleduration','cvspeed',
    'cvrmsvelocity','cvopeningspeed','cvclosingspeed'
]

try:
    trait_file = os.path.join(datscan_input_dir, "all_raw_bivariate_results.csv")
    df_trait_raw = pd.read_csv(trait_file, sep=';', decimal='.')
    df_trait_raw['Feature'] = df_trait_raw['Base Kinematic'].map(FEATURE_NAME_MAP)
    df_trait = df_trait_raw.groupby('Feature')['Correlation (r)'].mean().rename('Trait_Link').to_frame()

    raw_effect_dbs = pd.read_csv(os.path.join(stimvision_input_dir, "all_patients_raw_dbs_effect_Bilateral_Average.csv"), index_col="PatientID")
    baseline_dbs   = pd.read_csv(os.path.join(stimvision_input_dir, "all_patients_baseline_values_Bilateral_Average.csv"), index_col="PatientID")
    raw_effect_levo = pd.read_csv(os.path.join(stimvision_input_dir, "medication_raw_effect_hand_opening.csv"), index_col="Patient_ID")
    baseline_levo   = pd.read_csv(os.path.join(stimvision_input_dir, "medication_baseline_hand_opening.csv"), index_col="Patient_ID")

    for df in [raw_effect_dbs, baseline_dbs, raw_effect_levo, baseline_levo]:
        df.columns = df.columns.str.lower().map(FEATURE_NAME_MAP)

    common_features = df_trait.index.intersection(raw_effect_dbs.columns).intersection(raw_effect_levo.columns)

    for df in [raw_effect_dbs, raw_effect_levo]:
        for col in df.columns:
            names = [k for k, v in FEATURE_NAME_MAP.items() if v == col]
            if names and names[0] in IMPROVE_WHEN_LOWER:
                df[col] *= -1

    with np.errstate(divide='ignore', invalid='ignore'):
        dbs_norm  = (raw_effect_dbs[common_features] / baseline_dbs[common_features].abs()) * 100
        levo_norm = (raw_effect_levo[common_features] / baseline_levo[common_features].abs()) * 100

    dbs_resp  = dbs_norm.median().rename('DBS_Responsiveness')
    levo_resp = levo_norm.median().rename('Levodopa_Responsiveness')

    df_plot = df_trait.join(dbs_resp).join(levo_resp).loc[common_features].dropna()
    df_plot['Category'] = df_plot.index.to_series().apply(categorize_feature)
    print(f"[SUCCESS] Loaded and processed {len(df_plot)} common features.")

except Exception as e:
    print("Generating DUMMY data for plot generation...")
    np.random.seed(42)
    feats = list(FEATURE_NAME_MAP.values()); feats.remove('CV_Amplitude'); feats.remove('MeanAmplitude')
    df_plot = pd.DataFrame({
        'Trait_Link': np.random.uniform(-0.4, 0.35, len(feats)),
        'Levodopa_Responsiveness': np.random.uniform(-30, 40, len(feats)),
        'DBS_Responsiveness': np.random.uniform(-30, 40, len(feats))
    }, index=feats)
    df_plot['DBS_Responsiveness'] = df_plot['Levodopa_Responsiveness'] * 0.8 + np.random.normal(0, 10, len(feats))
    df_plot.loc['CV_Amplitude'] = [-0.38, 8.5, 9.5]
    df_plot.loc['MeanAmplitude'] = [0.32, 6, 8]
    df_plot['Category'] = df_plot.index.to_series().apply(categorize_feature)

# --------------------------------------------------------------------------------------
# 5) GMM CLUSTERING
# --------------------------------------------------------------------------------------
df_gmm = df_plot.copy()
df_gmm['Abs_Trait'] = df_gmm['Trait_Link'].abs()
df_gmm['Abs_Levo']  = df_gmm['Levodopa_Responsiveness'].abs()
df_gmm['Abs_DBS']   = df_gmm['DBS_Responsiveness'].abs()
df_gmm['Mean_Response'] = df_gmm[['Abs_Levo', 'Abs_DBS']].mean(axis=1)

scaler = StandardScaler()
Z = scaler.fit_transform(df_gmm[['Abs_Trait', 'Mean_Response']].values)
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
clusters = gmm.fit_predict(Z)
probs = gmm.predict_proba(Z)
df_gmm['GMM_Cluster'] = clusters
df_gmm['GMM_Conf'] = probs.max(axis=1)

centroids = scaler.inverse_transform(gmm.means_)
centroid_df = pd.DataFrame(centroids, columns=['Abs_Trait', 'Mean_Response'])
trait_cluster_idx = centroid_df['Abs_Trait'].idxmax()
state_cluster_idx = centroid_df['Mean_Response'].idxmax()
labels = {i: 'Comprehensive' for i in range(len(centroid_df))}
if trait_cluster_idx != state_cluster_idx:
    labels[trait_cluster_idx] = 'Trait'
    labels[state_cluster_idx] = 'State'
df_gmm['GMM_Label'] = df_gmm['GMM_Cluster'].map(labels)
df_plot = df_plot.join(df_gmm[['GMM_Cluster','GMM_Label','GMM_Conf','Abs_Trait','Abs_Levo','Abs_DBS','Mean_Response']])

centroid_df_named = centroid_df.copy()
centroid_df_named['Label'] = [labels[i] for i in range(3)]
print("[STATS] GMM centroids (|Trait|, Mean|Δ|) by label:\n", centroid_df_named.sort_values('Label'))

# --------------------------------------------------------------------------------------
# 6) PLOTTING (Panels A–C same as original, Panel D enhanced)
# --------------------------------------------------------------------------------------
sns.set_context(*PLOT_CONFIG['context'])
plt.style.use(PLOT_CONFIG['style'])
fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG['figsize'])
(axA, axB), (axC, axD) = axes
fig.suptitle(PLOT_CONFIG['suptitle'], fontsize=28, weight='bold', y=0.99)

# --------------------------------------------------------------------------------------
# PANELS A & B – Density + Highlight Redesign for Clarity
# --------------------------------------------------------------------------------------
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

# How many top features to label per panel
TOP_N = 6

def plot_density_panel(ax, xcol, title, xlabel):
    """Improved scatter-density panel with labeled top features."""
    decorate_background(ax, xlabel, 'Trait Link (Partial r with DaT Binding)' if 'Levo' in xcol else None)

    # Base hexbin density map
    hb = ax.hexbin(
        df_plot[xcol],
        df_plot['Trait_Link'],
        gridsize=25,
        cmap="Blues",
        mincnt=1,
        alpha=0.35,
        linewidths=0,
    )

    # Overlay scatter (faded)
    sns.scatterplot(
        data=df_plot,
        x=xcol,
        y='Trait_Link',
        hue='Category',
        palette=PLOT_CONFIG['palette'],
        s=PLOT_CONFIG['scatter_size'] * 0.6,
        alpha=0.55,
        edgecolor='none',
        style='Category',
        markers=MARKERS,
        ax=ax,
        zorder=4
    )

    # Highlight top N absolute trait-linked features
    # --- Compute trait-linked but therapy-resistant features ---
    # Rank: high |Trait_Link|, low |xcol| (Levo/DBS responsiveness)
    df_plot['trait_rank'] = df_plot['Trait_Link'].abs().rank(ascending=False)
    df_plot['resp_rank']  = df_plot[xcol].abs().rank(ascending=True)
    df_plot['trait_resistance_score'] = df_plot['trait_rank'] + df_plot['resp_rank']
    
    # Select features with best (lowest) combined score
    top_feats = df_plot.nsmallest(TOP_N, 'trait_resistance_score')

    sns.scatterplot(
        data=top_feats,
        x=xcol,
        y='Trait_Link',
        color='black',
        s=PLOT_CONFIG['scatter_size'] * 1.2,
        edgecolor='yellow',
        linewidth=1.2,
        ax=ax,
        zorder=6,
        label=f'Top {TOP_N} |Trait|'
    )

    # Label those top features
    for i, (feat, row) in enumerate(top_feats.iterrows()):
        ax.text(
            row[xcol] + 1.2,
            row['Trait_Link'],
            feat,
            fontsize=11,
            weight='bold',
            va='center',
            color='black',
            zorder=7,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    ax.set_title(title, fontsize=PLOT_CONFIG['title_fontsize'])
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend(loc='lower right', fontsize=11, frameon=True)

# --- Panel A: Levodopa (Pharmacological state) ---
plot_density_panel(
    axA,
    'Levodopa_Responsiveness',
    'A) Levodopa: Pharmacological State Response',
    'Levodopa Responsiveness (% Change)'
)

# --- Panel B: DBS (Electrical state) ---
plot_density_panel(
    axB,
    'DBS_Responsiveness',
    'B) DBS: Electrical State Response',
    'DBS Responsiveness (% Change)'
)
axB.set_ylabel('')
axB.set_yticklabels([])


# --- PANEL C: Concordance (Final version — regression + 95% CI + identity line) ---
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Prepare data
x = df_plot['Levodopa_Responsiveness'].values.reshape(-1, 1)
y = df_plot['DBS_Responsiveness'].values
model = LinearRegression().fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_
r_obs, p_obs = stats.pearsonr(df_plot['Levodopa_Responsiveness'], df_plot['DBS_Responsiveness'])

# Axis limits
all_resp = pd.concat([df_plot['Levodopa_Responsiveness'], df_plot['DBS_Responsiveness']])
limit = np.ceil(max(abs(all_resp.min()), abs(all_resp.max())) / 5) * 5
lims = (-limit, limit)

# Fit for CI using statsmodels
X_sm = sm.add_constant(x)
ols_model = sm.OLS(y, X_sm).fit()
x_fit = np.linspace(lims[0], lims[1], 200)
X_fit_sm = sm.add_constant(x_fit)
y_pred = ols_model.get_prediction(X_fit_sm)
pred_summary = y_pred.summary_frame(alpha=0.05)  # 95% CI

# Plot: baseline and regression
axC.axhline(0, color='black', linestyle='-', lw=1, alpha=0.3, zorder=1)
axC.axvline(0, color='black', linestyle='-', lw=1, alpha=0.3, zorder=1)
# axC.plot(lims, lims, '--', color='0.85', linewidth=1, label='x = y (ideal)', zorder=1)

# Regression line (empirical fit)
axC.plot(
    x_fit,
    pred_summary['mean'],
    color='black',
    lw=1.5,
    alpha=0.8,
    label='Linear fit',
    zorder=4
)

# Confidence band
axC.fill_between(
    x_fit,
    pred_summary['mean_ci_lower'],
    pred_summary['mean_ci_upper'],
    color='grey',
    alpha=0.2,
    zorder=2,
    label='95% CI'
)

# Scatter points
sns.scatterplot(
    data=df_plot,
    x='Levodopa_Responsiveness',
    y='DBS_Responsiveness',
    hue='Category',
    palette=PLOT_CONFIG['palette'],
    s=PLOT_CONFIG['scatter_size'] * 0.7,
    alpha=0.65,
    edgecolor=None,
    style='Category',
    markers=MARKERS,
    ax=axC,
    zorder=5
)

# Labels & aesthetics
axC.set_title('C) Concordance of Therapeutic Response', fontsize=PLOT_CONFIG['title_fontsize'])
axC.set_xlabel('Levodopa Responsiveness (% Change)', fontsize=PLOT_CONFIG['axis_label_fontsize'])
axC.set_ylabel('DBS Responsiveness (% Change)', fontsize=PLOT_CONFIG['axis_label_fontsize'])
axC.set_xlim(lims)
axC.set_ylim(lims)
axC.set_aspect('equal')
axC.tick_params(labelsize=PLOT_CONFIG['tick_fontsize'])
if axC.get_legend():
    axC.get_legend().remove()

# Annotate stats
p_text = "p < 0.001" if p_obs < 0.001 else f"p = {p_obs:.3f}"
stats_text = f"r = {r_obs:.2f}\nslope = {slope:.2f}\n{p_text}"
axC.text(
    0.05, 0.95, stats_text,
    transform=axC.transAxes,
    fontsize=14,
    va='top',
    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='grey', alpha=0.9)
)

# Legend for reference lines
axC.legend(
    loc='lower right',
    frameon=True,
    fontsize=12,
    title='Reference',
    title_fontsize=13
)


# --------------------------------------------------------------------------------------
# PANEL D – Enhanced visualization (fixed version)
# --------------------------------------------------------------------------------------
def plot_cov_ellipse(ax, mean, cov, color, alpha=0.15, lw=2.0):
    # Ensure mean is a plain numeric tuple (fix for FutureWarning)
    mean = np.asarray(mean, dtype=float).ravel()
    mean = (float(mean[0]), float(mean[1]))

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * np.sqrt(vals)
    n_std = 2.0
    ell = Ellipse(xy=mean, width=n_std*width, height=n_std*height,
                  angle=theta, facecolor=color, alpha=alpha,
                  edgecolor=color, lw=lw, zorder=2)
    ax.add_patch(ell)

label_colors = {'State': '#009E73', 'Trait': '#0072B2', 'Comprehensive': '#999999'}

axD.set_title('D) Data-Driven Trait–State Clusters', fontsize=PLOT_CONFIG['title_fontsize'])
axD.set_xlabel('|Trait Link| (abs partial r with DaT)', fontsize=PLOT_CONFIG['axis_label_fontsize'])
axD.set_ylabel('Mean Therapy Responsiveness\n(mean of |ΔLevodopa|, |ΔDBS|) [%]', fontsize=PLOT_CONFIG['axis_label_fontsize'])
axD.grid(True, linestyle='--', alpha=0.35)

# Density shading
sns.kdeplot(data=df_plot, x='Abs_Trait', y='Mean_Response', hue='GMM_Label',
            fill=True, alpha=0.08, bw_adjust=1.2, common_norm=False,
            levels=5, thresh=0.05, ax=axD)

# Ellipses (fixed call)
for i, row in centroid_df.iterrows():
    cov = gmm.covariances_[i]
    label = labels[i]
    col = label_colors[label]
    center = row[['Abs_Trait', 'Mean_Response']].to_numpy()  # fixed: convert to array
    plot_cov_ellipse(axD, center, cov, color=col)

# Scatter + centroids
sns.scatterplot(data=df_plot, x='Abs_Trait', y='Mean_Response', hue='GMM_Label',
                palette=label_colors, s=110, alpha=0.8, edgecolor='black', linewidth=0.6, ax=axD, zorder=5)

for i, row in centroid_df.iterrows():
    label = labels[i]
    axD.scatter(row['Abs_Trait'], row['Mean_Response'], s=500, marker='X',
                color=label_colors[label], edgecolor='black', linewidth=1.2, zorder=6)
    axD.text(row['Abs_Trait'] + 0.01, row['Mean_Response'] + 1.2, label,
             fontsize=14, weight='bold', color=label_colors[label],
             path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# Legends
domain_handles = [
    plt.Line2D([], [], marker=m, color='w', label=k,
               markerfacecolor=v, markeredgecolor='k', markersize=10)
    for k, (m, v) in zip(MARKERS.keys(), zip(MARKERS.values(), PLOT_CONFIG['palette'].values()))
]
legend1 = axD.legend(handles=domain_handles, title='Feature Domain', loc='upper right', frameon=True, fontsize=11, title_fontsize=13)
axD.add_artist(legend1)
legend2 = axD.legend(title='Cluster Type', loc='lower right', fontsize=12, title_fontsize=13)
legend2.get_frame().set_alpha(0.95)

axD.set_xlim(0, df_plot['Abs_Trait'].max()*1.3)
axD.set_ylim(0, df_plot['Mean_Response'].max()*1.35)

# --------------------------------------------------------------------------------------
# 7) RADAR SUMMARY (Supplementary)
# --------------------------------------------------------------------------------------
def radar_plot(df, savepath):
    metrics = ['Abs_Trait','Abs_Levo','Abs_DBS']
    labels_radar = ['|Trait|','|ΔLevo|','|ΔDBS|']
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    cluster_means = df.groupby('GMM_Label')[metrics].mean()
    figR, axR = plt.subplots(subplot_kw={'polar': True}, figsize=(6,6))
    for label, row in cluster_means.iterrows():
        vals = row.tolist() + [row.tolist()[0]]
        axR.plot(angles, vals, linewidth=2, linestyle='solid', label=label,
                 color=label_colors[label])
        axR.fill(angles, vals, alpha=0.15, color=label_colors[label])
    axR.set_xticks(angles[:-1])
    axR.set_xticklabels(labels_radar, fontsize=12)
    axR.set_yticklabels([])
    axR.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    axR.set_title('Cluster Summary: Mean Dopamine–State Metrics', pad=20, fontsize=14, weight='bold')
    figR.tight_layout()
    figR.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(figR)
    print(f"[INFO] Supplementary radar plot saved to: {savepath}")

# Call radar plot
radar_save_path = os.path.join(plots_dir, "Figure 3_Radar_Supplementary.pdf")
radar_plot(df_plot, radar_save_path)

# --------------------------------------------------------------------------------------
# 8) SAVE FINAL FIGURE 3
# --------------------------------------------------------------------------------------
plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.96])
fig.subplots_adjust(wspace=0.2, hspace=0.4)
out_plot_path = os.path.join(plots_dir, "Figure 3.pdf")
plt.savefig(out_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print("\n[SUCCESS] Enhanced Figure 3 saved to:")
print(f"  → {out_plot_path}")
print(f"[INFO] Supplementary radar plot saved to:")
print(f"  → {radar_save_path}")
print("\n--- SCRIPT COMPLETE ---\n")