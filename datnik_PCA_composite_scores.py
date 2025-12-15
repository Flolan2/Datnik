# --- IV_datnik_PCA_composite_scores.py ---
# Derives subject-level Trait and State composite scores via PCA.
# TARGET: DaTscan Cohort (Dataset 1) in OFF state.
# 100% REAL DATA ONLY.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

print("\n" + "="*88)
print("--- RUNNING: PCA Composite Score Generation (Target: DaTscan Cohort) ---")
print("="*88 + "\n")

# --------------------------------------------------------------------------------------
# 1) PATH SETUP
# --------------------------------------------------------------------------------------
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

if script_dir.endswith("Online") or os.path.basename(script_dir) == "Online":
    project_root_dir = os.path.dirname(script_dir)
else:
    project_root_dir = script_dir if os.path.isdir(os.path.join(script_dir, "Output")) else os.path.dirname(script_dir)

# Inputs for Defining Clusters (The "Rules")
stimvision_input_dir = os.path.join(project_root_dir, "Input", "Combined_Analysis")
datscan_stats_dir = os.path.join(project_root_dir, "Output", "Data") 

# Input for The Target Cohort (The "Subjects")
processed_data_dir = os.path.join(project_root_dir, "Output", "Data_Processed")

# Output
output_dir = os.path.join(project_root_dir, "Output", "Data")
plots_dir = os.path.join(project_root_dir, "Output", "Plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

print(f"[PATHS] Rules Source:   {stimvision_input_dir}")
print(f"[PATHS] Target Cohort:  {processed_data_dir}")
print(f"[PATHS] Output CSV:     {output_dir}")

# --------------------------------------------------------------------------------------
# 2) CONFIGURATION
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

# Features where "Lower Value" = "Worse/Parkinsonian"
# We flip these so PCA PC1 always points towards "Impairment"
LOWER_IS_WORSE = [
    'MeanAmplitude', 'MeanSpeed', 'MeanRMSVelocity', 'MeanOpeningSpeed', 'MeanClosingSpeed',
    'Rate', 'Frequency', 'AmplitudeDecay', 'VelocityDecay', 'RateDecay'
]

# --------------------------------------------------------------------------------------
# 3) STEP A: DEFINE TRAIT/STATE FEATURES (Using Logic from Fig 3)
# --------------------------------------------------------------------------------------
print("\n[STEP A] Defining Trait vs. State features (HM-Only Logic)...")

try:
    # 3.1 Load Bivariate (Trait Link)
    trait_file = os.path.join(datscan_stats_dir, "all_raw_bivariate_results.csv")
    df_trait_raw = pd.read_csv(trait_file, sep=';', decimal='.')
    
    # Parse HM features only
    def parse_kinematic(row):
        s = str(row['Kinematic_Variable'])
        if '_' in s: return s.split('_', 1)
        return [None, s]
    
    df_trait_raw[['Prefix', 'Base']] = df_trait_raw.apply(lambda x: pd.Series(parse_kinematic(x)), axis=1)
    df_trait_hm = df_trait_raw[df_trait_raw['Prefix'] == 'hm'].copy()
    df_trait_hm['Feature'] = df_trait_hm['Base'].map(FEATURE_NAME_MAP)
    df_trait = df_trait_hm.groupby('Feature')['r'].mean().rename('Trait_Link').to_frame()

    # 3.2 Load Responsiveness (State Link)
    raw_effect_dbs = pd.read_csv(os.path.join(stimvision_input_dir, "all_patients_raw_dbs_effect_Bilateral_Average.csv"), index_col="PatientID")
    baseline_dbs   = pd.read_csv(os.path.join(stimvision_input_dir, "all_patients_baseline_values_Bilateral_Average.csv"), index_col="PatientID")
    raw_effect_levo = pd.read_csv(os.path.join(stimvision_input_dir, "medication_raw_effect_hand_opening.csv"), index_col="Patient_ID")
    baseline_levo   = pd.read_csv(os.path.join(stimvision_input_dir, "medication_baseline_hand_opening.csv"), index_col="Patient_ID")

    for df in [raw_effect_dbs, baseline_dbs, raw_effect_levo, baseline_levo]:
        df.columns = df.columns.str.lower().map(FEATURE_NAME_MAP)

    common_features = df_trait.index.intersection(raw_effect_dbs.columns).intersection(raw_effect_levo.columns)

    with np.errstate(divide='ignore', invalid='ignore'):
        dbs_norm  = (raw_effect_dbs[common_features] / baseline_dbs[common_features].abs()) * 100
        levo_norm = (raw_effect_levo[common_features] / baseline_levo[common_features].abs()) * 100
    
    # 3.3 GMM Clustering
    df_clus = df_trait.loc[common_features].copy()
    df_clus['Mean_Response'] = (dbs_norm.median().abs() + levo_norm.median().abs()) / 2
    df_clus['Abs_Trait'] = df_clus['Trait_Link'].abs()

    scaler_gmm = StandardScaler()
    Z = scaler_gmm.fit_transform(df_clus[['Abs_Trait', 'Mean_Response']].values)
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    clusters = gmm.fit_predict(Z)
    
    # Label Clusters
    centroids = pd.DataFrame(scaler_gmm.inverse_transform(gmm.means_), columns=['Abs_Trait', 'Mean_Response'])
    trait_idx = centroids['Abs_Trait'].idxmax()
    state_idx = centroids['Mean_Response'].idxmax()
    
    trait_features_list = df_clus.index[clusters == trait_idx].tolist()
    state_features_list = df_clus.index[clusters == state_idx].tolist()
    
    print(f"   -> Identified {len(trait_features_list)} Trait Features: {trait_features_list}")
    print(f"   -> Identified {len(state_features_list)} State Features: {state_features_list}")

except Exception as e:
    print(f"[CRITICAL ERROR] Could not define feature sets: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------------------
# 4) STEP B: LOAD TARGET COHORT
# --------------------------------------------------------------------------------------
print("\n[STEP B] Loading Target Cohort (DaTscan Patients)...")
target_file = os.path.join(processed_data_dir, "final_merged_data.csv")

if not os.path.exists(target_file):
    print(f"[ERROR] Could not find {target_file}.")
    sys.exit(1)

df_target = pd.read_csv(target_file, sep=';', decimal='.')

# Filter for OFF state
if 'Medication Condition' in df_target.columns:
    df_target['Medication Condition'] = df_target['Medication Condition'].astype(str).str.lower().str.strip()
    df_target_off = df_target[df_target['Medication Condition'] == 'off'].copy()
    print(f"   -> Filtered for OFF-State. Rows: {len(df_target)} -> {len(df_target_off)}")
else:
    df_target_off = df_target.copy()

# --------------------------------------------------------------------------------------
# 5) STEP C: PREPARE PCA INPUTS
# --------------------------------------------------------------------------------------
print("\n[STEP C] Preparing PCA Inputs...")

def prepare_pca_matrix(feature_list, df_source):
    mapped_data = pd.DataFrame(index=df_source.index)
    cols_found = []
    
    for feat in feature_list:
        # We need the HM raw column name
        raw_base = [k for k, v in FEATURE_NAME_MAP.items() if v == feat][0]
        col_name = f"hm_{raw_base}" 
        
        if col_name in df_source.columns:
            # FLIP: Lower Value (Worse) * -1 -> Higher Value (Worse)
            if feat in LOWER_IS_WORSE:
                mapped_data[feat] = df_source[col_name] * -1
            else:
                mapped_data[feat] = df_source[col_name]
            cols_found.append(feat)
        else:
            print(f"     [WARN] Feature {feat} (col: {col_name}) missing in target data.")
            
    return mapped_data[cols_found]

df_trait_input = prepare_pca_matrix(trait_features_list, df_target_off)
df_state_input = prepare_pca_matrix(state_features_list, df_target_off)

# --------------------------------------------------------------------------------------
# 6) STEP D: RUN PCA
# --------------------------------------------------------------------------------------
print("\n[STEP D] Running PCA...")

def run_pca(df_in, label):
    if df_in.empty: return None, None
    
    # Impute & Scale
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(df_in)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    
    # PCA
    pca = PCA(n_components=1)
    scores = pca.fit_transform(X_scaled).flatten()
    
    # Force Positive Loading = Worse
    loadings = pca.components_[0]
    if np.mean(loadings) < 0:
        scores = scores * -1
        loadings = loadings * -1
        
    expl_var = pca.explained_variance_ratio_[0] * 100
    print(f"     -> {label} PC1: {expl_var:.1f}% variance explained.")
    return scores, loadings

trait_scores, trait_loadings = run_pca(df_trait_input, "Trait")
state_scores, state_loadings = run_pca(df_state_input, "State")

# --------------------------------------------------------------------------------------
# 7) STEP E: SAVE OUTPUTS
# --------------------------------------------------------------------------------------
print("\n[STEP E] Saving Results...")

if trait_scores is not None and state_scores is not None:
    # Create final DF including HAND info
    output_df = pd.DataFrame({
        'Patient ID': df_target_off['Patient ID'],
        'Hand': df_target_off['Hand_Performed'],  # <--- ADDED THIS LINE
        'Trait_Score': trait_scores,
        'State_Score': state_scores
    })
    
    if 'Date of Visit' in df_target_off.columns:
        output_df['Date of Visit'] = df_target_off['Date of Visit']

    out_file = os.path.join(output_dir, "IV_PCA_Composite_Scores_DaTscanCohort.csv")
    output_df.to_csv(out_file, index=False, sep=';')
    print(f"   -> SAVED: {out_file}")
    
    # Plot QC
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pd.Series(trait_loadings, index=df_trait_input.columns).sort_values().plot(kind='barh', ax=axes[0], color='gold')
    axes[0].set_title("Trait Dimension (Degeneration)")
    pd.Series(state_loadings, index=df_state_input.columns).sort_values().plot(kind='barh', ax=axes[1], color='#d62728')
    axes[1].set_title("State Dimension (Therapy)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "Figure_4_PCA_Composition.pdf"))

else:
    print("[ERROR] PCA failed.")

print("\n--- SCRIPT COMPLETE ---")