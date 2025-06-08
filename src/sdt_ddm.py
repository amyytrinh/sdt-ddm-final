"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# Mapping dictionaries for categorical variables
# These convert categorical labels to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# FACTOR LEVEL NAMES 
DIFFICULTY_NAMES = {0: 'Easy', 1: 'Hard'}
STIMULUS_NAMES = {0: 'Simple', 1: 'Complex'}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # PREPARE DATA FOR DELTA PLOT ANALYSIS
    elif prepare_for == 'delta plots':
        dp_data = []
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                if len(c_data) == 0: 
                    continue

                # OVERALL RTS
                overall_rt = c_data['rt']
                if len(overall_rt) > 0:
                    percentiles = {f'p{p}': np.percentile(overall_rt, p) for p in PERCENTILES}
                    dp_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'mode': 'overall',
                        **percentiles
                    })
                
                # ACCURATE RTS
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt'].values
                if len(accurate_rt) > 0:
                    percentiles = {f'p{p}': np.percentile(accurate_rt, p) for p in PERCENTILES}
                    dp_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'mode': 'accurate',
                        **percentiles
                    })
                
                # ERROR RTS
                error_rt = c_data[c_data['accuracy'] == 0]['rt'].values
                if len(error_rt) > 0:
                    percentiles = {f'p{p}': np.percentile(error_rt, p) for p in PERCENTILES}
                    dp_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'mode': 'error',
                        **percentiles
                    })

        data = pd.DataFrame(dp_data)
                
        if display:
            print("\nDelta plots data:")
            print(data.head())

    return data


def apply_hierarchical_sdt_model(data):
    """Apply a hierarchical Signal Detection Theory model using PyMC.
    
    This function implements a Bayesian hierarchical model for SDT analysis,
    allowing for both group-level and individual-level parameter estimation.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object
    """
    # GET UNIQUE PARTICIPANT IDS
    P = len(data['pnum'].unique())
    participant_ids = sorted(data['pnum'].unique())

    # MAP PARTICIPANTS TO INDICES
    pnum_to_index = {pid: i for i, pid in enumerate(participant_ids)}
    data['p_index'] = data['pnum'].map(pnum_to_index)

    # EXTRACT FACTOR LEVELS
    difficulty = data['condition'] // 2  
    stimulus_type = data['condition'] % 2

    # Define the hierarchical model
    with pm.Model() as sdt_model:
        # GROUP-LEVEL PRIORS AND D-PRIME EFFECTS
        intercept_d = pm.Normal('intercept_d', mu=1.0, sigma=1.0)
        difficulty_effect_d = pm.Normal('difficulty_effect_d', mu=0.0, sigma=0.5)
        stimulus_effect_d = pm.Normal('stimulus_effect_d', mu=0.0, sigma=0.5)
        interaction_effect_d = pm.Normal('interaction_effect_d', mu=0.0, sigma=0.5)
        
        # CRITERION/BIAS EFFECTS
        intercept_c = pm.Normal('intercept_c', mu=0.0, sigma=1.0)
        difficulty_effect_c = pm.Normal('difficulty_effect_c', mu=0.0, sigma=0.5)
        stimulus_effect_c = pm.Normal('stimulus_effect_c', mu=0.0, sigma=0.5)
        interaction_effect_c = pm.Normal('interaction_effect_c', mu=0.0, sigma=0.5)
        
        # CONSIDER INDIVIDUAL VARIATION
        sigma_d = pm.HalfNormal('sigma_d', sigma=0.5)
        sigma_c = pm.HalfNormal('sigma_c', sigma=0.5)
                
       # CALCULATE CONDITION-SPECIFIC GROUP MEANS
        mu_d = (intercept_d + 
                difficulty_effect_d * difficulty + 
                stimulus_effect_d * stimulus_type +
                interaction_effect_d * difficulty * stimulus_type)
        
        mu_c = (intercept_c + 
                difficulty_effect_c * difficulty + 
                stimulus_effect_c * stimulus_type +
                interaction_effect_c * difficulty * stimulus_type)

        # INDIVIDUAL PARAMETERS
        d_prime_raw = pm.Normal('d_prime_raw', mu=0, sigma=1, shape=P)
        criterion_raw = pm.Normal('criterion_raw', mu=0, sigma=1, shape=P)
        
        # PARAMETERIZATION
        d_prime = mu_d + sigma_d * d_prime_raw[data['p_idx']]
        criterion = mu_c + sigma_c * criterion_raw[data['p_idx']]
        
        # DEFINE SDT MODEL
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)

        pm.Binomial('hit_obs', n=data['nSignal'], p=hit_rate, observed=data['hits'])
        pm.Binomial('false_alarm_obs', n=data['nNoise'], p=false_alarm_rate, 
                   observed=data['false_alarms'])
    
    return sdt_model

def analyze_sdt_results(trace, model):
    """ANALYZE AND VISUALIZE SDT MODEL RESULTS"""
    
    # SUMMARY STATISTICS
    print("*** SUMMARY STATISTICS ***")
    summary = az.summary(trace, hdi_prob=0.94)
    print(summary)
    
    # EXTRACT KEY PARAMETERS
    difficulty_d = trace.posterior['difficulty_effect_d'].values.flatten()
    stimulus_d = trace.posterior['stimulus_effect_d'].values.flatten()
    interaction_d = trace.posterior['interaction_effect_d'].values.flatten()
    
    difficulty_c = trace.posterior['difficulty_effect_c'].values.flatten()
    stimulus_c = trace.posterior['stimulus_effect_c'].values.flatten()
    interaction_c = trace.posterior['interaction_effect_c'].values.flatten()
    
    # CREATE FIGURE
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # D-PRIME
    axes[0,0].hist(difficulty_d, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].axvline(0, color='red', linestyle='--')
    axes[0,0].set_title('Difficulty Effect on d-prime')
    axes[0,0].set_xlabel('Effect Size')
    
    axes[0,1].hist(stimulus_d, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0,1].axvline(0, color='red', linestyle='--')
    axes[0,1].set_title('Stimulus Type Effect on d-prime')
    axes[0,1].set_xlabel('Effect Size')
    
    axes[0,2].hist(interaction_d, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[0,2].axvline(0, color='red', linestyle='--')
    axes[0,2].set_title('Interaction Effect on d-prime')
    axes[0,2].set_xlabel('Effect Size')
    
    # CRITERION
    axes[1,0].hist(difficulty_c, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1,0].axvline(0, color='red', linestyle='--')
    axes[1,0].set_title('Difficulty Effect on Criterion')
    axes[1,0].set_xlabel('Effect Size')
    
    axes[1,1].hist(stimulus_c, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1,1].axvline(0, color='red', linestyle='--')
    axes[1,1].set_title('Stimulus Type Effect on Criterion')
    axes[1,1].set_xlabel('Effect Size')
    
    axes[1,2].hist(interaction_c, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1,2].axvline(0, color='red', linestyle='--')
    axes[1,2].set_title('Interaction Effect on Criterion')
    axes[1,2].set_xlabel('Effect Size')
    
    plt.tight_layout()
    plt.savefig('sdt_effects_posterior.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # EFFECT SIZE SUMMARY
    print("\n *** EFFECT SIZE SUMMARY *** ")
    effects = {
        'Difficulty -> d-prime': difficulty_d,
        'Stimulus Type -> d-prime': stimulus_d,
        'Interaction -> d-prime': interaction_d,
        'Difficulty -> Criterion': difficulty_c,
        'Stimulus Type -> Criterion': stimulus_c,
        'Interaction -> Criterion': interaction_c
    }
    
    results_df = []
    for name, samples in effects.items():
        mean_effect = np.mean(samples)
        hdi_low, hdi_high = np.percentile(samples, [2.5, 97.5])
        prob_positive = np.mean(samples > 0)
        prob_negative = np.mean(samples < 0)
        
        results_df.append({
            'Effect': name,
            'Mean': mean_effect,
            'HDI_Low': hdi_low,
            'HDI_High': hdi_high,
            'P(Effect > 0)': prob_positive,
            'P(Effect < 0)': prob_negative
        })
    
    results_df = pd.DataFrame(results_df)
    print(results_df.round(3))
    
    return results_df



def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences
    - Lower triangle shows RT differences split by correct/error responses
    
    Args:
        data: DataFrame with RT percentile data
        pnum: Participant number to plot
    """
    # GET UNIQUE PARTICIPANTS AND CONDITIONS
    participants = data['pnum'].unique()
    conditions = sorted(data['condition'].unique())
    
    
    # COMPARISON MATRIX
    comparisons = [
        (0, 1, 'Easy: Simple vs Complex'),  # Stimulus effect in Easy
        (2, 3, 'Hard: Simple vs Complex'),  # Stimulus effect in Hard  
        (0, 2, 'Simple: Easy vs Hard'),     # Difficulty effect in Simple
        (1, 3, 'Complex: Easy vs Hard')     # Difficulty effect in Complex
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (cond1, cond2, title) in enumerate(comparisons):
        ax = axes[idx]
        
        # OVERALL DELTAS ACROSS PARTICIPANTS
        overall_deltas = []
        accurate_deltas = []
        error_deltas = []
        
        for pnum in participants:
            p_data = data[data['pnum'] == pnum]
            
            # GET PERCENTILES FOR EACH CONDITION
            c1_overall = p_data[(p_data['condition'] == cond1) & (p_data['mode'] == 'overall')]
            c2_overall = p_data[(p_data['condition'] == cond2) & (p_data['mode'] == 'overall')]
            
            c1_accurate = p_data[(p_data['condition'] == cond1) & (p_data['mode'] == 'accurate')]
            c2_accurate = p_data[(p_data['condition'] == cond2) & (p_data['mode'] == 'accurate')]
            
            c1_error = p_data[(p_data['condition'] == cond1) & (p_data['mode'] == 'error')]
            c2_error = p_data[(p_data['condition'] == cond2) & (p_data['mode'] == 'error')]
            
            if len(c1_overall) > 0 and len(c2_overall) > 0:
                overall_delta = []
                for p in PERCENTILES:
                    delta = c2_overall[f'p{p}'].iloc[0] - c1_overall[f'p{p}'].iloc[0]
                    overall_delta.append(delta)
                overall_deltas.append(overall_delta)
            
            if len(c1_accurate) > 0 and len(c2_accurate) > 0:
                accurate_delta = []
                for p in PERCENTILES:
                    delta = c2_accurate[f'p{p}'].iloc[0] - c1_accurate[f'p{p}'].iloc[0]
                    accurate_delta.append(delta)
                accurate_deltas.append(accurate_delta)
            
            if len(c1_error) > 0 and len(c2_error) > 0:
                error_delta = []
                for p in PERCENTILES:
                    delta = c2_error[f'p{p}'].iloc[0] - c1_error[f'p{p}'].iloc[0]
                    error_delta.append(delta)
                error_deltas.append(error_delta)
        
        # CONVERT TO ARRAY AND CALCULATE MEAN
        if overall_deltas:
            overall_deltas = np.array(overall_deltas)
            overall_mean = np.mean(overall_deltas, axis=0)
            overall_sem = stats.sem(overall_deltas, axis=0)
            overall_ci = overall_sem * 1.96
            
            ax.errorbar(PERCENTILES, overall_mean, yerr=overall_ci, 
                       color='black', marker='o', linewidth=2, markersize=6,
                       label='Overall')
        
        if accurate_deltas:
            accurate_deltas = np.array(accurate_deltas)
            accurate_mean = np.mean(accurate_deltas, axis=0)
            accurate_sem = stats.sem(accurate_deltas, axis=0)
            accurate_ci = accurate_sem * 1.96
            
            ax.errorbar(PERCENTILES, accurate_mean, yerr=accurate_ci,
                       color='green', marker='s', linewidth=2, markersize=6,
                       label='Accurate')
        
        if error_deltas:
            error_deltas = np.array(error_deltas)
            error_mean = np.mean(error_deltas, axis=0)
            error_sem = stats.sem(error_deltas, axis=0)
            error_ci = error_sem * 1.96
            
            ax.errorbar(PERCENTILES, error_mean, yerr=error_ci,
                       color='red', marker='^', linewidth=2, markersize=6,
                       label='Error')
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile')
        ax.set_ylabel('RT Difference (s)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('delta_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def main_analysis(file_path):
    """ Run analysis """
    sdt_data = read_data(file_path, prepare_for='sdt', display=True)
    
    # Read data for delta plot analysis  
    delta_data = read_data(file_path, prepare_for='delta plots', display=True)
    
    print("\n*** RUNNING HIERARCHICAL SDT MODEL ***")
    
    # Make delta plot
    sdt_model = draw_delta_plots(sdt_data)
    
    # Sample from posterior
    with sdt_model:
        trace = pm.sample(2000, tune=1000, chains=4, 
                         target_accept=0.94, random_seed=42)
    
    # Analyze results
    results_df = analyze_sdt_results(trace, sdt_model)
    
    print("\n*** CREATING ENHANCED DELTA PLOTS ***")
    
    # Create delta plots
    enhanced_delta_plots(delta_data)
    
    print("\n*** SYNTHESIS AND INTERPRETATION ***")
    
    # Extract key findings for interpretation
    difficulty_d_mean = results_df[results_df['Effect'] == 'Difficulty → d-prime']['Mean'].iloc[0]
    stimulus_d_mean = results_df[results_df['Effect'] == 'Stimulus Type → d-prime']['Mean'].iloc[0]
    
    difficulty_c_mean = results_df[results_df['Effect'] == 'Difficulty → Criterion']['Mean'].iloc[0] 
    stimulus_c_mean = results_df[results_df['Effect'] == 'Stimulus Type → Criterion']['Mean'].iloc[0]
    
    print(f"""
    KEY FINDINGS:
    
    SDT ANALYSIS:
    - Difficulty effect on sensitivity (d'): {difficulty_d_mean:.3f}
    - Stimulus type effect on sensitivity (d'): {stimulus_d_mean:.3f}
    - Difficulty effect on criterion: {difficulty_c_mean:.3f}
    - Stimulus type effect on criterion: {stimulus_c_mean:.3f}
    
    INTERPRETATION:
    - {'Hard trials reduce' if difficulty_d_mean < 0 else 'Hard trials increase'} sensitivity
    - {'Complex stimuli reduce' if stimulus_d_mean < 0 else 'Complex stimuli increase'} sensitivity
    - Delta plots show RT distribution differences between conditions
    - Combined analysis reveals both perceptual and decisional effects
    """)
    
    return sdt_data, delta_data, trace, results_df

# Main execution
if __name__ == "__main__":
    file_path = "data.csv"
    sdt_data, delta_data, trace, results = main_analysis(file_path)