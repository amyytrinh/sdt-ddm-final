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

def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences
    - Lower triangle shows RT differences split by correct/error responses
    
    Args:
        data: DataFrame with RT percentile data
        pnum: Participant number to plot
    """
    # Filter data for specified participant
    data = data[data['pnum'] == pnum]
    
    # Get unique conditions and create subplot matrix
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    
    # Create figure with subplots matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    # Create output directory
    OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define marker style for plots
    marker_style = {
        'marker': 'o',
        'markersize': 10,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 3
    }
    
    # Create delta plots for each condition pair
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Add labels only to edge subplots
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
                
            # Skip diagonal and lower triangle for overall plots
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            
            # Create masks for condition and plotting mode
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            
            # Calculate RT differences for overall performance
            quantiles1 = [data[cmask1 & overall_mask][f'p{p}'] for p in PERCENTILES]
            quantiles2 = [data[cmask2 & overall_mask][f'p{p}'] for p in PERCENTILES]
            overall_delta = np.array(quantiles2) - np.array(quantiles1)
            
            # Calculate RT differences for error responses
            error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
            
            # Calculate RT differences for accurate responses
            accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
            
            # Plot overall RT differences
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            
            # Plot error and accurate RT differences
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')

            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add condition labels
            axes[i,j].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            axes[j,i].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            plt.tight_layout()
            
    # Save the figure
    plt.savefig(OUTPUT_DIR / f'delta_plots_{pnum}.png')

# Main execution
if __name__ == "__main__":
    file_to_print = Path(__file__).parent / 'README.md'
    with open(file_to_print, 'r') as file:
        print(file.read())