import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

def plot_log_probability_distribution(df, column='Log Probability', save_path='outputs/log_probability_distribution.png'):
    """
    Plot the distribution of log probabilities with:
    - 'CLC' category in red,
    - 'caps', 'punctuation', 'spacing', 'shuffle sentence' categories in green,
    - all other categories in blue.
    The x-axis starts at zero and goes -1, -2, -3, etc. to the right (reversed).
    """
    plt.figure(figsize=(12, 6))
    valid_data = df[df[column].notnull()]
    
    if 'Category' in df.columns:
        green_cats = {'caps', 'punctuation', 'spacing', 'shuffle sentence'}
        clc_data = valid_data[valid_data['Category'] == 'CLC'][column]
        green_data = valid_data[valid_data['Category'].isin(green_cats)][column]
        other_data = valid_data[
            (~valid_data['Category'].isin(green_cats)) & (valid_data['Category'] != 'CLC')
        ][column]
        
        bins = 30
        alpha = 0.7
        
        if len(other_data) > 0:
            plt.hist(other_data, bins=bins, alpha=alpha, color='blue', label='Other Categories')
        if len(green_data) > 0:
            plt.hist(green_data, bins=bins, alpha=alpha, color='green', label='Caps/Punctuation/Spacing/Shuffle Sentence')
        if len(clc_data) > 0:
            plt.hist(clc_data, bins=bins, alpha=alpha, color='red', label='CLC')
        
        plt.legend()
    else:
        plt.hist(valid_data[column], bins=30, alpha=0.7)
    
    # Get current x-axis limits after plotting
    x_min, x_max = plt.xlim()
    
    # Set new x-axis limits: 0 on left, minimum on right (reversed)
    plt.xlim(0, x_min)
    
    # Create ticks from 0 to x_min (negative values)
    num_ticks = 7
    ticks = np.linspace(0, x_min, num=num_ticks)
    plt.xticks(ticks, labels=[f"{int(tick)}" for tick in ticks])
    
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")






def plot_log_probability_by_category_with_threshold(df, category_column='Category', value_column='Log Probability', 
                                                    save_path='outputs/log_probability_by_category_threshold.png', threshold=None):
    """
    Plot boxplot of log probability by category with a horizontal threshold line calculated as
    the minimum log probability value in the 'CLC' category.
    """
    plt.figure(figsize=(12, 6))
    valid_data = df[df[value_column].notnull()]
    
    if len(valid_data) == 0 or category_column not in valid_data.columns:
        print(f"Cannot create plot: missing data or {category_column} column")
        return
    
    if 'CLC' in valid_data[category_column].unique():
        threshold = valid_data.loc[valid_data[category_column] == 'CLC', value_column].min()
    else:
        threshold = None
        print("Warning: 'CLC' category not found, no threshold will be drawn.")
    
    palette = {cat: 'red' if cat == 'CLC' else 'blue' for cat in valid_data[category_column].unique()}
    sns.boxplot(x=category_column, y=value_column, data=valid_data, palette=palette)
    
    if threshold is not None:
        plt.axhline(y=threshold, color='green', linestyle='--', linewidth=2, 
                    label=f'CLC Min Log Probability Threshold: {threshold:.4f}')
        plt.legend()
    
    plt.title(f'{value_column} by {category_column} with Threshold')
    plt.xlabel(category_column)
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_perplexity_by_category_with_threshold(df, category_column='Category', value_column='Perplexity', 
                                              save_path='outputs/perplexity_by_category_threshold.png', log_scale=True, threshold=None):
    """
    Plot boxplot of perplexity by category with a horizontal threshold line calculated as
    the maximum perplexity value in the 'CLC' category.
    """
    plt.figure(figsize=(12, 6))
    valid_data = df[df[value_column].notnull()]
    
    if len(valid_data) == 0 or category_column not in valid_data.columns:
        print(f"Cannot create plot: missing data or {category_column} column")
        return
    
    if 'CLC' in valid_data[category_column].unique():
        threshold = valid_data.loc[valid_data[category_column] == 'CLC', value_column].max()
    else:
        threshold = None
        print("Warning: 'CLC' category not found, no threshold will be drawn.")
    
    palette = {cat: 'red' if cat == 'CLC' else 'blue' for cat in valid_data[category_column].unique()}
    sns.boxplot(x=category_column, y=value_column, data=valid_data, palette=palette)
    
    if threshold is not None:
        plt.axhline(y=threshold, color='green', linestyle='--', linewidth=2, 
                    label=f'CLC Max Perplexity Threshold: {threshold:.4f}')
        plt.legend()
    
    plt.title(f'{value_column} by {category_column} with Threshold')
    plt.xlabel(category_column)
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

# with tinyllama_results.xlsx
if __name__ == "__main__":
    import os
    input_file = "outputs/tinyllama_results.xlsx"
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
    else:
        df = pd.read_excel(input_file)
        plot_log_probability_distribution(df)     
          
        plot_log_probability_by_category_with_threshold(df)
        plot_perplexity_by_category_with_threshold(df)
