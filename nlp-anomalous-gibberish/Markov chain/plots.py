import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_probability_score_distribution(df, column='Probability Score', save_path='outputs/probability_score_distribution.png'):
    """
    Plot the distribution of 'Probability Score' values with:
    - 'CLC' category in red,
    - 'caps', 'punctuation', 'spacing', 'shuffle sentence' categories in green,
    - all other categories in blue.
    The x-axis is trimmed at 0.3 maximum.
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
        
        # Plot histograms for each category group
        if len(other_data) > 0:
            plt.hist(other_data, bins=bins, alpha=alpha, color='blue', label='Other Categories', range=(valid_data[column].min(), 0.3))
        if len(green_data) > 0:
            plt.hist(green_data, bins=bins, alpha=alpha, color='green', label='Caps/Punctuation/Spacing/Shuffle Sentence', range=(valid_data[column].min(), 0.3))
        if len(clc_data) > 0:
            plt.hist(clc_data, bins=bins, alpha=alpha, color='red', label='CLC', range=(valid_data[column].min(), 0.3))
        
        plt.legend()
    else:
        plt.hist(valid_data[column], bins=30, alpha=0.7, color='blue', range=(valid_data[column].min(), 0.3))
    
    plt.xlim(left=valid_data[column].min(), right=0.3)
    
    plt.title(f'Distribution of {column} (x-axis trimmed at 0.3)')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_probability_score_by_category_with_threshold(df, category_column='Category', value_column='Probability Score', 
                                                     save_path='outputs/probability_score_by_category_threshold.png'):
    """
    Plot boxplot of 'Probability Score' by category with:
    - Horizontal threshold line at minimum 'Probability Score' in 'CLC' category.
    - Y-axis trimmed at 0.3 max.
    - Categories on x-axis sorted alphabetically.
    """
    plt.figure(figsize=(12, 6))
    valid_data = df[df[value_column].notnull()]
    
    if len(valid_data) == 0 or category_column not in valid_data.columns:
        print(f"Cannot create plot: missing data or {category_column} column")
        return
    
    # Sort categories alphabetically
    categories_sorted = sorted(valid_data[category_column].dropna().unique())
    valid_data[category_column] = pd.Categorical(valid_data[category_column], categories=categories_sorted, ordered=True)
    
    if 'CLC' in categories_sorted:
        threshold = valid_data.loc[valid_data[category_column] == 'CLC', value_column].min()
    else:
        threshold = None
        print("Warning: 'CLC' category not found, no threshold will be drawn.")
    
    palette = {cat: 'red' if cat == 'CLC' else 'blue' for cat in categories_sorted}
    sns.boxplot(x=category_column, y=value_column, data=valid_data, palette=palette, order=categories_sorted)
    
    if threshold is not None:
        plt.axhline(y=threshold, color='green', linestyle='--', linewidth=2, 
                    label=f'CLC Min Probability Score Threshold: {threshold:.4f}')
        plt.legend()
    
    plt.ylim(top=0.3)  # Trim y-axis at 0.3
    
    plt.title(f'{value_column} by {category_column} with Threshold (Y-axis max 0.3)')
    plt.xlabel(category_column)
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    import os
    input_file = "outputs/markov_results.xlsx"
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
    else:
        df = pd.read_excel(input_file)
        plot_probability_score_distribution(df)
        plot_probability_score_by_category_with_threshold(df)


   