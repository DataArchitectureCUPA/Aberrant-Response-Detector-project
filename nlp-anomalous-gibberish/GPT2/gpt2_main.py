import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import time
import datetime

def load_model_and_tokenizer(model_name: str = 'gpt2'):
    """
    Loading model and tokenizer 
    
    Args:
        model_name (str): GPT-2 model 
        
    Returns:
        tuple: (model, tokenizer)
    """
    start_time = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")
    return model, tokenizer, load_time

def minimal_preprocess(text, convert_full_width=True, pad_token=' '):
    """
    Minimal preprocessing on input text.

    Parameters:
    - text: input string
    - convert_full_width: if True, converts full-width characters to ASCII equivalents
    - pad_token: token to pad short texts with instead of duplicating

    Returns:
    - processed text or None if input is not a string
    """
    if not isinstance(text, str):
        return None

    # Convert full-width characters to ASCII if requested
    if convert_full_width:
        new_chars = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:
                new_chars.append(chr(code - 0xFEE0))
            else:
                new_chars.append(char)
        text = ''.join(new_chars)

    # Pad very short texts (less than 2 characters after stripping) with pad_token
    if len(text.strip()) < 2:
        text = text + pad_token

    return text

def create_category_summary(df, column_name, category_column='Category'):
    """
    Create summary statistics by category if category column exists.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        column_name (str): Name of the column to analyze
        category_column (str): Name of the category column
        
    Returns:
        pd.DataFrame or None: DataFrame with category-wise summary or None if no category column
    """
    if category_column not in df.columns:
        return None
        
    # Calculate statistics by category
    result = df.groupby(category_column)[column_name].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Std Dev', 'std'),
        ('Min', 'min'),
        ('25%', lambda x: x.quantile(0.25)),
        ('Median', 'median'),
        ('75%', lambda x: x.quantile(0.75)),
        ('Max', 'max')
    ])
    
    return result

def process_excel_file(file_path: str, model, tokenizer, max_length: int = 1024):
    """
    Reads an Excel file with a 'Text' column, computes log probability and perplexity for each text using GPT-2,
    and appends results and timing info including a new 'Perplexity' column.

    Args:
        file_path (str): Path to the Excel file
        model: Pre-loaded GPT-2 model
        tokenizer: Pre-loaded GPT-2 tokenizer
        max_length (int): Maximum token length to process

    Returns:
        pd.DataFrame: DataFrame with original data plus new columns for log probability, perplexity, processing time, tokens count, tokens per second
    """
    # Load Excel file
    df = pd.read_excel(file_path)

    if 'Text' not in df.columns:
        raise ValueError("Input Excel file must contain a 'Text' column.")

    # Initialize result columns
    df['Log Probability'] = None
    df['Perplexity'] = None
    df['Processing Time'] = None
    df['Tokens Count'] = None
    df['Tokens per Second'] = None

    model.eval()

    for i, text in enumerate(df['Text']):
        if not isinstance(text, str):
            # Skip or mark invalid entries
            df.at[i, 'Log Probability'] = None
            df.at[i, 'Perplexity'] = None
            df.at[i, 'Processing Time'] = 0
            df.at[i, 'Tokens Count'] = 0
            df.at[i, 'Tokens per Second'] = 0
            continue

        # Minimal preprocessing (optional, define minimal_preprocess elsewhere)
        processed_text = minimal_preprocess(text)
        if processed_text is None:
            df.at[i, 'Log Probability'] = None
            df.at[i, 'Perplexity'] = None
            df.at[i, 'Processing Time'] = 0
            df.at[i, 'Tokens Count'] = 0
            df.at[i, 'Tokens per Second'] = 0
            continue

        try:
            start_time = time.time()

            # Tokenize once with truncation and special tokens
            encoded = tokenizer(
                processed_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']

            seq_len = input_ids.size(1)

            # Pad short sequences with EOS token if needed
            if seq_len < 2:
                eos_id = tokenizer.eos_token_id
                if eos_id is None:
                    raise ValueError("Tokenizer has no EOS token id for padding.")
                pad_len = 2 - seq_len
                pad_tensor = torch.full((1, pad_len), eos_id, dtype=torch.long)
                input_ids = torch.cat([input_ids, pad_tensor], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, pad_len), dtype=torch.long)], dim=1)
                seq_len = input_ids.size(1)

            # Model inference
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss.item()

            # Compute log probability and perplexity
            log_prob = -loss * (seq_len - 1)
            normalized_log_prob = log_prob / (seq_len - 1)
            perplexity = math.exp(loss)

            processing_time = time.time() - start_time
            tokens_per_second = seq_len / processing_time if processing_time > 0 else 0

            # Save results in DataFrame
            df.at[i, 'Log Probability'] = normalized_log_prob
            df.at[i, 'Perplexity'] = perplexity
            df.at[i, 'Processing Time'] = processing_time
            df.at[i, 'Tokens Count'] = seq_len
            df.at[i, 'Tokens per Second'] = tokens_per_second

        except Exception as e:
            print(f"Error processing text at index {i}: {e}")
            df.at[i, 'Log Probability'] = None
            df.at[i, 'Perplexity'] = None
            df.at[i, 'Processing Time'] = 0
            df.at[i, 'Tokens Count'] = 0
            df.at[i, 'Tokens per Second'] = 0

    return df

def generate_summary_statistics(df, column_name):
    """
    Generate summary statistics for a specified column in a DataFrame.
    """
    stats = {
        'Count': df[column_name].count(),
        'Mean': df[column_name].mean(),
        'Std Dev': df[column_name].std(),
        'Min': df[column_name].min(),
        '25%': df[column_name].quantile(0.25),
        'Median': df[column_name].median(),
        '75%': df[column_name].quantile(0.75),
        '90%': df[column_name].quantile(0.90),
        '95%': df[column_name].quantile(0.95),
        '99%': df[column_name].quantile(0.99),
        'Max': df[column_name].max()
    }
    return pd.DataFrame(stats, index=[0]).T.rename(columns={0: column_name})

def save_results_with_summary(df, output_path):
    """
    Save the results to an Excel file with a summary statistics tab.

    Args: 
        df (pd.DataFrame): Processed dataframe
        output_path (str): Path to save the output Excel file
    """
    try:
        # Create Excel writer object
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write main data to the first sheet
            df.to_excel(writer, sheet_name='Data', index=False)

            # Determine processed rows (where log probability is not null)
            processed_mask = df['Log Probability'].notnull()
            processed_df = df[processed_mask]

            if len(processed_df) > 0:
                # Create summary statistics DataFrames
                log_prob_stats = generate_summary_statistics(processed_df, 'Log Probability')
                perplexity_stats = generate_summary_statistics(processed_df, 'Perplexity')
                time_stats = generate_summary_statistics(processed_df, 'Processing Time')
                tokens_stats = generate_summary_statistics(processed_df, 'Tokens Count')
                tokens_per_sec_stats = generate_summary_statistics(processed_df, 'Tokens per Second')

                # Category statistics (if available)
                log_prob_by_category = create_category_summary(processed_df, 'Log Probability')
                perplexity_by_category = create_category_summary(processed_df, 'Perplexity')

                # Create summary tab
                summary_dfs = []

                summary_dfs.append(pd.DataFrame(['Overall Log Probability Statistics'], columns=['Summary Statistics']))
                summary_dfs.append(log_prob_stats)
                summary_dfs.append(pd.DataFrame([' '], columns=['Summary Statistics']))

                summary_dfs.append(pd.DataFrame(['Overall Perplexity Statistics'], columns=['Summary Statistics']))
                summary_dfs.append(perplexity_stats)
                summary_dfs.append(pd.DataFrame([' '], columns=['Summary Statistics']))

                summary_dfs.append(pd.DataFrame(['Processing Time Statistics (seconds)'], columns=['Summary Statistics']))
                summary_dfs.append(time_stats)
                summary_dfs.append(pd.DataFrame([' '], columns=['Summary Statistics']))

                summary_dfs.append(pd.DataFrame(['Token Count Statistics'], columns=['Summary Statistics']))
                summary_dfs.append(tokens_stats)
                summary_dfs.append(pd.DataFrame([' '], columns=['Summary Statistics']))

                summary_dfs.append(pd.DataFrame(['Tokens Per Second Statistics'], columns=['Summary Statistics']))
                summary_dfs.append(tokens_per_sec_stats)
                summary_dfs.append(pd.DataFrame([' '], columns=['Summary Statistics']))

                # Add category statistics if available
                if log_prob_by_category is not None:
                    summary_dfs.append(pd.DataFrame(['Log Probability by Category'], columns=['Summary Statistics']))
                    summary_dfs.append(log_prob_by_category.reset_index())
                    summary_dfs.append(pd.DataFrame([' '], columns=['Summary Statistics']))

                if perplexity_by_category is not None:
                    summary_dfs.append(pd.DataFrame(['Perplexity by Category'], columns=['Summary Statistics']))
                    summary_dfs.append(perplexity_by_category.reset_index())
                    summary_dfs.append(pd.DataFrame([' '], columns=['Summary Statistics']))

                # Combine all into one DataFrame
                summary_df = pd.concat(summary_dfs, ignore_index=True)
                summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)

                # Processing status summary
                processed_count = processed_mask.sum()
                failed_count = len(df) - processed_count
                total_count = len(df)
                status_summary = pd.DataFrame({
                    'Status': ['Processed', 'Failed', 'Total'],
                    'Count': [processed_count, failed_count, total_count],
                    'Percentage': [
                        f"{processed_count / total_count * 100:.1f}%",
                        f"{failed_count / total_count * 100:.1f}%",
                        "100.0%"
                    ]
                })
                status_summary.to_excel(writer, sheet_name='Processing Status', index=False)

                # Timing summary sheet if present
                if hasattr(df, 'timing_summary'):
                    df.timing_summary.to_excel(writer, sheet_name='Timing Summary', index=False)

        print(f"Results and summary statistics saved to {output_path}")

    except Exception as e:
        print(f"Error saving results: {str(e)}")


###PLOTS###

# Plotting functions with PNG saving and CLC highlighting
def plot_log_probability_distribution(df, column='Log Probability', save_path='log_probability_distribution.png'):
    """
    Plot the distribution of log probabilities with:
    - 'CLC' category in red,
    - 'caps', 'punctuation', 'spacing' categories in green,
    - all other categories in blue.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        column (str): Column name for log probability
        save_path (str): Path to save the PNG file
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    valid_data = df[df[column].notnull()]
    
    if 'Category' in df.columns:
        # Define category groups
        green_cats = {'caps', 'punctuation', 'spacing'}
        
        # Extract data per group
        clc_data = valid_data[valid_data['Category'] == 'CLC'][column]
        green_data = valid_data[valid_data['Category'].isin(green_cats)][column]
        other_data = valid_data[
            (~valid_data['Category'].isin(green_cats)) & (valid_data['Category'] != 'CLC')
        ][column]
        
        # Plot histograms for each group if data exists
        bins = 30
        alpha = 0.7
        
        if len(other_data) > 0:
            plt.hist(other_data, bins=bins, alpha=alpha, color='blue', label='Other Categories')
        if len(green_data) > 0:
            plt.hist(green_data, bins=bins, alpha=alpha, color='green', label='Caps/Punctuation/Spacing')
        if len(clc_data) > 0:
            plt.hist(clc_data, bins=bins, alpha=alpha, color='red', label='CLC')
        
        plt.legend()
    else:
        # No category column, plot all data in default color
        plt.hist(valid_data[column], bins=30, alpha=0.7)
    
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_perplexity_distribution(df, column='Perplexity', save_path='perplexity_distribution.png', log_scale=True):
    """
    Plot the distribution of perplexity with CLC category highlighted in red.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        column (str): Column name for perplexity
        save_path (str): Path to save the PNG file
        log_scale (bool): Whether to use log scale for x-axis
    """
    plt.figure(figsize=(12, 6))
    
    # Filter out null values
    valid_data = df[df[column].notnull()]
    
    if 'Category' in df.columns:
        # Create separate data for CLC and others
        clc_data = valid_data[valid_data['Category'] == 'CLC'][column]
        other_data = valid_data[valid_data['Category'] != 'CLC'][column]
        
        # Plot histograms
        if len(other_data) > 0:
            plt.hist(other_data, bins=30, alpha=0.7, color='blue', label='Other Categories')
        
        if len(clc_data) > 0:
            plt.hist(clc_data, bins=30, alpha=0.7, color='red', label='CLC')
            
        plt.legend()
    else:
        # If no category column, plot as normal
        plt.hist(valid_data[column], bins=30, alpha=0.7)
    
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.xscale('log')
    
    # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_tokens_per_second(df, column='Tokens per Second', save_path='tokens_per_second.png'):
    """
    Plot the distribution of tokens per second with CLC category highlighted in red.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        column (str): Column name for tokens per second
        save_path (str): Path to save the PNG file
    """
    plt.figure(figsize=(12, 6))
    
    # Filter out null values
    valid_data = df[df[column].notnull()]
    
    if 'Category' in df.columns:
        # Create separate data for CLC and others
        clc_data = valid_data[valid_data['Category'] == 'CLC'][column]
        other_data = valid_data[valid_data['Category'] != 'CLC'][column]
        
        # Plot histograms
        if len(other_data) > 0:
            plt.hist(other_data, bins=30, alpha=0.7, color='blue', label='Other Categories')
        
        if len(clc_data) > 0:
            plt.hist(clc_data, bins=30, alpha=0.7, color='red', label='CLC')
            
        plt.legend()
    else:
        # If no category column, plot as normal
        plt.hist(valid_data[column], bins=30, alpha=0.7)
    
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_log_probability_by_category(df, category_column='Category', value_column='Log Probability', 
                                    save_path='log_probability_by_category.png'):
    """
    Plot boxplot of log probability by category with CLC category in red,
    caps/punctuation/spacing categories in green, others in blue.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 6))
    
    valid_data = df[df[value_column].notnull()]
    
    if len(valid_data) == 0 or category_column not in valid_data.columns:
        print(f"Cannot create plot: missing data or {category_column} column")
        return
    
    categories = valid_data[category_column].unique()
    
    # Define sets for the special categories
    green_categories = {'caps', 'punctuation', 'spacing'}
    
    # Create palette with three colors
    palette = {}
    for cat in categories:
        if cat == 'CLC':
            palette[cat] = 'red'
        elif cat in green_categories:
            palette[cat] = 'green'
        else:
            palette[cat] = 'blue'
    
    sns.boxplot(x=category_column, y=value_column, data=valid_data, palette=palette)
    
    plt.title(f'{value_column} by {category_column}')
    plt.xlabel(category_column)
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_perplexity_by_category(df, category_column='Category', value_column='Perplexity', 
                               save_path='perplexity_by_category.png', log_scale=True):
    """
    Plot boxplot of perplexity by category with CLC category highlighted in red.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        category_column (str): Column name for categories
        value_column (str): Column name for perplexity values
        save_path (str): Path to save the PNG file
        log_scale (bool): Whether to use log scale for y-axis
    """
    plt.figure(figsize=(12, 6))
    
    # Filter out null values
    valid_data = df[df[value_column].notnull()]
    
    if len(valid_data) == 0 or category_column not in valid_data.columns:
        print(f"Cannot create plot: missing data or {category_column} column")
        return
    
    # Prepare custom palette with CLC in red
    palette = {cat: 'red' if cat == 'CLC' else 'blue' for cat in valid_data[category_column].unique()}
    
    # Create boxplot
    sns.boxplot(x=category_column, y=value_column, data=valid_data, palette=palette)
    
    plt.title(f'{value_column} by {category_column}')
    plt.xlabel(category_column)
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_log_probability_by_category_with_threshold(df, category_column='Category', value_column='Log Probability', 
                                                    save_path='log_probability_by_category_threshold.png', threshold=None):
    """
    Plot boxplot of log probability by category with a horizontal threshold line calculated as
    the minimum log probability value in the 'CLC' category.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        category_column (str): Column name for categories
        value_column (str): Column name for log probability values
        save_path (str): Path to save the PNG file
    """
    plt.figure(figsize=(12, 6))
    
    valid_data = df[df[value_column].notnull()]
    if len(valid_data) == 0 or category_column not in valid_data.columns:
        print(f"Cannot create plot: missing data or {category_column} column")
        return
    
    # Calculate threshold as min log probability in 'CLC' category
    if 'CLC' in valid_data[category_column].unique():
        threshold = valid_data.loc[valid_data[category_column] == 'CLC', value_column].min()
    else:
        threshold = None
        print("Warning: 'CLC' category not found, no threshold will be drawn.")
    
    palette = {cat: 'red' if cat == 'CLC' else 'blue' for cat in valid_data[category_column].unique()}
    ax = sns.boxplot(x=category_column, y=value_column, data=valid_data, palette=palette)
    
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
                                              save_path='perplexity_by_category_threshold.png', log_scale=True, threshold=None):
    """
    Plot boxplot of perplexity by category with a horizontal threshold line calculated as
    the maximum perplexity value in the 'CLC' category.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        category_column (str): Column name for categories
        value_column (str): Column name for perplexity values
        save_path (str): Path to save the PNG file
        log_scale (bool): Whether to use log scale for y-axis
    """
    plt.figure(figsize=(12, 6))
    
    valid_data = df[df[value_column].notnull()]
    if len(valid_data) == 0 or category_column not in valid_data.columns:
        print(f"Cannot create plot: missing data or {category_column} column")
        return
    
    # Calculate threshold as max perplexity in 'CLC' category
    if 'CLC' in valid_data[category_column].unique():
        threshold = valid_data.loc[valid_data[category_column] == 'CLC', value_column].max()
    else:
        threshold = None
        print("Warning: 'CLC' category not found, no threshold will be drawn.")
    
    palette = {cat: 'red' if cat == 'CLC' else 'blue' for cat in valid_data[category_column].unique()}
    ax = sns.boxplot(x=category_column, y=value_column, data=valid_data, palette=palette)
    
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


if __name__ == "__main__":
    # 1. Load the model and tokenizer
    model, tokenizer, load_time = load_model_and_tokenizer("gpt2")
    
    # 2. Process your Excel file 
    processed_df = process_excel_file("../data/data_expanded.xlsx", model, tokenizer)
    
    # 3. Save the results 
    save_results_with_summary(processed_df, "outputs/gpt2_results_with_summary.xlsx")

    # 4. Generate and save plots as PNG files
    print("\nGenerating plots and saving as PNG files...")
    
    # Reload the processed DataFrame 
    processed_df = pd.read_excel("outputs/gpt2_results_with_summary.xlsx")
    
    # Plot log probability distribution
    plot_log_probability_distribution(processed_df, column='Log Probability', 
                                     save_path='outputs/log_probability_distribution.png')
    
    # Plot perplexity distribution
    plot_perplexity_distribution(processed_df, column='Perplexity', 
                                save_path='outputs/perplexity_distribution.png')
    
    # Plot tokens per second distribution
    plot_tokens_per_second(processed_df, column='Tokens per Second', 
                          save_path='outputs/tokens_per_second.png')
    
    #  'Category' column, plot by category:
    if 'Category' in processed_df.columns:
        # Extract CLC minimum values as thresholds
        clc_data = processed_df[processed_df['Category'] == 'CLC']
        
        if len(clc_data) > 0:
            # Get min log probability and perplexity for CLC
            log_prob_threshold = clc_data['Log Probability'].min()
            perplexity_threshold = clc_data['Perplexity'].min()
            
            print(f"CLC Log Probability threshold (min value): {log_prob_threshold:.4f}")
            print(f"CLC Perplexity threshold (min value): {perplexity_threshold:.4f}")
        else:
            log_prob_threshold = None
            perplexity_threshold = None
            print("No CLC data found to establish thresholds")
        
        # Regular category plots
        plot_log_probability_by_category(processed_df, 
                                        category_column='Category', 
                                        value_column='Log Probability',
                                        save_path='outputs/log_probability_by_category.png')
        
        plot_perplexity_by_category(processed_df, 
                                   category_column='Category', 
                                   value_column='Perplexity',
                                   save_path='outputs/perplexity_by_category.png')
        
        # Additional plots with threshold lines
        plot_log_probability_by_category_with_threshold(processed_df,
                                                      category_column='Category',
                                                      value_column='Log Probability',
                                                      threshold=log_prob_threshold,
                                                      save_path='outputs/log_probability_by_category_threshold.png')
        
        plot_perplexity_by_category_with_threshold(processed_df,
                                                 category_column='Category',
                                                 value_column='Perplexity',
                                                 threshold=perplexity_threshold,
                                                 save_path='outputs/perplexity_by_category_threshold.png')
    
    print("All plots generated and saved successfully!")