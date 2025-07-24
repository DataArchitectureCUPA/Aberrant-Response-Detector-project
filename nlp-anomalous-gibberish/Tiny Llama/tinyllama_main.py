import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import math
import openpyxl
import time
from datetime import timedelta

def load_model_and_tokenizer(model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device

def compute_log_probability(text: str, model, tokenizer, device) -> tuple:
    if not isinstance(text, str):
        return None, None, 0, 0
    try:
        start_time = time.time()
        inputs = tokenizer(str(text), return_tensors='pt')
        token_count = inputs['input_ids'].size(1)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss.item()
        sequence_length = inputs['input_ids'].size(1)
        log_prob = -loss * (sequence_length - 1)
        normalized_log_prob = log_prob / (sequence_length - 1)
        perplexity = math.exp(loss)
        processing_time = time.time() - start_time
        return normalized_log_prob, perplexity, processing_time, token_count
    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {str(e)}")
        return None, None, 0, 0

def process_excel_file(file_path, model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0') -> tuple:
    df = pd.read_excel(file_path)
    text_col = None
    for col in df.columns:
        if col.lower() == 'text':
            text_col = col
            break
    if text_col is None:
        raise ValueError("No 'Text' column found in the Excel file.")
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    df['Log Probability'] = None
    df['Perplexity'] = None
    df['Inference Time'] = None
    df['Token Count'] = None
    total_tokens = 0
    total_inference_time = 0
    processing_times = []
    for idx, text in tqdm(df[text_col].items(), desc="Processing texts"):
        log_prob, perplexity, infer_time, tokens = compute_log_probability(text, model, tokenizer, device)
        df.at[idx, 'Log Probability'] = log_prob
        df.at[idx, 'Perplexity'] = perplexity
        df.at[idx, 'Inference Time'] = infer_time
        df.at[idx, 'Token Count'] = tokens
        if tokens is not None:
            total_tokens += tokens
        if infer_time is not None:
            total_inference_time += infer_time
        processing_times.append(infer_time)
    timing_metrics = {
        'total_inference_time': total_inference_time,
        'total_tokens': total_tokens,
        'avg_inference_time_per_text': total_inference_time / len(df) if len(df) > 0 else 0,
        'avg_tokens_per_text': total_tokens / len(df) if len(df) > 0 else 0,
        'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
        'min_processing_time': min(processing_times) if processing_times else 0,
        'max_processing_time': max(processing_times) if processing_times else 0,
    }
    return df, timing_metrics

def save_results(df: pd.DataFrame, output_path: str, timing_metrics: dict):
    try:
        # Calculate summary statistics for 'Log Probability' and 'Perplexity'
        summary_stats = df[['Log Probability', 'Perplexity']].describe().transpose()
        summary_stats['stat'] = summary_stats.index
        summary_stats.reset_index(drop=True, inplace=True)
        # Create timing metrics dataframe
        timing_df = pd.DataFrame([timing_metrics])
        # Check if 'Category' column exists (case-insensitive)
        category_column = None
        for col_name in df.columns:
            if col_name.lower() == 'category':
                category_column = col_name
                break
        # If category column exists, calculate summary statistics per category
        if category_column:
            category_summary = df.groupby(category_column)[['Log Probability', 'Perplexity']].describe()
        else:
            category_summary = pd.DataFrame()
        # Save the dataframe to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            timing_df.to_excel(writer, sheet_name='Timing_Metrics', index=False)
            if not category_summary.empty:
                category_summary.to_excel(writer, sheet_name='Category_Summary', index=True)
        # Adjust column widths for all sheets
        wb = openpyxl.load_workbook(output_path)
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            for col in sheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                sheet.column_dimensions[column].width = adjusted_width
        wb.save(output_path)
        print(f"Results and summary statistics saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

# Example usage
if __name__ == "__main__":
    input_file = "../data/data_expanded.xlsx"
    output_file = "outputs/tinyllama_results.xlsx"
    model_to_use = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    start_time = time.time()
    results_df, timing_metrics = process_excel_file(input_file, model_name=model_to_use)
    total_runtime = time.time() - start_time
    timing_metrics['total_runtime'] = total_runtime
    print("\nOverall Statistics:")
    print(f"Number of texts processed: {len(results_df)}")
    print(f"Average log probability: {results_df['Log Probability'].mean():.4f}")
    print(f"Average perplexity: {results_df['Perplexity'].mean():.4f}")
    print(f"Min log probability: {results_df['Log Probability'].min():.4f}")
    print(f"Max log probability: {results_df['Log Probability'].max():.4f}")
    print(f"Min perplexity: {results_df['Perplexity'].min():.4f}")
    print(f"Max perplexity: {results_df['Perplexity'].max():.4f}")
    print("\nTiming Metrics:")
    print(f"Total runtime: {str(timedelta(seconds=timing_metrics['total_runtime']))}")
    print(f"Inference time: {str(timedelta(seconds=timing_metrics['total_inference_time']))}")
    print(f"Average processing time per text: {timing_metrics['avg_processing_time']:.4f} seconds")
    print(f"Min processing time: {timing_metrics['min_processing_time']:.4f} seconds")
    print(f"Max processing time: {timing_metrics['max_processing_time']:.4f} seconds")
    print(f"Total tokens processed: {timing_metrics['total_tokens']}")
    print(f"Average tokens per second: {timing_metrics['total_tokens']/timing_metrics['total_inference_time']:.2f}")
    save_results(results_df, output_file, timing_metrics)
