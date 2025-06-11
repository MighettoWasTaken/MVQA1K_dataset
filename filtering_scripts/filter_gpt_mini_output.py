import pandas as pd

filepath_messy = "partial_results.csv"
def cleanup(filepath):
    
    print("running")
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset=['figure_path', 'question'])  # or use set(df['Figure_path']) logic if you're going raw

    df.to_csv("test_" + filepath, index=False)
    return df 

df = cleanup(filepath_messy)


# Filtering logic
correct_margin = 0.02
gibberish_threshold = 0.3
gunky = 0.75
correct_thresh = .75

def is_garbage_output(output):
    # Heuristics to catch garbage:
    if isinstance(output, str):
        lowered = output.lower()
        if ('a:' in lowered and 'b:' in lowered) or ('c:' in lowered and 'd:' in lowered):
            return True
        
        if output.count(':') >= 2:
            return True

def is_truly_difficult(row):
    sim_to_answer = row['similarity_to_answer']
    max_sim = row['max_similarity_to_choices']
    output = row['model_output']
    if is_garbage_output(output):
        return False

    # Case 1: Clearly correct (or nearly correct)
    if sim_to_answer > correct_thresh:
        return False 

    if sim_to_answer >= max_sim - correct_margin:
        return False

    # Case 2: Gibberish (too low across the board)
    if sim_to_answer < gibberish_threshold and max_sim < gibberish_threshold:
        return False

    # Case 3: Model wasn't close to any option
    if max_sim <= gunky:
        return False

    return True  # Only keep if it truly struggled

# Apply the filter
filtered_df = df[df.apply(is_truly_difficult, axis=1)]

# Save or inspect
filtered_df.to_csv("filtered_difficult_cases.csv", index=False)

# Load full training data
train_df = pd.read_csv("medv_int/Dataset/train.csv")

# Ensure column names match
train_df.rename(columns={"Figure_path": "figure_path", "Question": "question"}, inplace=True)

# Merge on both figure_path and question
merged_df = filtered_df.merge(train_df, on=["figure_path", "question"], how="inner")

merged_df = merged_df.drop(columns=[
    "model_output", 
    "true_answer", 
    "similarity_to_answer", 
    "max_similarity_to_choices"
])

# Save result
merged_df.to_csv("1k_pretrace.csv", index=False)