import os
import csv
import math
import numpy as np
from PIL import Image
from sklearn.mixture import GaussianMixture
from scipy.stats import ttest_rel
import pandas as pd

# 1. Define the 8 Primary Anchors (Baseline) from Table 2 of your paper
PRIMARY_8_ANCHORS = [
    (255, 255, 0),   # Joy
    (0, 200, 0),     # Trust
    (100, 56, 135),  # Fear
    (0, 170, 138),   # Surprise
    (70, 100, 173),  # Sadness
    (128, 128, 0),   # Disgust
    (220, 0, 0),     # Anger
    (255, 155, 0)    # Anticipation
]

# 2. Load the 48 Anchors from your CSV
def load_48_anchors():
    anchors = []
    with open('data/labels.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            anchors.append((int(row['R']), int(row['G']), int(row['B'])))
    return anchors

# 3. K-Means/GMM Feature Extraction (Strictly K=25 as per paper/app)
def extract_clusters(image_path, k=25):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((250, 250), Image.NEAREST)
    img_data = np.array(image).reshape(-1, 3)
    
    unique_colors = np.unique(img_data, axis=0)
    actual_k = min(k, len(unique_colors))
    
    # Gaussian Mixture refinement
    model = GaussianMixture(n_components=actual_k, covariance_type='tied', random_state=42)
    model.fit(img_data)
    labels = model.predict(img_data)
    
    _, counts = np.unique(labels, return_counts=True)
    total = sum(counts)
    
    colors = [tuple(map(int, model.means_[i])) for i in range(actual_k)]
    weights = [count / total for count in counts] # Normalized proportion (0.0 to 1.0)
    
    return colors, weights

# 4. WEA Mathematical Formula (Linearly Normalized Proximity)
def calculate_wea(colors, weights, anchors):
    wea_score = 0
    max_dist = math.sqrt(255**2 * 3) # Max RGB Euclidean distance (~441.67)
    
    for cluster_color, weight in zip(colors, weights):
        # Find distance to nearest anchor
        min_dist = min(math.dist(cluster_color, anchor) for anchor in anchors)
        
        # MC_i = 1 - (d / d_max) (Matches your updated LaTeX paper exactly)
        mc_i = 1.0 - (min_dist / max_dist) 
        
        # WEA sum
        wea_score += (weight * mc_i)
        
    return wea_score

# 5. Main Validation Loop
def run_validation():
    anchors_48 = load_48_anchors()
    dataset_dir = 'dataset'
    # Updated to match your exact 4 curated styles
    styles = ['Cubism', 'Impressionism', 'Nihonga', 'Romanticism']
    
    results = []
    
    print("Starting Academic Validation Pipeline...")
    for style in styles:
        print(f"Processing style: {style}...")
        style_dir = os.path.join(dataset_dir, style)
        
        if not os.path.exists(style_dir):
            print(f"Directory {style_dir} not found. Skipping.")
            continue
            
        wea_8_list = []
        wea_48_list = []
        
        for filename in os.listdir(style_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(style_dir, filename)
                
                # Extract K=25 clusters
                colors, weights = extract_clusters(img_path, k=25)
                
                # Calculate Baseline WEA (8 Anchors)
                wea_8 = calculate_wea(colors, weights, PRIMARY_8_ANCHORS)
                wea_8_list.append(wea_8)
                
                # Calculate Proposed WEA (48 Anchors)
                wea_48 = calculate_wea(colors, weights, anchors_48)
                wea_48_list.append(wea_48)
        
        # Calculate Statistics for the Style
        if len(wea_48_list) > 0:
            mean_8 = np.mean(wea_8_list)
            mean_48 = np.mean(wea_48_list)
            std_48 = np.std(wea_48_list)
            
            # Paired t-test
            t_stat, p_value = ttest_rel(wea_48_list, wea_8_list)
            
            # Calculate Mean Difference (D_bar) and Cohen's d
            differences = np.array(wea_48_list) - np.array(wea_8_list)
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1) # Sample standard deviation
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0
            
            results.append({
                'Style': style,
                'Samples': len(wea_48_list),
                'Baseline (8)': round(mean_8, 3),
                'Proposed (48)': round(mean_48, 3),
                'Mean Diff': f"+{round(mean_diff, 3)}", # Added '+' sign for clarity
                'Std Dev (48)': round(std_48, 3),
                'Cohen d': round(cohens_d, 2),
                'p-value': "< 0.001" if p_value < 0.001 else round(p_value, 3)
            })

    # Export to CSV and Print
    df = pd.DataFrame(results)
    
    # Reordering columns to match the LaTeX table output perfectly
    columns_order = ['Style', 'Samples', 'Baseline (8)', 'Proposed (48)', 'Mean Diff', 'Std Dev (48)', 'Cohen d', 'p-value']
    df = df[columns_order]
    
    df.to_csv('validation_results.csv', index=False)
    print("\n--- VALIDATION COMPLETE ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_validation()