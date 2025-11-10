from src.preprocess import preprocess_dataset
import os

if __name__ == "__main__":
    input_path = os.path.join("data", "dataset.csv")
    output_path = os.path.join("data", "processed_dataset.csv")

    if not os.path.exists(input_path):
        print(f"❌ ERROR: dataset not found at {input_path}")
    else:
        processed_df = preprocess_dataset(input_path)
        processed_df.to_csv(output_path, index=False)
        print(f"📁 Processed dataset saved to: {output_path}")
