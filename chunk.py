import pandas as pd
from typing import List
import os


def preprocess_table_to_chunks(table: pd.DataFrame, chunk_size: int = 10) -> List[pd.DataFrame]:
    """
    Preprocess a table (DataFrame) and convert it into document chunks.
    Retains blank or NaN values in the processing.

    :param table: Pandas DataFrame representing the table.
    :param chunk_size: Number of rows per chunk.
    :return: List of DataFrame chunks.
    """
    chunks = []
    # Iterate through the DataFrame in chunks
    for start_row in range(0, len(table), chunk_size):
        end_row = start_row + chunk_size
        chunk = table.iloc[start_row:end_row]
        chunks.append(chunk)
    return chunks


def save_chunks(chunks: List[pd.DataFrame], output_dir: str):
    """
    Save each chunk as a separate CSV file.

    :param chunks: List of DataFrame chunks to save.
    :param output_dir: Directory where the chunks will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(output_dir, f"chunk_{i + 1}.csv")
        chunk.to_csv(chunk_file, index=False)
        print(f"Saved: {chunk_file}")


def main(data_path: str, chunk_size: int, output_dir: str):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Preprocess table into chunks
    chunks = preprocess_table_to_chunks(df, chunk_size=chunk_size)

    # Save the chunks for later processing
    save_chunks(chunks, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocess and save table chunks for later use.")
    parser.add_argument("data", type=str, help="Path to the dataset file (CSV format).")
    parser.add_argument("--chunk_size", type=int, default=10, help="Number of rows per chunk.")
    parser.add_argument("--output_dir", type=str, default="./chunks", help="Directory to save the chunks.")

    args = parser.parse_args()

    # Run the main function
    main(data_path=args.data, chunk_size=args.chunk_size, output_dir=args.output_dir)
