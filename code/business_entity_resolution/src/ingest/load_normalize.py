import pandas as pd
import re
import argparse
import time
import os

LEGAL_SUFFIX_MAP = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "llp": "llp",
    "llc": "llc",
    "pvt ltd": "pvt ltd",
    "co": "co",
    "company": "co"
}

ADDR_ABBREV = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "highway": "hwy",
    "building": "bldg",
    "apartment": "apt",
    "suite": "ste"
}

def normalize_names(series: pd.Series) -> pd.Series:
    """
    Normalizes business names using vectorized pandas operations.
    Preserves Unicode (Devanagari etc.) via word-char regex patterns.
    """
    s = series.fillna("").astype(str).str.lower()
    # Unify & and 'and'
    s = s.str.replace("&", " and ", regex=False)
    # Strip all non-alphanumeric/non-whitespace characters (preserves unicode word chars)
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    # Remove underscores explicitly since \w includes them
    s = s.str.replace(r'_', ' ', regex=True)
    
    # Canonicalize legal suffixes using word boundaries
    suffix_regex = r'\b(' + '|'.join(LEGAL_SUFFIX_MAP.keys()) + r')\b'
    s = s.str.replace(suffix_regex, lambda m: LEGAL_SUFFIX_MAP[m.group(1)], regex=True)
    
    # Collapse multiple spaces into one and strip
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def normalize_addresses(series: pd.Series) -> pd.Series:
    """
    Normalizes addresses with abbreviation expansion.
    """
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    s = s.str.replace(r'_', ' ', regex=True)
    
    abbrev_regex = r'\b(' + '|'.join(ADDR_ABBREV.keys()) + r')\b'
    s = s.str.replace(abbrev_regex, lambda m: ADDR_ABBREV[m.group(1)], regex=True)
    
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def load_and_normalize(filepath: str) -> pd.DataFrame:
    """
    Loads raw TSV and appends normalized columns.
    """
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep="\t", encoding="utf-8", encoding_errors="replace")
    
    print("Normalizing names...")
    df['norm_name'] = normalize_names(df['business_name'])
    
    print("Normalizing addresses...")
    df['norm_address'] = normalize_addresses(df['business_address'])
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Ingest and Normalize dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to input TSV")
    parser.add_argument("--output", type=str, required=True, help="Path to output TSV")
    args = parser.parse_args()
    
    start = time.time()
    df = load_and_normalize(args.input)
    
    print(f"Writing to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)
    
    print(f"Processed {len(df)} rows in {time.time() - start:.2f}s")
    
    # Print a small sample for visual verification
    print("\n--- SAMPLE OUTPUT ---")
    sample = df[['business_name', 'norm_name', 'business_address', 'norm_address']].head(3)
    for _, row in sample.iterrows():
        print(f"Original Name : {row['business_name']}")
        print(f"Norm Name     : {row['norm_name']}")
        print(f"Original Addr : {row['business_address']}")
        print(f"Norm Addr     : {row['norm_address']}")
        print("-" * 40)
