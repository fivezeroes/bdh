#!/usr/bin/env python3
"""
search_parquet.py - Search parquet files for text matches.

Usage examples:
    python search_parquet.py --dir /mnt/scratch/data --pattern "openai" --context 40
  python search_parquet.py --dir /mnt/scratch/data --pattern "OpenAI" --regex --case-sensitive
  python search_parquet.py --files "data/*.parquet" --pattern "http" --summary
    python search_parquet.py --files "data/*.parquet" --pattern "foo" --filter "language==en" --filter "score<0.6"
"""

import argparse
import re
import os
import glob
import sys
from typing import List, Iterable, Tuple, Dict, Any, Optional
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed

# If your repo root needs to be importable for fetch_parquet_files:
# sys.path.insert(0, os.path.dirname(__file__))

try:
    from data import fetch_parquet_files
except Exception:
    # Not mandatory; optional fallback if fetch_parquet_files can't be imported.
    fetch_parquet_files = None


def discover_files(dir_path: str = None, file_glob: str = None) -> List[str]:
    if file_glob:
        return sorted(glob.glob(file_glob))
    if dir_path:
        if fetch_parquet_files is not None:
            return fetch_parquet_files(dir_path)
        else:
            return sorted(glob.glob(os.path.join(dir_path, "*.parquet")))
    raise ValueError("Either --dir or --files must be provided")


def _search_in_text(text: str, pattern: str, regex: bool, case_sensitive: bool) -> Iterable[Tuple[int, int]]:
    if text is None:
        return ()
    if not case_sensitive:
        text_search = text.lower()
        pattern_search = pattern.lower()
    else:
        text_search = text
        pattern_search = pattern

    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error:
            return ()
        return [(m.start(), m.end()) for m in compiled.finditer(text)]
    else:
        # find all non-overlapping occurrences
        start = 0
        results = []
        while True:
            idx = text_search.find(pattern_search, start)
            if idx == -1:
                break
            results.append((idx, idx + len(pattern)))
            start = idx + len(pattern_search)
        return results


def _search_file(file_path: str, pattern: str, regex: bool, case_sensitive: bool, columns: List[str], context: int, max_matches_per_file: int, filters: Optional[List[Dict]] = None) -> Dict[str, Any]:
    # Backwards-compatible signature, filter logic may be added later
    results = []
    try:
        pf = pq.ParquetFile(file_path)
    except Exception as e:
        return {"file": file_path, "error": str(e), "matches": []}

    # Filters is a list of dicts with keys: col, op, val, is_num
    if filters is None:
        filters = []

    # Count through row groups to compute a global row index
    row_group_counts = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    cum_counts = [0]
    for n in row_group_counts:
        cum_counts.append(cum_counts[-1] + n)

    matches_found = 0
    # Determine all columns we need to read: search columns + filter columns
    filter_cols = {f['col'] for f in filters}
    read_cols = list(dict.fromkeys(columns + list(filter_cols)))  # preserve order and unique

    for rg in range(pf.num_row_groups):
        try:
            table = pf.read_row_group(rg, columns=read_cols)
        except Exception:
            # try reading whole table
            try:
                table = pf.read(columns=read_cols)
            except Exception:
                continue
        # Convert needed columns to Python lists for row-wise access
        col_lists = {}
        for col in read_cols:
            if col not in table.column_names:
                col_lists[col] = None
                continue
            col_lists[col] = table.column(col).to_pylist()

        # row-wise iteration to apply filters
        # find number of rows in this row group: pick one non-None column list to determine length
        num_rows = 0
        for l in col_lists.values():
            if l is not None:
                num_rows = len(l)
                break
        if num_rows == 0:
            continue

        def _value_matches_filter(value, f):
            op = f['op']
            expected = f['val']
            is_num = f['is_num']

            # Handle None expected
            if expected is None:
                if op == '==':
                    return value is None
                if op == '!=':
                    return value is not None
                # other ops don't make sense with None
                return False

            # If value is None and expected is not None -> mismatch
            if value is None:
                return False

            # numeric comparison if expected was numeric
            if is_num:
                try:
                    val_num = float(value)
                except Exception:
                    return False
                if op == '==':
                    return val_num == float(expected)
                if op == '!=':
                    return val_num != float(expected)
                if op == '<=':
                    return val_num <= float(expected)
                if op == '>=':
                    return val_num >= float(expected)
                if op == '<':
                    return val_num < float(expected)
                if op == '>':
                    return val_num > float(expected)
                return False

            # string comparison (respect case sensitivity flag)
            try:
                val_str = str(value)
            except Exception:
                return False
            exp_str = str(expected)
            if not case_sensitive:
                val_str = val_str.lower()
                exp_str = exp_str.lower()

            if op == '==':
                return val_str == exp_str
            if op == '!=':
                return val_str != exp_str
            # lexicographic compare for <, > with strings
            if op == '<=':
                return val_str <= exp_str
            if op == '>=':
                return val_str >= exp_str
            if op == '<':
                return val_str < exp_str
            if op == '>':
                return val_str > exp_str
            return False

        # Now iterate rows
        for local_idx in range(num_rows):
            # Evaluate filters first
            match_filters = True
            for f in filters:
                col = f['col']
                if col not in col_lists or col_lists[col] is None:
                    match_filters = False
                    break
                value = col_lists[col][local_idx]
                if not _value_matches_filter(value, f):
                    match_filters = False
                    break
            if not match_filters:
                continue

            # Search only in the columns specified by 'columns'
            for col in columns:
                if col not in col_lists or col_lists[col] is None:
                    continue
                text = col_lists[col][local_idx]
                if text is None:
                    continue
                if not isinstance(text, str):
                    text = str(text)

                spans = _search_in_text(text, pattern, regex, case_sensitive)
                if spans:
                    global_row_idx = cum_counts[rg] + local_idx
                    for start, end in spans:
                        snippet = text[max(0, start - context):min(len(text), end + context)]
                        results.append({"row": global_row_idx, "col": col, "start": start, "end": end, "snippet": snippet})
                        matches_found += 1
                        if max_matches_per_file and matches_found >= max_matches_per_file:
                            break
                if max_matches_per_file and matches_found >= max_matches_per_file:
                    break
            if max_matches_per_file and matches_found >= max_matches_per_file:
                break
        if max_matches_per_file and matches_found >= max_matches_per_file:
            break

    return {"file": file_path, "error": None, "matches": results}


def main():
    parser = argparse.ArgumentParser(description="Search parquet files for text or regex matches.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", "-d", help="Directory that contains parquet files")
    group.add_argument("--files", "-f", help="Glob for parquet files, e.g. 'data/*.parquet'")
    parser.add_argument("--pattern", "-p", required=False, default=None, help="Search pattern (plain text or regex if --regex). If omitted, rows matching filters will be shown.")
    parser.add_argument("--regex", action="store_true", help="Interpret --pattern as a regex")
    parser.add_argument("--case-sensitive", action="store_true", help="Make search case-sensitive (default: false)")
    parser.add_argument("--cols", default="text", help="Comma-separated columns to search (default: 'text')")
    parser.add_argument("--filter", "-F", action="append", default=[], help="Filter expression(s) to restrict rows, e.g. 'language==en' or 'score<0.6' (can be used multiple times)")
    parser.add_argument("--context", "-c", type=int, default=40, help="Characters of context to show around match")
    parser.add_argument("--max-per-file", type=int, default=10, help="Max number of matches to show per file (0 = unlimited)")
    parser.add_argument("--summary", action="store_true", help="Only show counts (file and number of matches)")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers to scan files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    files = discover_files(dir_path=args.dir, file_glob=args.files)
    if not files:
        print("No parquet files found.")
        return

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    # parse filters; they will be a list of strings like 'language==en'
    filters_raw = args.filter

    def parse_filter_str(s: str):
        # Supports operators: == != <= >= < >
        for op in ['==', '!=', '<=', '>=', '<', '>']:
            if op in s:
                parts = s.split(op)
                if len(parts) != 2:
                    raise ValueError(f"Invalid filter: {s}")
                col = parts[0].strip()
                val = parts[1].strip()
                # try to coerce value to number if possible
                is_num = False
                coerced_val = val
                if val.lower() == 'none' or val.lower() == 'null':
                    coerced_val = None
                else:
                    try:
                        coerced_val = float(val)
                        is_num = True
                    except Exception:
                        # strip surrounding quotes if present
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            coerced_val = val[1:-1]
                        else:
                            coerced_val = val
                return {'col': col, 'op': op, 'val': coerced_val, 'is_num': is_num}
        raise ValueError(f"Invalid filter, must contain a comparison operator: {s}")

    filters = []
    for f in filters_raw:
        if f:
            try:
                filters.append(parse_filter_str(f))
            except ValueError as e:
                print(f"Error parsing filter '{f}': {e}")
                return

    if args.verbose and filters:
        print(f"Applying filters: {filters}")

    # Validate regex/pattern usage
    if args.regex and not args.pattern:
        print("Error: --regex requires --pattern to be provided")
        return

    if args.workers and args.workers > 1:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(_search_file, fpath, args.pattern, args.regex, args.case_sensitive, cols, args.context, args.max_per_file, filters): fpath
                for fpath in files
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"file": futures[fut], "error": str(e), "matches": []})
    else:
        results = [ _search_file(fpath, args.pattern, args.regex, args.case_sensitive, cols, args.context, args.max_per_file, filters) for fpath in files ]

    total = 0
    for r in results:
        if r["error"]:
            if args.verbose:
                print(f"ERROR reading {r['file']}: {r['error']}")
            continue

        count = len(r["matches"])
        total += count
        if args.summary:
            print(f"{r['file']}: {count} matches")
            continue

        if count == 0:
            if args.verbose:
                print(f"{r['file']}: 0 matches")
            continue

        print("=" * 80)
        print(f"File: {r['file']} ({count} matches)")
        for m in r["matches"]:
            print(f"  Row: {m['row']:<8} Column: {m['col']} snippet: {repr(m['snippet'])}")

    print("=" * 80)
    print(f"Total matches across files: {total}")


if __name__ == "__main__":
    main()