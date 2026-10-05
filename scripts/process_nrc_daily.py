"""Extract one NRC daily HTML report into compressed Parquet.
Usage: python scripts/process_nrc_daily.py
Recursively parses data/raw/nrc and writes daily Parquet files by year.
"""
import argparse
import json
import re
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup


def extract_report(path):
    soup = BeautifulSoup(Path(path).read_bytes(), 'html.parser')
    canonical = soup.find('link', rel='canonical')
    url = canonical.get('href', '') if canonical else ''
    match = re.search(r'/(\d{8})ps/?$', url)
    if not match:
        raise ValueError('Cannot determine report date from canonical URL')
    report_date = pd.to_datetime(match.group(1), format='%Y%m%d')
    title = soup.find('h1') or soup.title
    title_match = re.search(r'for (\w+ \d{1,2}, \d{4})', title.get_text(' ', strip=True)) if title else None
    if title_match and pd.to_datetime(title_match.group(1)) != report_date:
        raise ValueError('Report title and URL dates disagree')
    records = []
    regions = set()
    expected = ['Unit', 'Power', 'Down', 'Reason or Comment', 'Change in Report (*)', 'Scrams (#)']
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if not rows:
            continue
        headers = [c.get_text(' ', strip=True) for c in rows[0].find_all(['td', 'th'])]
        if not headers or headers[0] != 'Unit':
            continue
        if headers != expected:
            raise ValueError(f'Unexpected headers: {headers}')
        heading = table.find_previous(['h2', 'h3', 'h4'])
        region_match = re.fullmatch(r'Region\s+(\d+)', heading.get_text(' ', strip=True)) if heading else None
        if not region_match:
            raise ValueError('Cannot determine table region')
        region = int(region_match.group(1))
        regions.add(region)
        for row in rows[1:]:
            cells = [c.get_text(' ', strip=True) for c in row.find_all(['td', 'th'])]
            if not cells:
                continue
            if len(cells) != 6 or not cells[0]:
                raise ValueError(f'Unexpected row: {cells}')
            unit, power, down, reason, change, scrams = cells
            records.append(dict(date=report_date, region=region, unit_name=unit,
                power_pct=power or None, down_raw=down or None,
                reason_comment=reason or None, change_raw=change or None,
                scrams_raw=scrams or None, source_url=url))
    if regions != {1, 2, 3, 4}:
        raise ValueError(f'Expected four regions; found {regions}')
    df = pd.DataFrame(records)
    df['region'] = df['region'].astype('Int8')
    df['power_pct'] = pd.to_numeric(df['power_pct'], errors='raise').astype('Float32')
    for col in ['unit_name', 'down_raw', 'reason_comment', 'change_raw', 'scrams_raw', 'source_url']:
        df[col] = df[col].astype('string')
    if df.duplicated(['date', 'unit_name']).any():
        raise ValueError('Duplicate reactor-date records')
    if df['power_pct'].dropna().lt(0).any():
        raise ValueError('Negative power value')
    return df.sort_values(['region', 'unit_name']).reset_index(drop=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=root / 'data/raw/nrc')
    parser.add_argument('--output-dir', type=Path, default=root / 'data/processed/nrc/daily')
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f'Input directory does not exist: {args.input_dir}')
    files = sorted(p for p in args.input_dir.rglob('*')
                   if p.is_file() and p.suffix.lower() in {'.html', '.htm'})
    if not files:
        parser.error(f'No HTML files found in {args.input_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    seen_dates = {}
    completed = 0
    total_rows = 0
    for path in files:
        try:
            df = extract_report(path)
            day = df['date'].iloc[0]
            date_key = day.strftime('%Y-%m-%d')
            if date_key in seen_dates:
                raise ValueError(f'Duplicate report date; also found in {seen_dates[date_key]}')
            seen_dates[date_key] = str(path)
            destination = args.output_dir / str(day.year) / f'{day:%Y%m%d}ps.csv'
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.csv.part')
            df.to_csv(temporary, index=False, encoding='utf-8', date_format='%Y-%m-%d')
            temporary.replace(destination)
            completed += 1
            total_rows += len(df)
            print(f'{path.name}: {len(df)} rows -> {destination}', flush=True)
        except Exception as error:
            failures.append({'input_file': str(path), 'error': str(error)})
            print(f'FAILED {path}: {error}', flush=True)
    log = args.output_dir / 'processing_failures.json'
    log.write_text(json.dumps(failures, indent=2) + '\n', encoding='utf-8')
    print(f'Processed {completed}/{len(files)} reports; {total_rows} rows; {len(failures)} failures.')
    if failures:
        raise SystemExit(f'Review failures in {log}')


if __name__ == '__main__':
    main()
