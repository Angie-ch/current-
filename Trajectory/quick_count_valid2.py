#!/usr/bin/env python3
"""快速统计所有有效台风数量 - 一次读取CSV"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from paper_eval_common import traj_data_cfg, traj_model_cfg, heuristic_forecast_start_idx

track_csv = 'processed_typhoon_tracks.csv'
df = pd.read_csv(track_csv)

lat_range = traj_data_cfg.lat_range
lon_range = traj_data_cfg.lon_range

# Group by typhoon_id once
grouped = df.groupby('typhoon_id').agg({
    'lat': ['min', 'max', 'count'],
    'lon': ['min', 'max']
})
grouped.columns = ['lat_min', 'lat_max', 'lat_count', 'lon_min', 'lon_max']

valid = []
skipped_insufficient = []
skipped_bounds = []
skipped_invalid_start = []

for storm_id, row in grouped.iterrows():
    # Convert lon to 0-360 if needed
    lon_min = row['lon_min']
    lon_max = row['lon_max']
    if lon_min < 0:
        lon_min = lon_min % 360
        lon_max = row['lon_max'] % 360
    lat_min = row['lat_min']
    lat_max = row['lat_max']
    # Bounds check
    lat_ok = (lat_min >= lat_range[0]) and (lat_max <= lat_range[1])
    lon_ok = (lon_min >= lon_range[0]) and (lon_max <= lon_range[1])
    if not (lat_ok and lon_ok):
        skipped_bounds.append(storm_id)
        continue
    # Length check (already have count from groupby)
    if row['lat_count'] < traj_model_cfg.t_history + traj_model_cfg.t_future:
        skipped_insufficient.append(storm_id)
        continue
    # Need to load full track for forecast_start_idx check
    # Load just this typhoon's rows from the already-read dataframe (FAST)
    track_df = df[df['typhoon_id'] == storm_id].copy()
    track_df = track_df.sort_index().reset_index(drop=True)
    forecast_start_idx = heuristic_forecast_start_idx(track_df, traj_model_cfg.t_history, traj_model_cfg.t_future)
    if forecast_start_idx < traj_model_cfg.t_history:
        skipped_invalid_start.append(storm_id)
        continue
    valid.append(storm_id)

print(f'Total typhoons in CSV: {len(grouped)}')
print(f'Valid (pass all filters): {len(valid)}')
print(f'Skipped - insufficient length: {len(skipped_insufficient)}')
print(f'Skipped - out of bounds: {len(skipped_bounds)}')
print(f'Skipped - invalid start: {len(skipped_invalid_start)}')
print(f'\nFirst 20 valid typhoon IDs:')
for tid in valid[:20]:
    print(f'  {tid}')
