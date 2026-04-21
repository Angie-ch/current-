#!/usr/bin/env python3
"""快速统计所有有效台风数量"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from paper_eval_common import load_track_data, traj_data_cfg, traj_model_cfg, heuristic_forecast_start_idx

track_csv = 'processed_typhoon_tracks.csv'
df_full = pd.read_csv(track_csv)
all_ids = sorted(df_full['typhoon_id'].unique().tolist())

lat_range = traj_data_cfg.lat_range
lon_range = traj_data_cfg.lon_range

# Pre-compute bounds per typhoon (single groupby, fast)
grouped = df_full.groupby('typhoon_id').agg({
    'lat': ['min', 'max'],
    'lon': ['min', 'max', 'count']
})
grouped.columns = ['lat_min', 'lat_max', 'lon_min', 'lon_max', 'count']

valid = []
skipped_insufficient = []
skipped_bounds = []
skipped_invalid_start = []

for storm_id in all_ids:
    bounds = grouped.loc[storm_id]
    # Convert lon to 0-360 if needed
    lon_min = bounds['lon_min']
    lon_max = bounds['lon_max']
    if lon_min < 0:
        lon_min = lon_min % 360
        lon_max = bounds['lon_max'] % 360
    lat_min = bounds['lat_min']
    lat_max = bounds['lat_max']
    # Quick bounds check (no CSV read)
    lat_ok = (lat_min >= lat_range[0]) and (lat_max <= lat_range[1])
    lon_ok = (lon_min >= lon_range[0]) and (lon_max <= lon_range[1])
    if not (lat_ok and lon_ok):
        skipped_bounds.append(storm_id)
        continue
    # Now load full track to check length and forecast_start_idx (only for bounds-passing typhoons)
    track_df = load_track_data(track_csv, storm_id)
    if track_df is None:
        continue
    if len(track_df) < traj_model_cfg.t_history + traj_model_cfg.t_future:
        skipped_insufficient.append(storm_id)
        continue
    forecast_start_idx = heuristic_forecast_start_idx(track_df, traj_model_cfg.t_history, traj_model_cfg.t_future)
    if forecast_start_idx < traj_model_cfg.t_history:
        skipped_invalid_start.append(storm_id)
        continue
    valid.append(storm_id)

print(f'Total typhoons in CSV: {len(all_ids)}')
print(f'Valid (pass all filters): {len(valid)}')
print(f'Skipped - insufficient length: {len(skipped_insufficient)}')
print(f'Skipped - out of bounds: {len(skipped_bounds)}')
print(f'Skipped - invalid start: {len(skipped_invalid_start)}')
print(f'\nFirst 20 valid typhoon IDs:')
for tid in valid[:20]:
    print(f'  {tid}')
