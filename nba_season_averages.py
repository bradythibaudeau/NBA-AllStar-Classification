# Script to fetch NBA player season averages from 1999-2026 and mark All-Star players
# Data is combined into a single CSV file for use in classification modeling

import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

# Define the seasons to fetch (NBA season format: 'YYYY-YY', e.g., '1999-00', '2000-01')
# Spans from 1999-2000 through 2025-26
seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(1999, 2026)]

# Create output directory for storing the combined CSV file
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

# List of columns to exclude from the final dataset (mostly rank columns and fantasy points)
# These are redundant for classification and increase noise
columns_to_drop = [
    'NICKNAME',
    'TEAM_ID',
    'NBA_FANTASY_PTS',
    'WNBA_FANTASY_PTS',
    'GP_RANK',
    'W_PCT_RANK',
    'MIN_RANK',
    'FGM_RANK',
    'FGA_RANK',
    'FG_PCT_RANK',
    'FG3M_RANK',
    'FG3A_RANK',
    'FG3_PCT_RANK',
    'FTM_RANK',
    'FTA_RANK',
    'FT_PCT_RANK',
    'OREB_RANK',
    'DREB_RANK',
    'REB_RANK',
    'AST_RANK',
    'STL_RANK',
    'BLK_RANK',
    'TOV_RANK',
    'PF_RANK',
    'PTS_RANK',
    'TEAM_COUNT',
    'PFD_RANK',
    'PLUS_MINUS_RANK',
    'NBA_FANTASY_PTS_RANK',
    'WNBA_FANTASY_PTS_RANK',
    'W_RANK',
    'L_RANK',
    'BLKA_RANK',
    'DD2',
    'TD3',
    'DD2_RANK',
    'TD3_RANK',
]

# List to accumulate dataframes for each season before combining
all_season_frames = []


def fetch_player_stats(season, season_type, per_mode='PerGame', timeout_seconds=120, max_retries=5):
    """Fetch a season split with retries to handle transient NBA stats API timeouts."""
    for attempt in range(1, max_retries + 1):
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed=per_mode,
                season_type_all_star=season_type,
                timeout=timeout_seconds,
            )
            return stats.get_data_frames()[0]
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to fetch {season} ({season_type}) after {max_retries} attempts"
                ) from exc

            wait_seconds = min(10 * attempt, 60)
            print(
                f"Request failed for {season} ({season_type}) on attempt {attempt}/{max_retries}: {exc}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)


# Main loop: fetch stats for each season
for season in seasons:
    print(f"Fetching data for {season}...")

    # Fetch regular season player statistics (per-game averages)
    df = fetch_player_stats(
        season=season,
        season_type='Regular Season',
        per_mode='PerGame',
        timeout_seconds=120,
        max_retries=5,
    )

    # Fetch All-Star player pool for the same season to identify All-Stars
    all_star_df = fetch_player_stats(
        season=season,
        season_type='All Star',
        per_mode='PerGame',
        timeout_seconds=120,
        max_retries=5,
    )
    # Extract All-Star player IDs from the All-Star dataset
    all_star_player_ids = set(all_star_df['PLAYER_ID']) if 'PLAYER_ID' in all_star_df.columns else set()

    # Process the regular season dataframe
    df = df.sort_values(by='PLAYER_NAME')
    df['SEASON'] = season  # Add season identifier
    df['ALL_STAR'] = df['PLAYER_ID'].isin(all_star_player_ids).astype(int)  # Binary flag: 1 if All-Star, 0 otherwise
    df = df.drop(columns=columns_to_drop, errors='ignore')  # Remove unnecessary columns
    all_season_frames.append(df)  # Add to the list of seasons

    # Pause between requests to avoid rate-limiting from the NBA API
    time.sleep(2)

# Combine all seasons into a single dataframe and save to CSV
combined_output_path = os.path.join(output_dir, 'nba_player_season_averages_all_seasons.csv')

if all_season_frames:
    # Concatenate all season dataframes, sort by season and player name, and reset index
    combined_df = (
        pd.concat(all_season_frames, ignore_index=True)
        .sort_values(by=['SEASON', 'PLAYER_NAME'])
        .reset_index(drop=True)
    )
    # Save the combined dataset to CSV for use in model training
    combined_df.to_csv(combined_output_path, index=False)
    print(f"Saved combined file: {combined_output_path}")
else:
    print('No data was fetched.')
