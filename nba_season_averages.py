import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

# Define the seasons (NBA season format: 'YYYY-YY')
seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(1999, 2026)]

# Ensure data directory exists
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

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


for season in seasons:
    print(f"Fetching data for {season}...")

    # Regular season player stats.
    df = fetch_player_stats(
        season=season,
        season_type='Regular Season',
        per_mode='PerGame',
        timeout_seconds=120,
        max_retries=5,
    )

    # All-Star player pool for the same season.
    all_star_df = fetch_player_stats(
        season=season,
        season_type='All Star',
        per_mode='PerGame',
        timeout_seconds=120,
        max_retries=5,
    )
    all_star_player_ids = set(all_star_df['PLAYER_ID']) if 'PLAYER_ID' in all_star_df.columns else set()

    df = df.sort_values(by='PLAYER_NAME')
    df['SEASON'] = season
    df['ALL_STAR'] = df['PLAYER_ID'].isin(all_star_player_ids).astype(int)
    df = df.drop(columns=columns_to_drop, errors='ignore')
    all_season_frames.append(df)

    # Light pacing between seasons to reduce rate-limiting pressure.
    time.sleep(2)

combined_output_path = os.path.join(output_dir, 'nba_player_season_averages_all_seasons.csv')

if all_season_frames:
    combined_df = (
        pd.concat(all_season_frames, ignore_index=True)
        .sort_values(by=['SEASON', 'PLAYER_NAME'])
        .reset_index(drop=True)
    )
    combined_df.to_csv(combined_output_path, index=False)
    print(f"Saved combined file: {combined_output_path}")
else:
    print('No data was fetched. Combined file was not created.')
