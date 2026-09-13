# Data dictionary

Column names, units and meaning. Nothing here describes data quality — judge that yourself.

---

## `sleep_sessions.csv`

One row per sleep session recorded by the ring. Includes every session the device logged.

| Column | Type | Meaning |
|---|---|---|
| `session_id` | string | Identifier for the session |
| `user_id` | string | Identifier for the user |
| `session_start` | string | When the session began, as recorded by the ingesting client |
| `session_end` | string | When the session ended, same source |
| `time_in_bed_minutes` | number | Length of the session |
| `total_sleep_minutes` | number | Time asleep within the session |
| `deep_minutes` | number | Time in deep sleep |
| `rem_minutes` | number | Time in REM sleep |
| `light_minutes` | number | Time in light sleep |
| `awake_minutes` | number | Time awake within the session |
| `sleep_efficiency` | number | Device's own ratio of sleep to time in bed |
| `avg_hr_bpm` | number | Mean heart rate across the session, beats per minute |
| `min_hr_bpm` | number | Lowest sustained heart rate, beats per minute |
| `avg_hrv_rmssd_ms` | number | Mean heart rate variability (RMSSD), milliseconds |
| `avg_spo2_pct` | number | Mean blood oxygen saturation, percent |
| `avg_resp_rate_bpm` | number | Mean respiratory rate, breaths per minute |
| `temperature_deviation_c` | number | Skin temperature relative to the user's own baseline, degrees Celsius |
| `awakenings_count` | integer | Number of separate wake periods |
| `movement_index` | number | Mean movement, arbitrary units. Higher means more restless |
| `firmware_version` | string | Ring firmware that produced this session |

## `nightly_signals.csv.gz`

Sensor readings at five-minute resolution inside each sleep session.

| Column | Type | Meaning |
|---|---|---|
| `session_id` | string | Links to `sleep_sessions.session_id` |
| `user_id` | string | Identifier for the user |
| `timestamp_utc` | datetime | Start of the five-minute window, **in UTC** |
| `hr_bpm` | number | Heart rate, beats per minute |
| `hrv_rmssd_ms` | number | Heart rate variability (RMSSD), milliseconds |
| `motion_index` | number | Movement, arbitrary units |
| `temp_delta_c` | number | Skin temperature relative to baseline, degrees Celsius |
| `spo2_pct` | number | Blood oxygen saturation, percent |

Sleep stages are not given at this resolution. The per-session totals are in `sleep_sessions.csv`.

## `user_profiles.csv`

| Column | Type | Meaning |
|---|---|---|
| `user_id` | string | Identifier for the user |
| `age_years` | integer | Age |
| `sex` | string | `M` or `F` as recorded at onboarding |
| `height_cm` | number | Height, centimetres |
| `weight` | number | Body weight, **as the user entered it in the app** |
| `timezone` | string | IANA timezone the user lives in |
| `plan_tier` | string | `free` or `plus` |
| `onboarding_date` | date | When the user joined |
| `profile_updated_at` | date | When this profile row was last written |

## `daily_context.csv`

One row per user per calendar day.

| Column | Type | Meaning |
|---|---|---|
| `userId` | string | Identifier for the user |
| `date` | date | The **waking day** this row describes — the day's activity, before that night's sleep |
| `steps` | integer | Step count for the day |
| `active_minutes` | integer | Minutes of activity above a movement threshold |
| `workout_notes` | string | Free text the user typed into the workout log |
| `alcohol_units` | number | Units the user logged |
| `caffeine_mg` | number | Caffeine the user logged, milligrams |
| `last_caffeine_hours_before_bed` | number | Hours between the last logged caffeine and bedtime |
| `stress_score` | number | Self-reported stress, 1 to 100. Optional prompt |
| `travel_flag` | integer | 1 if the user crossed a timezone that day |
| `legacy_readiness_shown` | integer | The Readiness score the app displayed **on the morning after this date**, produced by the heuristic in production today, 1 to 100 |

## `morning_checkins.csv`

The target. An optional prompt shown in the app after the user wakes.

| Column | Type | Meaning |
|---|---|---|
| `user_id` | string | Identifier for the user |
| `date` | date | The **morning** the user answered on |
| `subjective_feeling` | integer | How the user says they feel, 1 (worst) to 5 (best) |
| `mood_tags` | string | Free-text tags the user picked or typed |
| `submitted_at` | datetime | When they answered, in their own local time |
