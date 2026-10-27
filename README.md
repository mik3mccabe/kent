# Kent Brockman Frame Finder v3.3 — Replace old Script Scan scenes on load

## Change

- **Test selected episode** starts a fresh results list. Previously loaded scenes are removed rather than being pushed down below the new ones.
- **Scan season** starts a fresh session and progressively collects only the episodes from that run.
- **Load Kent episodes** (and changing the season) removes the previous Script Scan scene list after the new episode index successfully loads.
- A new scan also invalidates any still-running earlier scan to avoid stale episode results reappearing.
- Does not alter the **Zine Library**, saved frames, positive/negative training examples, or the CSV/JSON recovery features from v3.2.

## Install

Replace `app.py`, `requirements.txt`, `railway.json`, and `README.md` in your GitHub repository and redeploy Railway. Confirm `/health` reports `3.3`. Reload the app.

## Check

Use Script Scan: load Kent episodes, test one episode, then test a different episode. The first episode's scenes should disappear and only the newly tested episode's dialogue should show. Saved zine frames must remain unchanged.
