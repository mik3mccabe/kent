# Kent Brockman Frame Finder v3.4 — Stay on the results page when saving

Builds on v3.3, retaining speaker-attributed Script Scan, Dialogue Search, Visual Scan, manual frame adjustment, CSV/JSON restore and backup, and Zine Library.

## What changed

- **Save to zine** now leaves you on the results page and displays **Added to Zine Library** briefly in the bottom-right.
- **Save adjusted frame to zine** now closes the timestamp editor and displays the same confirmation without opening Zine Library.
- Saving the same already-saved frame a second time reports **Already saved to Zine Library** rather than accidentally removing it.
- Editing a frame from inside Zine Library keeps you in Zine Library and reports **Zine frame updated**.
- The application page title, heading and `/health` now all report version 3.4.

The `kent-v2-labels` localStorage key is unchanged. Existing saved frames, notes and training labels are not intentionally modified by this update.

## Deployment

Replace `app.py`, `requirements.txt`, `railway.json`, and `README.md` in your GitHub repository and let Railway redeploy. Check `/health` for version `3.4`. If you have saved zine frames, download a full library backup (JSON) before upgrading.
