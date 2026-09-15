# Kent Brockman Frame Finder v2.1

Dialogue search and trainable visual scan for cataloguing Kent Brockman news graphics from Frinkiac.

## v2.1 changes
- Defaults to 100 raw subtitle matches, with optional Scan all matches.
- Merges subtitle hits from the same episode when they occur within 30 seconds, so one broadcast scene appears once.
- Deduplicates nearby frames.
- Adds Mark scene reviewed and Hide reviewed scenes.
- Adds Expand ±30 sec for a wider contact sheet.
- Adds News, no graphic as a separate training label.
- Keeps Good example, Not relevant and Save to zine.
- Keeps the visual season scan and CSV export from v2.0.

## Deploy
Replace the files in your existing GitHub repository and commit. Railway should redeploy automatically. Visit `/health` and confirm version `2.1`.

## Suggested workflow
Start with Dialogue Search. Use 100 raw matches or Scan all matches. Review merged scenes, label representative frames, then use Visual Scan after you have a varied positive set. Use News, no graphic for genuine broadcast frames that should not be treated as title-card positives.
