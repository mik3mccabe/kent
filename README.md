# Kent Brockman Frame Finder v1.2

A small Flask app for reviewing real Frinkiac frames around dialogue search hits and saving useful frames for a Kent Brockman title-card zine.

## v1.2 fixes

- Uses Frinkiac's real `/api/frames/{episode}/{timestamp}/{before}/{after}` frame endpoint instead of inventing one-second timestamps. This removes the broken-image problem from v1.1.
- Groups frames by dialogue search hit.
- Shows dialogue context when available.
- Adds a configurable maximum scene count.
- Adds **Save to zine** using browser local storage.
- Exports saved frames to CSV with episode, exact timestamp, timecode, clean image URL, dialogue, and blank cataloguing fields.
- `/health` reports version 1.2.

## Deploy on Railway

Replace the files in your existing GitHub repository with these files and commit. Railway should redeploy automatically.

After deployment, visit `/health`. It should return `{"ok":true,"version":"1.2"}`.

## Suggested use

Search for distinctive spoken lines from Kent broadcasts rather than just a character name. Frinkiac searches subtitle dialogue, not visual content. Start with an 8 to 15 second window and inspect the resulting real frames. Save any title card or over-the-shoulder graphic you want to catalogue.

Saved selections live in that browser's local storage. Export CSV periodically so you have a backup.
