# Kent Brockman Frame Finder v3.2 — Safe restore of previously saved zine frames

## Restore from the older CSV without losing new saves

1. Install the four root files (`app.py`, `requirements.txt`, `railway.json`, `README.md`) and redeploy Railway.
2. Open **Zine Library** on the same site and browser profile used to save your current frames.
3. First click **Download full library backup (JSON)** to preserve your current entries and training labels.
4. Click **Restore saved frames from CSV**, choose the earlier `kent-brockman-zine (1).csv`, and approve the merge.
5. The CSV importer accepts the old 7-column exports and newer exports with timestamps and subtitles. Existing saved frames and manually edited titles/notes take priority over duplicates; no frames or labels are cleared.
6. Confirm the counter and download another full library backup (JSON).

The user's supplied backup contains 47 distinct frames. If the current two saved frames are different, the restored library should contain 49 frames.

## Changes

- Adds CSV recovery in the Zine Library without needing a network connection or Frinkiac lookup.
- Adds full browser label/training-data backup to JSON and merge-style JSON restore.
- Preserves current browser state in the `kent-v32-before-restore` localStorage key before any import.
- Improves the Clear all labels warning.
- Does not change underlying persistent storage key (`kent-v2-labels`) or v3.1's timestamp adjustment, Script Scan, Dialogue Search, Visual Scan, or Frinkiac links.

## Deployment check

`/health` reports 3.2. Browser page heading displays v3.2.

## Limitations

Existing browser data depends on visiting the same exact origin and profile; this cannot repair cleared localStorage without an exported backup. CSV imports restore zine entries, not positive/negative visual training labels. Full JSON backup restores training labels too.
