# Kent Brockman Frame Finder v2.2

Adds a visual Zine Library to v2.1.

## New in v2.2
- Saved zine frames are shown as a dedicated gallery, sorted by episode and timestamp.
- Edit exact title-card wording, category, and notes for each saved frame.
- Download an individual clean JPG.
- Download all saved clean JPGs as a ZIP.
- ZIP is organised into episode folders and includes `index.csv` metadata.
- Existing Dialogue Search, scene deduplication, training labels, and Visual Scan remain.

## Deploy on Railway
Replace the files in your existing GitHub repository and commit. Railway should redeploy automatically.

Check `/health`. It should report version `2.2`.

## Storage note
Labels and zine metadata are stored in your browser localStorage. Use the same browser/profile. The ZIP is generated on demand from your saved browser library and Frinkiac images.
