# Kent Brockman Frame Finder v2.0

Two connected discovery methods for the Kent Brockman title-card zine:

1. Dialogue Search: search Frinkiac subtitles and inspect real nearby frames.
2. Visual Scan: use frames you label in Dialogue Search as positive and negative examples, then rank sampled S01-S10 frames by visual similarity.

## Training labels

- Good example: useful example of Kent/news/title-card visual language.
- Not relevant: negative training example.
- Save to zine: final selection. It is also treated as a strong positive example.

Labels and zine selections are stored in your browser localStorage, so they survive refreshes on the same browser/device.

## Visual Scan

Choose a season, sampling interval and result count. The server samples each episode through Frinkiac's real-frame endpoint, extracts lightweight visual features, and ranks frames against your labelled examples.

This is a lightweight similarity classifier, not face recognition. It works best after 20+ varied positive examples and 20+ negatives. Include Kent at the desk, Smartline, Eye on Springfield, Channel 6 graphics, and other layouts you want the zine to find.

## Deploy

Replace the existing repository files with these files and commit. Railway should redeploy automatically.

Health check: `/health` should report version `2.0`.
