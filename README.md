# Kent Brockman Frame Finder v3.0

## Fix: Kent's dialogue no longer depends on Frinkiac search

Script Scan now loads speaker-attributed **spoken_words** from the Simpsons dialogue dataset, filtering the character table for the **exact** normalized name `kent brockman` (ID **211**) and Season 1–10 episode IDs.

The v2.9 scan often showed no useful results because it attempted slow, overly exact Frinkiac searches **before returning the actual dialogue**. Also, "Test one episode" always used E01, even when Kent was not in that episode.

v3.0 separates these operations:

1. In **Kent dialogue**, choose a season, click **Load Kent episodes**. The episode dropdown shows only episodes in that season with Kent-attributed lines, plus the count of lines. Source and dataset coverage appear under the toolbar.
2. Click **Test selected episode**, or **Scan season** to collect all available Kent-attributed lines. Results show all original lines with episode ID and the original dataset time, whether or not any subtitles or screenshots match. Completed episode results are saved to browser storage.
3. Click **Find matching Frinkiac frames** for one scene, or **Match all to Frinkiac**. Each scene has its own short HTTP request. Only links that match Frinkiac subtitle text are called verified. Unmatched dialogue stays visible; the approximate dataset time is linked separately and labelled unverified.
4. Click **Export Kent dialogue CSV** for all collected speaker-attributed lines. This is independent of the Zine Library CSV.

Each screenshot retains the **View on Frinkiac** link, and the existing Zine Library, dialogue search, visual scan and exports are retained.

## Deploy to Railway

Replace **app.py**, **requirements.txt**, **railway.json**, and **README.md** in your GitHub repo, commit, and redeploy.

Health: `/health` must report `"version":"3.0"`.

The dataset is loaded from:
https://github.com/sujanjoejacob/Text-mining-with-Simpsons-Data

The first dataset load on a new Railway instance downloads the CSV files from GitHub; if Railway has network or timeout issues, the toolbar reports an error rather than silently displaying zero Kent lines.

## Browser data

The existing zine and training labels live under `kent-v2-labels` and are **not modified**. v3.0 uses a new `kent-v30-script` progress key so old v2.8/2.9 scan results cannot be mistaken for verified dialogue.

## Validation and limitations

- Python source compiles and browser JavaScript passes Node syntax validation.
- Mocked backend tests verify that Kent-attributed lines are returned even when Frinkiac yields no matches, and that image matching is handled separately.
- This environment cannot test Railway's live dataset download or Frinkiac's live responses. Therefore the real availability and match yield remain to be checked on your deployment.
- Character-attributed records are not a perfect list of every Kent appearance. Silent appearances, omitted speaker IDs, or an occasional incorrect label remain possible. Keep Visual Scan as a secondary method.
- A matching subtitle does not, by itself, prove a Kent news graphic is on screen. Visually verify proposed captures before putting them in the zine.
