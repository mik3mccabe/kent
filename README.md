# Kent Brockman Frame Finder v2.8

Deploy the four root files to your GitHub/Railway repository. Check `/health` for version 2.8.

## Speaker-attributed Script Scan

This release replaces Springfield! Springfield! transcript scraping with a character-labelled Simpsons World-derived dataset hosted at https://github.com/sujanjoejacob/Text-mining-with-Simpsons-Data . It filters by Kent Brockman's character ID 211, groups nearby lines into scenes, searches Frinkiac for exact spoken dialogue, and retrieves nearby frames. It reports unmatched dialogue instead of inventing frames.

The three dataset files are downloaded by the Railway server on first use and cached in `/tmp/kent-brockman-data` (ephemeral across deploys). No transcript is bundled in this ZIP. Internet access from Railway to raw.githubusercontent.com is required. Check `/api/script-dataset-status` for source and dataset counts; it returns an explicit error if download fails. First load can be slow; retry after initialization if a request times out. Dataset availability, Frinkiac matching and production deployment have not been live-tested here.

The Script Scan browser cache key is new for v2.8, so previously completed but empty v2.7 scans will not suppress new results. Existing saved zine frames and training labels use their original browser storage keys.

Note: dataset timestamps can differ from Frinkiac timestamps; only Frinkiac-matched frames are shown, and unmatched speaker-attributed dialogue remains listed for review. Script Scan does not guarantee all visual title cards are found.
