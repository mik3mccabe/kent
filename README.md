# Kent Brockman Frame Finder v2.4

Railway-ready Flask app for finding and cataloguing Kent Brockman news graphics from Frinkiac.

## v2.4

- Visual Scan now runs one episode per HTTP request, avoiding long Railway/browser request cutoffs.
- Progress is saved in browser localStorage after every completed episode.
- Resume Scan continues a stopped, refreshed, or interrupted season.
- Failed episodes do not discard completed work and are retried when you resume.
- Ranked candidates update after each episode.
- Diagnostic mode still scans one episode.
- Dialogue Search, training labels, Zine Library, and ZIP download remain available.

Deploy by replacing the files in the existing GitHub repository. Railway should redeploy automatically. Check `/health` for version 2.4.
