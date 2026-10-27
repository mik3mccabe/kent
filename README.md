# Kent Brockman Frame Finder v3.1

## New: adjust the screenshot timestamp before saving to the zine

Each screenshot in **Dialogue Search**, **Script Scan**, and **Visual Scan** now has **Adjust timestamp / Add to zine** beneath it. A **Choose timestamp / Add to zine** button also appears beside each Script Scan scene, even when no dialogue match has been verified. Previously saved entries in **Zine Library** have an **Adjust timestamp** button.

1. Open **Adjust timestamp / Add to zine** beside a candidate screenshot.
2. Nudge with **−5s / −1s / −0.1s / +0.1s / +1s / +5s**, or type a time such as `08:24.500`.
3. You can also paste a Frinkiac page URL such as `https://frinkiac.com/caption/S03E09/172000` (must be the same episode).
4. Click **Preview frame**. The server finds the nearest *existing* Frinkiac screenshot, displays that image, and reports the difference between the requested and actual screenshot timestamps. It does not fabricate a frame at an arbitrary timestamp.
5. Click **Save adjusted frame to zine**. This saves the **actual previewed frame's** episode, timestamp, timecode, and Frinkiac image URL. The modal does not allow saving without a successful preview.

You can reopen an existing saved zine frame, adjust it, and save. Its title, notes and category are retained; the old frame is replaced rather than duplicated. Existing browser-local zine records remain compatible without migration.

**CSV exports** (standalone export and `index.csv` in the Zine ZIP) include:
- `original_timestamp` and `original_timecode` (the original matched/reference frame)
- `adjusted_timestamp` and `adjusted_timecode` (actual selected Frinkiac screenshot)
- `requested_timestamp` and `requested_timecode` (what you typed before snapping to the nearest image)
- `frinkiac_url` (page link to the final screenshot)
- the existing episode, timestamp, title, category, notes, subtitle_text, image fields

For older saved frames, the original/requested timestamp defaults to the saved timestamp. Subtitle text is still fetched at **the actual adjusted timestamp** during export.

## Deploy

Replace the following four root-level files in your GitHub repository and redeploy to Railway:
- `app.py`
- `requirements.txt`
- `railway.json`
- `README.md`

Check `/health`: `version` should now be `3.1`. Refresh the page to see the updated header. **Don't clear site data or click Clear all labels**: training examples and saved zine frames use the existing `kent-v2-labels` localStorage key, and Script Scan uses the existing `kent-v30-script` key.

## Implementation and testing

Backend `GET /api/frame-at-time?episode=S03E09&timestamp=172000` uses Frinkiac `/api/frames/{episode}/{time}/{before}/{after}`, first searching ±2.5 seconds and then ±8 seconds if necessary, and returns the closest real indexed screenshot. If no suitable frame is available, it fails visibly rather than saving a guessed image. The Frinkiac page link updates to that exact frame.

Validated: Python syntax, JavaScript syntax, mocked backend nearest-frame/error tests, and mocked browser tests of time parsing, Frinkiac URL pasting, previewing, preservation of existing zine details, and CSV fields. **Live Railway and Frinkiac connectivity were not tested in this environment**.
