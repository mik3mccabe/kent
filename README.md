# Kent Brockman Frame Finder

A small visual research tool for locating Simpsons scenes through Frinkiac dialogue search and inspecting nearby clean frames.

## Deploy to Railway

1. Create a new GitHub repository and upload these files.
2. In Railway choose **New Project → Deploy from GitHub repo**.
3. Select the repository and deploy.
4. In the Railway service go to **Settings → Networking → Generate Domain**.
5. Open the generated public URL.

No database or API key is required by this app. `FRINKIAC_BASE` defaults to `https://frinkiac.com` and can be overridden as an environment variable.

## Use

Search for dialogue likely to occur in a Kent scene, for example `Kent Brockman`, `Channel 6`, `Smartline`, `Eye on Springfield`, or a known line. The page displays clean frames at one-second offsets around each matching dialogue hit.

Endpoints:

- `/` browser UI
- `/api/search?q=...` normalized Frinkiac search results
- `/api/nearby?episode=S06E18&timestamp=123456` raw nearby-frame data
- `/export.csv?q=...` CSV export
- `/health` Railway health check

## Important limitation

Frinkiac searches subtitle dialogue, not pixels. A title card whose text is never spoken will not be found by searching its visual text. Use dialogue to locate Kent broadcasts, then inspect surrounding frames visually.

## v1.1 fix

- Accepts Frinkiac search results where `Episode` is a string as well as an object.
- Labels the numeric search control as "Seconds either side". A value of 3 samples from 3 seconds before through 3 seconds after each dialogue hit.
