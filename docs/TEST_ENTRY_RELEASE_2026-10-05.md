# October test entry release

User authorized proceeding with the proposed new entry and its publication on 2026-10-05.

- Entry: `https://voice-app-v2.onrender.com/test-202610`
- Existing Render service and `voice-app-v2` branch only; no plan or original-service changes.
- Old root, `/?preview=five`, speed-retest, direct analysis/advice/config and static URLs remain closed.
- New entry uses only its own static/config/analysis/help paths, activates five-metric mode without a query parameter, and rejects legacy analysis mode.
- New entry closes on 2026-11-01 00:00 JST (2026-10-31 15:00 UTC), matching the LP's deadline.
- New pages display “声診断アプリ Ver.2.0”. Search indexing and caching are disabled.
- Includes the previously approved total-80 / all-five-above-12 expressive criteria, flexible near band and short three-step comments.
- Runtime code and relevant tests only are published; private QA recordings, local reports, source knowledge drafts, account settings and the separate LP checkout are excluded.

Validation before publication: 35 Python tests, 111 JavaScript tests, inline-script syntax, namespace links in the browser. Two simultaneous generated-audio requests to localhost succeeded (about 0.61 seconds each); this is a transport/analysis check and not a real-voice or phone-device validation.

After deployment: check old entry closure, new assets/config, generated-audio analysis and browser readiness; then configure and republish the LP's app button. Real iPhone/Safari and Android/Chrome microphone, recording and playback checks remain part of the small participant test.
