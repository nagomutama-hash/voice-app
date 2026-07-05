# voice-app-v2 Screenshot Operation Manual

This folder stores daily GA4 and Meta ad screenshots for voice-app-v2.

Do not mix Project A / voice-app screenshots here.

## Folder Rule

Save screenshots by date.

```text
reports/YYYY-MM-DD/
```

Examples:

```text
reports/2026-07-05/
reports/2026-07-06/
reports/2026-07-07/
```

If today's folder does not exist, create it before saving screenshots.

## Daily Start Command

When starting daily ad and measurement work, tell Codex:

```text
Codex今日の運用開始
```

Codex should then:

1. Check whether today's folder exists under `reports/`.
2. Create `reports/YYYY-MM-DD/` if it does not exist.
3. Show today's save destination.
4. Show the screenshot checklist.
5. Wait for screenshots to be saved.

Today's save destination format:

```text
C:\Users\climb\Desktop\voice-app-v2\reports\YYYY-MM-DD
```

Daily screenshot checklist:

```text
【GA4】
□ イベント一覧
□ コンバージョン
□ ファネル（設定済みの場合）

【Meta広告】
□ キャンペーン
□ 広告セット
□ 広告
```

After saving screenshots, tell Codex:

```text
今日のスクショを入れたのでお願いします
```

Then Codex should read screenshots, update KPI and Notion, compare with the
previous report date, and create a short analysis report.

## Recommended File Names

Exact names are helpful but not required.

```text
ga4_events.png
ga4_conversions.png
ga4_funnel.png
meta_campaign.png
meta_adset.png
meta_ads.png
```

Codex should still recognize flexible names when they include words such as:

```text
ga4
event
conversion
funnel
meta
campaign
adset
ads
```

## Daily GA4 Screenshots

### 1. Events

Capture the GA4 event list so these events are visible when possible:

```text
page_view
diagnosis_start
recording_complete
rerecord_click
rerecord_complete
line_register_click
```

Check:

```text
event count
change from previous period
```

### 2. Conversions

Capture this screen if conversions are configured.

Check:

```text
conversion count
change from previous period
```

### 3. Funnel

Capture this screen after the funnel is configured.

Funnel steps:

```text
LP view
diagnosis start
diagnosis complete
LINE click
```

Check:

```text
step counts
drop-off rate
change from previous period
```

## Daily Meta Screenshots

### 1. Campaigns

Check:

```text
amount spent
impressions
CTR
CPC
CPM
```

### 2. Ad Sets

Check:

```text
delivery status
learning status
conversions
```

### 3. Ads

Check:

```text
CTR
CPC
conversions
Frequency, if visible
```

## Codex Processing Flow

After screenshots are saved, ask Codex to process today's report folder.

Codex should:

1. Read numbers from screenshots.
2. Compare with the previous report date.
3. Update the Notion KPI database.
4. Update the Notion Dashboard notes.
5. Add a Codex implementation history entry.
6. Report a short analysis.

## Daily Analysis Report Format

Codex should report the following every time.

### Today's Changes

```text
CTR
CPC
diagnosis start rate
diagnosis completion rate
LINE click rate
```

### Notable Points

Summarize anything worth noticing from the previous comparison.

Examples:

```text
CTR improved but CPC also rose.
diagnosis start rate dropped, so the first view or CTA may need review.
LINE click rate improved after the result section change.
```

### Improvement Candidates

Suggest up to 3 high-priority ideas.

Examples:

```text
1. Improve first-view CTA copy.
2. Simplify recording instructions.
3. Strengthen the LINE benefit copy after results.
```

## KPI Mapping

GA4:

```text
page_view -> LP views
diagnosis_start -> diagnosis starts
recording_complete -> diagnosis completions
rerecord_click -> rerecord clicks
rerecord_complete -> rerecord completions
line_register_click -> LINE clicks
```

Meta:

```text
PageView -> LP views
Lead -> diagnosis starts
CompleteRegistration -> diagnosis completions
Contact -> LINE clicks
```

Manual or future measurement:

```text
LINE registrations
consultation reservations
```

## Future API Migration

This screenshot workflow is temporary and should be easy to replace later.

When GA4 API or Meta API is introduced, keep the same Notion KPI fields and
replace only the data source.

Data source options:

```text
screenshot
GA4 API
Meta API
manual input
```

## Git Rule

Screenshot image files under `reports/` are ignored by Git.

Text operation files such as this README should remain managed by Git.
