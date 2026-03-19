You are helping me audit TMLR (Transactions on Machine Learning Research) decision timelines using the OpenReview API.

## Background

TMLR's Action Editor guidelines state:
- Reviewers submit recommendations "no later than 4 weeks after the submission of the third review"
- Action Editors should submit decisions "within 1 week" after receiving recommendations
- Therefore: "decision proposal should therefore be submitted within 5 weeks of the beginning of the discussion"

I want to verify empirically whether decisions are posted within 5 weeks (35 days) of the third review.

## Task

Write and execute a Python script that:

### Part 1: Scrape OpenReview

1. Use the `openreview` package to connect to the API (no credentials needed for public data):
   ```python
   import openreview
   client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
   ```

2. Fetch all TMLR submissions:
   ```python
   submissions = client.get_all_notes(invitation='TMLR/-/Submission', details='replies')
   ```

3. For each submission, extract from `note.details['replies']`:
   - `t_reviews_public`: When reviews became visible (look for invitations containing 'Review')
   - `t_decision`: When decision was posted (look for invitations containing 'Decision')
   
   Each reply has `reply['invitation']` (string) and `reply['cdate']` (timestamp in ms).

4. Compute for each submission:
   - `gap_days_reviews_public_to_decision`: (t_decision - t_reviews_public) / (1000 * 60 * 60 * 24)
   - `censored`: True if no decision exists yet

### Part 2: Analyze

1. Filter to uncensored submissions only
2. Compute and print:
   - N (sample size)
   - Median, 75th, 90th, 95th, 99th percentiles of `gap_days_after_author_window`
   - Share exceeding 28 days (TMLR's stated maximum)
   - Share exceeding 35 days
   - Share exceeding 42 days

### Output format

```
=== TMLR Audit Results ===
N (uncensored): <N>

Quantiles (days after author window):
  Median: <X>
  75th:   <X>
  90th:   <X>
  95th:   <X>
  99th:   <X>

Compliance with stated "no later than 4 weeks":
  Share > 28 days: <X>% (violates stated max)
  Share > 35 days: <X>%
  Share > 42 days: <X>%
```

### Constraints

- Single file, <100 lines
- Dependencies: `openreview-py`, `pandas`
- Handle missing data gracefully (some submissions may lack reviews or decisions)
- Use the earliest review timestamp and earliest decision timestamp for each submission

### Key insight to verify

TMLR's email to authors states: "In about 2 weeks (Jan 04), and no later than 4 weeks (Jan 18), reviewers will submit their formal decision recommendation."

The script tests whether this "no later than 4 weeks" claim holds empirically.