# Hargrove Grant — Metrics Insights (Plain Language)

**Document Version:** 4.3
**Last Updated:** October 29, 2026

This guide explains what the numbers on the Hargrove Grant report are *suggesting* about what’s happening in the community and in service delivery.

It’s written for general readers. It does not assume you know how the dashboard is built.

Hargrove Grant reporting uses only records imported as Clallam County → Port Angeles. Sequim and other agencies do not use Hargrove funds, so their records are never included, and the report, Word export, and saved edits are available only to Port Angeles users and staff administrators. Every Port Angeles overdose referral is included in overdose metrics; there is no map-based service-area filter. Archived 2022-2024 figures are also Port Angeles records.

---

## How to read the report

### Quarter vs (YTD)

- **Quarter metrics** describe what happened during that 3‑month window.
- **(YTD) metrics** describe what has happened so far this year.

YTD will naturally change each quarter. Q4 is the only YTD that represents the full year.

The page and downloaded Word reports use matching, fixed column widths in every
section. Columns show the selected quarter's saved **Value**, each earlier
quarter's raw value (most recent first), and one **Running YTD** result. For Q3:
**ID | Metric | Q3 Value | Q2 Value | Q1 Value | Running YTD | Notes**.
Q1 has no earlier-quarter columns. Totals reset in January and include saved
quarterly edits for the report's agency. Saving a value refreshes prior-quarter
and running results automatically; an explicit warning appears if refreshing fails.
On the page, long Metric labels show the first five words. Expand a label to read
its full definition (or hover over it for the full text). Tables fit their container
and wrap cell contents rather than forcing a minimum width. Word reports retain
the full Metric labels.

Q3 2026 narrative defaults are drawn from Q1–Q3 2026 and same-period 2025 data,
plus the quarter's training, teaching, committee, and outreach activities. Answer 4
(Success Stories) is intentionally blank for the fire chief. Comments (answer 5,
labeled Additional Notes on the page) include a de-identified sepsis case: a
$483,044.73 hospital bill for 27 services during a two-week stay. The bill is
compared with, but not added to, the $661,762 estimated year-to-date savings for
477 patients served (counted once per quarter). It illustrates how much more an
admission can cost than the single ED evaluation the model credits. It does not
establish that this patient's illness was preventable. Existing saved narrative
edits still take precedence over the defaults.

### Cost-avoidance unit values

Avoided ambulance transports are valued at $3,800 and 911 responses without
transport at $1,900. Avoided ED visits are valued at $1,146 through Q1 2026.
Beginning in Q2 2026, they use the Cost Savings page's ED-evaluation average
from Olympic Medical Center, Harborview, and UW Medical Center standard charges
(currently $1,631). The ED value excludes labs, imaging, procedures, medications,
admissions, and readmissions. Each patient whose use declined is credited with
avoiding only one 911 response and/or one ED visit.

Fractional estimates are retained through every percentage in each formula.
Only the final event counts are truncated to whole numbers, and dollar savings
use those final counts. For 152 patients, this gives 45 calls prevented
(152 × 42% × 71% = 45.3264) and 25 transports prevented
(152 × 42% × 71% × 56% = 25.382784). ED visits follow the same rule.
The Cost Savings page uses this calculation on annual patient totals; the
Hargrove report calculates each quarter separately before summing YTD.

These are sums of quarterly counts, not deduplicated annual person counts:
someone served in two quarters can count twice. Dollar amounts are also summed.
Metrics already labeled YTD, prior-year same-period comparisons, and goal statuses
show the selected quarter's result in Running YTD rather than being summed again.
Percentages divide the sum of quarterly numerators by the sum of quarterly
denominators; they are not added or averaged. Averages similarly use pooled counts.
Distinct service types count the union of normalized service names. Service rows
retain their identity when their quarterly ranking changes; services found earlier
in the year remain visible even if they have no referrals in the selected quarter.

When editing a percentage, include its underlying counts, for example
**10% (1 of 10)**. For an average, use **2.00 (10 / 5)**. Raw saved values remain
visible, but running results use the supplied counts. Percentage-only overrides
cannot be pooled reliably. Editing a distinct-service count without its service
names also cannot establish the annual distinct count. Such results are explicitly
marked **Not available** rather than guessed. Missing or nonnumeric quarterly
values are not treated as zero. Blank archived quarters are not assumed to be zero.

### Repeat overdose events: two definitions

- **Same Quarter (grant reporting):** each patient's first overdose in a calendar
  quarter is not a repeat; each additional overdose that quarter is one repeat
  event. Two events for patient A means **one repeat event**, not two repeat
  patients. The denominator is all overdose events that quarter.
- **All-Time History (analysis):** each event after a patient's first recorded
  overdose in this agency's available history is a repeat. Prior quarters and
  prior years count as history. The displayed percentage still describes events
  in the selected reporting period, not a percentage over the patient's lifetime.

For both definitions, Running YTD adds quarterly repeat events and total events
and divides the resulting counts. If Q1 has **1 repeat / 10 events**, and Q2 has
**2 repeats / 20 events**, YTD is **3 / 30 = 10%**. If Q2 instead has **2 / 10**,
YTD is **3 / 20 = 15%**. Averaging quarterly percentages is wrong when the
denominators differ.

The grant calculation resets each calendar quarter. It is **not a rolling 90-day
measure**: one event on March 31 and another on April 1 are not same-quarter
repeats, but the April event is an all-time-history repeat. Records without a
patient ID stay in the total-event denominator but cannot be classified as
repeats. Records without an overdose date cannot be assigned to a reporting
period. Only available records in the report's agency are used.

These event-based rates replace the older patient-based repeat-percentage rows.
Old saved patient-based percentages are not reused as event-based rates; their
definitions and override keys differ. Grant-target rows **2026-03a/b** also count
same-quarter repeat **events**, not distinct patients or patients with multiple
events across YTD. Row 03a uses the selected quarter; 03b uses the same quarter in
the prior year. Their Running YTD columns sum the quarterly repeat-event counts,
with the first event excluded again at each quarter boundary. Row 03c compares
the selected quarter with the prior year's same quarter. New labels prevent
saved overrides for the old YTD patient counts or goal status from carrying over.

Rows **2026-03d/e/f** separately count unduplicated patients. Row 03d counts
patients with an overdose referral in the selected quarter; its Running YTD is
the union of patients across the year-to-date quarters. Row 03e counts patients
with at least two overdose referrals in that quarter; its Running YTD is the
union of those quarter-specific repeat-patient sets. Row 03f counts patients
with an overdose referral this year and at least two recorded overdose referrals
through quarter end, including any history before January 1. Each patient counts
once, and records without a patient ID cannot be counted.

Rows **2026-01a/b** retain percentages with event numerators and denominators;
**2026-02a/b** count fatal OD events in their respective YTD windows. These
metrics and the repeat-event targets use **OD referrals only**. Regular referrals,
including those tagged as overdose-related, do not contribute. Multiple events
for the same patient remain separate events. Existing fatal-disposition and
supportive-referral flag definitions are unchanged.

Supportive-services YTD rates now also use pooled counts, not an average of
quarterly percentages.

### Activity vs reach

- **Activity** (encounters/contacts/referrals) shows how much work happened.
- **Reach** (unique people served) shows how many individuals were touched by that work.

If activity rises while reach stays flat, it often means more repeat follow‑up for the same group of people.

### “Up” and “down” are not automatically good or bad

Some measures are positive when they rise (successful connections to services). Others are concerning when they rise (overdoses).

The goal is to describe direction and meaning: **what moved, what it likely means, and what to watch next.**

---

## Helpful patterns to watch

### 1) Referral flow (received → initiated → resolved)

It helps to think of referrals as a simple flow:

- **Referrals received:** a request for help was recorded.
- **Referrals initiated:** steps were taken to connect someone to a service.
- **Resolution / follow‑up outcomes:** the situation was closed out or moved forward.

If “received” rises but “initiated” does not, common explanations include limited capacity, delays in documentation, or limited availability of outside services.

### 2) Intensity (contacts per person served)

One useful shorthand is:

$$\text{Contacts per person} = \frac{\text{Total Contacts}}{\text{People Served}}$$

When this increases, it usually means more repeat follow‑ups per person. That can reflect complex needs, difficulty accessing services, or more proactive outreach.

### 3) Service gaps (what people request most)

When the same services appear again and again (housing, detox, treatment access, behavioral health, primary care), it is a practical picture of what people are asking for most.

When one service dominates, it often means that need is especially hard to meet locally (long waitlists, limited capacity, or not enough options).

---

## Section-by-section insights

### Services Delivered

#### Total Contacts

What it means:

- Overall activity across encounters and referrals.

What to watch for:

- **Contacts up while people served is flat** → more repeat contact per person.
- **Contacts down while people served is flat** → fewer touches per person (this can be good, or it can reflect delayed documentation).

How to use it:

- Read this alongside “people served” to understand whether change is about *more people* or *more intensity*.

#### Referrals Received

What it means:

- Demand and inbound requests for help.

What to watch for:

- **Sudden spikes** can reflect a community event, a change in referral pathways, or better reporting.
- **Seasonality** is common; comparing the same quarter year‑over‑year is often more meaningful than comparing adjacent quarters.

#### Referrals Initiated

What it means:

- How often a referral moved from “recorded” to “action taken.”

What to watch for:

- **Initiated > received** can happen when older referrals are worked and completed this quarter.
- **Initiated << received** can indicate limited capacity, delays, or limited availability of outside services.

---

### Individuals Served

#### People Served

What it means:

- How many unique individuals had meaningful contact during the quarter.

What to watch for:

- **People served up, contacts flat** → more people reached, but fewer repeat touches per person.
- **People served flat, contacts up** → more repeat follow‑up for the same group.

#### New Enrollments

What it means:

- How many new individuals entered the system during the quarter.

What to watch for:

- **Enrollments up** with stable people served → higher turnover (more new people replacing prior ones).
- **Enrollments down** with higher repeat contact → more time spent on complex follow‑up for an existing group.

---

### Outcomes

#### Total Encounters

What it means:

- How many service contacts occurred.

What to watch for:

- **Encounters up** can mean proactive outreach, or it can mean repeated crisis/stability issues.
- Read alongside “people served” and any high‑utilizer indicators to understand whether activity is concentrated.

#### Overdose-related metrics

What it means:

- Overdose volume and follow‑up activity.

What to watch for:

- Repeat overdoses concentrated in a small group can indicate opportunities for targeted follow‑up.
- Sustained overdoses in a geography or living situation segment can suggest uneven risk across the community.

---

### Geographic Distribution

#### ZIP distribution

What it means:

- Where service demand is coming from geographically.

What to watch for:

- **ZIPs with rising share** even if totals are flat → shifting demand.
- **Small ZIPs with sharp spikes** may reflect a local incident or a reporting change.

---

### Service Gaps (requested services)

This section summarizes which services were requested most often.

Helpful reads:

- **Distinct services requested:** breadth of need.
- **Most common requested services:** demand hotspots.
- **People per service:** helps distinguish “many people need this” vs “a few people need this repeatedly.”

What to watch for:

- **One service dominating:** that need may be especially hard to meet in the current system.
- **Many services requested with small counts:** needs may be diverse and complex.

How to use it:

- Treat the top services as a short list of the most common unmet needs.

---

## Narrative: what good looks like

The narrative is where the numbers are explained in plain language:

- “What changed?” (the main movement)
- “Why did it change?” (the most likely reasons)
- “Why does it matter?” (impact)
- “What happens next?” (next steps)

Good narratives don’t just repeat the table. They explain what changed and why it matters.

---

## When the numbers can look odd

- **Documentation lag:** activity happened, but it was entered later.
- **Category drift:** categories are entered with slightly different names (creating messy buckets).
- **Carryover work:** referrals received in one quarter, then initiated/resolved in another.

If a quarter looks surprising:

1) Check whether it’s a real change or just delayed/messy data entry.
2) If it’s real, use the service-gap section and narrative to explain the most likely reasons.
