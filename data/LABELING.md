# Labeling rules

These rules define the correct value for every extracted field. They are the ground truth for
`data/expected_extractions.json`, they feed the system prompt in `app/extractor.py`, and they are
the scoring reference in Phase 5.

Rules marked **[guide]** come from the Phase 2 guide. Rules marked **[added]** settle ambiguities
the guide leaves open. When two rules seem to conflict, the more specific rule wins. When a value
cannot be determined from the text alone, the answer is `null`. Never guess.

All claims in `data/` are invented. No real people, policy numbers or insurer wording.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `incident_date` | `YYYY-MM-DD` or `null` | The date the incident happened |
| `claimant_name` | string or `null` | The person filing the claim |
| `policy_number` | string or `null` | The policy number as written |
| `damage_type` | `water`, `fire`, `theft`, `collision`, `other`, `unclear` | What kind of damage |
| `estimated_amount` | number or `null` | The total amount claimed |
| `currency` | `EUR`, `USD`, `GBP`, `other` or `null` | Currency of the amount |
| `language` | `en`, `es`, `other` | Language of the claim text |
| `summary` | string | One neutral sentence (not scored by exact match) |

## Dates (`incident_date`)

1. **[guide]** Year not stated in the text -> `null`. "3 de marzo" is `null`, even if the
   current year seems obvious.
2. **[guide]** Relative date ("last Tuesday", "el martes pasado", "yesterday") -> `null`.
3. **[added]** A full date with day, month and year is written as `YYYY-MM-DD`, whatever format
   the text uses ("12 March 2026", "12/03/2026", "2026-03-12", "12 de marzo de 2026").
4. **[added]** A numeric date where day and month could be swapped is read day first
   (`03/04/2026` is 3 April 2026), following the European convention.
5. **[added]** A month and year without a day ("in March 2026") -> `null`. A date range
   ("between 10 and 12 March 2026") -> `null`.
6. **[added]** Only the date of the incident counts. The date the message was written, sent or
   signed is not the incident date.
7. **[added]** Two different dates given for the same incident with no way to tell which is
   right -> `null`.
8. **[added]** Several separate incidents in one text (`damage_type` is `unclear`) -> `null`,
   because there is no single incident date.

## Amounts (`estimated_amount`)

1. **[guide]** Several amounts -> use the total claimed. If it is unclear which is the total,
   `null`.
2. **[guide]** Amount written in words -> convert to a number ("dos mil quinientos euros" is
   `2500.0`).
3. **[added]** An estimate and a final figure for the same repair: the final, invoiced or actual
   figure is the total claimed. If nothing marks one as final, `null`.
4. **[added]** Parts of a claim with no stated total ("part one EUR 990, part two EUR 900") ->
   `null`. Extraction does not add figures up.
5. **[added]** An approximate amount ("unos 800 euros", "about EUR 800", "around 800") -> the
   number given (`800.0`). A range ("between 500 and 700") -> `null`.
6. **[added]** The amount is the amount claimed for the loss. Deductibles, policy limits,
   premiums and amounts a third party paid are not the claimed amount.
7. **[added]** Number formats: in English text `4,200` is `4200.0` and `4.5` is `4.5`. In
   Spanish text `2.500` is `2500.0` and `2.500,50` is `2500.5`.
8. **[added]** No amount stated -> `null`. An amount explicitly stated as zero -> `0.0`.

## Currency (`currency`)

1. **[guide]** Currency not stated and no symbol -> `null`, even when the language or country
   suggests one. A Spanish text with "800" and no currency word is `null`.
2. **[added]** Code, word or symbol maps to the code: "EUR", "euro(s)" or `€` is `EUR`;
   "USD", "dollar(s)", "dólar(es)" or `$` is `USD`; "GBP", "pound(s)", "libras (esterlinas)" or
   `£` is `GBP`.
3. **[added]** Any other named currency (CHF, "pesos", "kronor") -> `other`.
4. **[added]** If `estimated_amount` is `null`, `currency` is `null`.

## People and policy

1. **[guide]** The claimant is the person filing the claim, not witnesses, neighbours, tradespeople,
   adjusters or other third parties.
2. **[added]** Claimant name as written, without titles ("Mr. Marcus Bell" is `Marcus Bell`).
   A name in an email signature counts if the sender is the person filing the claim.
3. **[added]** `policy_number` exactly as written, with the same letters, digits and hyphens.
   Claim references, ticket numbers and phone numbers are not policy numbers. Not stated ->
   `null`.

## Damage type (`damage_type`)

| Value | Use when |
|---|---|
| `water` | Burst pipe, leak, flooding, water coming in |
| `fire` | Fire, smoke or heat damage |
| `theft` | Theft, burglary, robbery |
| `collision` | A vehicle or object hit something |
| `other` | A clear incident of a different kind (storm, vandalism, broken window) |
| `unclear` | The type cannot be determined from the text, or several incidents are described |

1. **[guide]** Several separate incidents in one text -> `unclear`.
2. **[guide]** Type cannot be determined -> `unclear`. This is different from `other`, which is
   used when the type is known but is not water, fire, theft or collision.
3. **[added]** One incident with knock-on damage (a fire, then water from putting it out) is one
   incident, labeled by its cause (`fire`).

## Language and summary

1. **[added]** `language` is the language of the claim text. In a mixed-language text, use the
   language of the claimant's own account.
2. **[added]** `summary` is one neutral sentence in English. It is not compared by exact match.

## Noise in the text

1. **[added]** Email signatures, disclaimers, phone numbers and addresses are not claim facts.
   The only thing taken from a signature is the claimant's name (People and policy, rule 2).
2. **[added]** Quoted reply history (lines starting with `>` or "On ... wrote:") is earlier
   correspondence. Values that appear only there are not used.

## Sync with the system prompt

The system prompt in `app/extractor.py` must carry every rule here that changes an output, so the
model and the ground truth follow the same rules. Rules not yet in the guide's starter prompt
(amount rules 3 to 8, date rules 3 to 8, currency rules 2 to 4, and most of the damage-type table)
get added in Session 3.
