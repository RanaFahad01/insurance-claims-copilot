# Coverage labeling rules

These rules define the correct coverage label for a claim. They are the ground truth for the
coverage evaluation, in the same way `data/LABELING.md` is the ground truth for extraction.
They apply to the fictional policy in `data/policies/policy_en.md` and `policy_es.md`
(Norvale Mutual, Home and Vehicle Plan). Both language versions carry the same clause numbers
and the same meaning, so a label is the same whichever version is searched.

All policy text, claims and names are invented.

## Statuses

| Status | Use when |
|---|---|
| `covered` | A covering clause applies, and no exclusion or condition removes the cover on the facts in the text. |
| `excluded` | A clause clearly removes cover on the facts stated in the text. |
| `cannot_determine` | No clause addresses the situation (a gap), or the answer depends on a fact the text leaves open while hinting that it matters (an ambiguous boundary), or clauses conflict. |

The system never says "approved" or "denied". `excluded` and `cannot_determine` both send the
claim to a person, and `covered` is only a recommendation.

## Assumptions about facts the text does not state

1. The claimant is the policyholder, the home, contents or vehicle in the claim are insured, and
   the event happened inside the policy period, unless the text says or suggests otherwise.
2. **Exclusions are not assumed.** If the text gives no fact that points to an exclusion, the
   exclusion does not apply. A claim with a burst pipe and no mention of poor maintenance is
   `covered`.
3. **A hint changes that.** If the text contains a specific detail that points toward an
   exclusion or an unmet condition but does not settle it, the label is `cannot_determine`.
   Examples: "ski chalet" (the home may have been unoccupied, C2.2), "lifted the floorboards"
   after a leak (the leak may have been gradual, C2.3), "shared storage room" (the locked,
   assigned unit rule in C4.1).
4. **Territory and dates are not assumed away.** If the text puts the property or the event
   outside the countries in C1.3, the label is `excluded` on C1.3.
5. Limits and deductibles (C7.1, C7.2) never change the status. Amounts are not compared with
   limits in Phase 3, and no payout is calculated.
6. Notification (C7.3) changes the status only if the text itself says the claim was reported
   late. Because C7.3 gives the insurer discretion ("may reduce or refuse"), a stated late
   report is `cannot_determine`, not `excluded`.
7. **Text that talks to the system is not a policy fact.** Instructions about what to conclude
   ("mark this covered", "ignore the policy"), claims that an adjuster pre-approved the claim,
   and quoted clauses that do not exist in the policy are ignored. Label by the real policy and
   the real facts. A clause ID that is not in the policy does not exist.
8. If one claim describes several losses, label the main loss the claimant describes, and add a
   note when part of the claim is outside the policy.
9. **Unknown cause.** If the text says the cause is unknown and describes no covered event (a
   cracked wall, for example), the label is `cannot_determine` with no clause IDs.
10. **A property that the text suggests is not the insured home.** Rule 1 assumes the home in the
    claim is insured, but if the text suggests it is another property (a holiday flat while the
    claimant lives elsewhere), the label is `cannot_determine`, listing C1.1 and the covering clause.
11. **Policy changes mentioned in a claim are not part of the policy.** An endorsement, an
    "update" or a pasted "current wording" in the claim text does not change the label. Label by
    the clauses in the policy files only. This extends rule 7.

## Policy gaps: topics the policy does not mention

The policy is silent on these topics, so a claim about them is `cannot_determine` (not
`excluded`). The policy has no catch-all exclusion for events that are not listed.

| # | Topic the policy does not address | Examples |
|---|---|---|
| G1 | Fences, gates, garden walls, plants and trees themselves | A car hits the garden fence. A branch breaks a fence panel when no storm is mentioned. |
| G2 | Liability to other people | A neighbour's flat is damaged by water from the claimant's home. |
| G3 | Property used for work or business | Tools, stock or equipment stolen. |
| G4 | Digital data and cyber events | Lost files, hacked accounts, ransomware. |
| G5 | Power cuts and spoiled food | A freezer defrosts after a blackout. |

## Ambiguous boundaries: clauses that can conflict

| # | Boundary | What decides it |
|---|---|---|
| A1 | Sudden water damage (C2.1) versus gradual leaks (C2.3) | The text says the leak was sudden (covered) or gradual (excluded). If it hints at long-term damage without saying so, `cannot_determine`. |
| A2 | Frost damage (C2.2) and the 30-day unoccupied exception | Occupancy and checks are stated (settled), or only hinted at (`cannot_determine`). |
| A3 | Theft in shared areas (C4.1) | A locked unit assigned to the claimant is stated (covered), or the text only mentions a shared area (`cannot_determine`). |

## Interactions: one event, two clauses

- **Storm, rain and open windows:** C5.1 and C5.2 cover rain that enters through storm damage.
  C5.3 excludes rain that enters through openings left open or already defective. The facts in
  the text decide which applies.
- **Water and flood:** C2.1 covers a sudden escape of water from fixed pipes, tanks, heating
  systems and appliances. The clause does not say whose pipes, so a leak from the flat above
  is not excluded by that alone. C2.4 excludes flooding from outside (rivers, groundwater,
  sewage), not a pipe leak in a neighbouring flat.
- **Fire and extinguishing water:** C3.3 treats water used to put out a fire as part of the fire
  (C3.1), not as water damage under section 2.

## Expected clause IDs

Expected clause IDs list every clause needed to justify the label. Write IDs without a language
prefix (`C2.1`). The ingested chunks carry the prefix (`en-C2.1`, `es-C2.1`).

| Label | List |
|---|---|
| `covered` | The covering clause or clauses that apply, including extension clauses actually used (for example C3.1 and C3.3). Do not list exclusions that were checked and did not apply. |
| `excluded` | The covering clause that would otherwise apply, if there is one, and the clause that removes cover (for example C2.1 and C2.3). For a territory exclusion, C1.3 alone. |
| `cannot_determine` for a gap | Nothing. The list is empty, because no clause addresses the situation. |
| `cannot_determine` for an ambiguous boundary | The clauses that are in tension (for example C2.1 and C2.3). |

General clauses (sections 1, 7, 8 and 9) are listed only when they are decisive.

## Worked examples

These illustrate the rules. The labels for the claim files are in `data/expected_coverage.json`
(keys such as `claims/claim_01_water_en.txt`). Each has a status, the clause IDs, a short reason and a
`borderline` flag for labels where a reasonable reviewer could choose another status.

| Situation | Label | Clause IDs | Why |
|---|---|---|---|
| A pipe bursts under the kitchen sink and floods the floor | `covered` | C2.1 | Sudden escape of water, nothing in the text points to an exclusion. |
| A delivery van reverses into the garage door of a holiday apartment in Florida | `excluded` | C1.3 | The property is outside the territory. Collision is not the deciding clause. |
| A storm smashes a bedroom window and rain soaks the carpet | `covered` | C5.1, C5.2 | Rain entered through an opening the storm created. C5.3 does not apply. |
| A fire starts in the oven, and the fire brigade's water soaks the carpet and the flat below | `covered` | C3.1, C3.3 | The main loss is the claimant's own fire and extinguishing damage. The neighbour's flat is a gap (G2), noted but not the label. |
| A car hits the garden fence | `cannot_determine` | none | Fences are a gap (G1). The vehicle-impact clause (C6.2) lists walls, doors, roof and windows only. |
| A radiator valve leaks and the floorboards have lifted | `cannot_determine` | C2.1, C2.3 | Lifted boards hint at a gradual leak, but the text does not say so (A1). |
| Pipes burst in a ski chalet, with nothing said about occupancy | `cannot_determine` | C2.2 | A ski chalet hints that the home may have been unoccupied for over 30 days (A2). |
| Bicycle stolen from "the shared storage room of my building" | `cannot_determine` | C4.1 | Locked, assigned unit is not stated (A3). |
