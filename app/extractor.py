from dotenv import load_dotenv
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

from .schemas import ClaimExtraction

load_dotenv()

MODEL = "gpt-6-luna"
client = OpenAI(timeout=30.0, max_retries=3)

SYSTEM_PROMPT = """You extract structured data from insurance claim texts.
The text may be in English, Spanish or another language.
If a field is not clearly stated in the text, return null. Never guess.

Dates (incident_date, format YYYY-MM-DD):
- Only a full date with day, month and year counts. Write it as YYYY-MM-DD whatever format the text uses.
- A numeric date such as 03/04/2026 is read day first (3 April 2026).
- If the year is not stated, incident_date is null.
- Relative dates ("last Tuesday", "yesterday") are null.
- A month and year without a day, or a date range, is null.
- Only the date of the incident counts, not the date the message was written, sent or signed.
- If two different dates are given for the same incident and it is not clear which is right, null.
- If the text describes several separate incidents, incident_date is null.

Amounts (estimated_amount, a plain number):
- Use the total amount claimed for the loss.
- If several amounts appear and a total is stated, use the total. If it is unclear which one is the total, null.
- If an estimate and a final, invoiced or actual figure are given for the same repair, use the final figure. If nothing marks one as final, null.
- If the text gives parts of a claim with no stated total, null. Do not add figures up.
- An approximate amount ("about 800", "unos 800") is that number. A range ("between 500 and 700") is null.
- Deductibles, excesses, policy limits and premiums are not the claimed amount.
- Convert amounts written in words to numbers.
- In English text 4,200 is 4200 and 4.5 is 4.5. In Spanish text 2.500 is 2500 and 2.500,50 is 2500.5.
- No amount stated is null. An amount stated as zero is 0.

Currency:
- Use EUR, USD or GBP when the text names the currency by code, word or symbol (EUR/euros/€, USD/dollars/$, GBP/pounds/£).
- Any other named currency is "other".
- If no currency is stated and there is no symbol, null, even if the language suggests one.
- If estimated_amount is null, currency is null.

People and policy:
- The claimant is the person filing the claim, not witnesses, neighbours, tradespeople, adjusters or other third parties.
- Give the claimant's name as written, without titles such as Mr. or Dña.
- policy_number is exactly as written. Claim references, ticket numbers and phone numbers are not policy numbers.

Damage type:
- water: burst pipe, leak, flooding, water coming in.
- fire: fire, smoke or heat damage.
- theft: theft, burglary, robbery.
- collision: a vehicle or object hit something.
- other: a clear incident of a different kind, such as a storm or vandalism.
- unclear: the type cannot be determined from the text, or the text describes several separate incidents.
- One incident with knock-on damage (a fire, then water from putting it out) is labeled by its cause.

Language and summary:
- language is the language of the claim text: "en", "es", or "other". In a mixed-language text, use the language of the claimant's own account.
- summary is one neutral sentence in English.

Noise:
- Email signatures, disclaimers, addresses and phone numbers are not claim facts. The only thing taken from a signature is the claimant's name.
- Quoted reply history (lines starting with > or "On ... wrote:") is earlier correspondence. Values that appear only there are not used.
"""


class ExtractionError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def extract_claim(raw_text: str) -> ClaimExtraction:
    try:
        response = client.responses.parse(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            text_format=ClaimExtraction,
            temperature=0,
            reasoning={"effort": "none"},
        )
    except RateLimitError as e:
        raise ExtractionError("rate_limited", "Rate limit hit after retries") from e
    except APITimeoutError as e:
        raise ExtractionError("timeout", "Request timed out after retries") from e
    except APIConnectionError as e:
        raise ExtractionError("connection", "Could not reach the API") from e
    except APIStatusError as e:
        raise ExtractionError("api_error", f"API returned status {e.status_code}") from e

    if response.status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else "unknown"
        raise ExtractionError("incomplete", f"Response was cut off: {reason}")

    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "refusal":
                    raise ExtractionError("refusal", f"Model refused: {part.refusal}")

    if response.output_parsed is None:
        raise ExtractionError("no_parsed_output", "Model returned no parsed result")

    return response.output_parsed
