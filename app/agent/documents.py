"""Supporting documents: finding them on disk safely, and reading the facts out of one.

The language model only extracts. Comparing the facts with the claim is plain code in compare.py.
A document's text is untrusted input: this prompt, like the Phase 2 and 3 ones, has no defence against text that talks
to the model. That comes later, so the before-and-after stays clean.
"""
import json
import re
from pathlib import Path
from typing import Literal, Protocol

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_DOCUMENT_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"
DOCUMENT_ID_PATTERN = re.compile(r"^doc_\d{2,4}$")
MAX_DOCUMENT_CHARS = 20_000

INVOICE_MODEL = "gpt-6-luna"
INVOICE_PROMPT_VERSION = "invoice-v1"

INVOICE_PROMPT = """You extract structured data from a supporting document for an insurance claim: a repair invoice, a receipt or a payment receipt.
The text may be in English or Spanish. If a field is not clearly stated, return null. Never guess.
If the text is unreadable or contains none of these fields, return null for every field.

- vendor: the name of the business that issued the document.
- document_date: the date of the document, as YYYY-MM-DD. A numeric date such as 03/04/2026 is day first (3 April 2026). Write it as YYYY-MM-DD whatever format the document uses.
- total: the final total of the document (total due, total paid, TOTAL), as a plain number. In English text 1,450.00 is 1450. In Spanish text 1.450,00 is 1450. Not a subtotal and not a single line item.
- currency: EUR, USD or GBP when the document names the currency by code, word or symbol. Any other named currency is "other". If none is named, null.
- customer_name: the person the document is made out to, or who paid, as written.
- policy_number: the insurance policy number, exactly as written.
- remarks: the text of the remarks or notes line, as written, or null if there is none.
"""


class InvoiceExtraction(BaseModel):
    vendor: str | None = Field(description="Name of the business that issued the document, or null.")
    document_date: str | None = Field(description="Date of the document as YYYY-MM-DD, or null.")
    total: float | None = Field(description="Final total as a plain number, or null.")
    currency: Literal["EUR", "USD", "GBP", "other"] | None = Field(description="Currency of the total, or null if not stated.")
    customer_name: str | None = Field(description="Person the document is made out to, as written, or null.")
    policy_number: str | None = Field(description="Insurance policy number exactly as written, or null.")
    remarks: str | None = Field(description="Text of the remarks or notes line as written, or null.")

    @property
    def is_unreadable(self) -> bool:
        """Nothing a comparison could use. A document with some fields missing is still readable."""
        return self.total is None and self.document_date is None and self.customer_name is None


class DocumentError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind          # invalid_id, not_found or too_large


class DocumentStore:
    """Documents are found through the manifest, never by building a path from what a caller supplied."""

    def __init__(self, directory: Path = DEFAULT_DOCUMENT_DIR):
        self.directory = directory.resolve()
        self._manifest: dict | None = None

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        return self._manifest

    def load(self, document_id: str) -> str:
        if not isinstance(document_id, str) or not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            raise DocumentError("invalid_id", "Document IDs look like doc_01.")
        entry = self.manifest.get(document_id)
        if entry is None:
            raise DocumentError("not_found", "No such document.")
        path = (self.directory / entry["file"]).resolve()
        if not path.is_relative_to(self.directory) or not path.is_file():
            raise DocumentError("not_found", "No such document.")
        text = path.read_text(encoding="utf-8")
        if len(text) > MAX_DOCUMENT_CHARS:
            raise DocumentError("too_large", "The document is too long to read.")
        return text

    def exists(self, document_id: str) -> bool:
        """True if the ID names a document on disk (a document that is too long still exists)."""
        try:
            self.load(document_id)
        except DocumentError as e:
            return e.kind == "too_large"
        return True


class InvoiceError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class InvoiceModel(Protocol):
    model: str
    prompt_version: str

    def extract(self, document_text: str) -> InvoiceExtraction:
        ...


class OpenAIInvoiceModel:
    def __init__(self, model: str = INVOICE_MODEL, client: OpenAI | None = None):
        self.model = model
        self.prompt_version = INVOICE_PROMPT_VERSION
        self._client = client if client is not None else OpenAI(timeout=30.0, max_retries=2)

    def extract(self, document_text: str) -> InvoiceExtraction:
        try:
            response = self._client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": INVOICE_PROMPT},
                    {"role": "user", "content": document_text},
                ],
                text_format=InvoiceExtraction,
                temperature=0,
                reasoning={"effort": "none"},
            )
        except RateLimitError as e:
            raise InvoiceError("rate_limited", "Rate limit hit after retries") from e
        except APITimeoutError as e:
            raise InvoiceError("timeout", "Request timed out after retries") from e
        except APIConnectionError as e:
            raise InvoiceError("connection", "Could not reach the API") from e
        except APIStatusError as e:
            raise InvoiceError("api_error", f"API returned status {e.status_code}") from e

        if response.status == "incomplete":
            raise InvoiceError("incomplete", "Response was cut off")
        for item in response.output:
            if item.type == "message" and any(part.type == "refusal" for part in item.content):
                raise InvoiceError("refusal", "Model refused")
        if response.output_parsed is None:
            raise InvoiceError("no_parsed_output", "Model returned no parsed result")
        return response.output_parsed
