import json
from types import SimpleNamespace

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.agent.documents import (
    INVOICE_MODEL,
    INVOICE_PROMPT,
    MAX_DOCUMENT_CHARS,
    DocumentError,
    DocumentStore,
    InvoiceError,
    InvoiceExtraction,
    OpenAIInvoiceModel,
)

REQUEST = httpx2.Request("POST", "http://example.invalid/v1/responses")


def reading(**overrides) -> InvoiceExtraction:
    base = dict(vendor="V", document_date="2026-03-14", total=800.0, currency="EUR", customer_name="Marcus Bell",
                policy_number="HG-2231-88", remarks=None)
    base.update(overrides)
    return InvoiceExtraction(**base)


# --- reading status --------------------------------------------------------------------------------------------

def test_a_reading_with_nothing_to_compare_is_unreadable():
    empty = dict(vendor=None, document_date=None, total=None, currency=None, customer_name=None, policy_number=None, remarks=None)
    assert InvoiceExtraction(**empty).is_unreadable
    assert InvoiceExtraction(**{**empty, "vendor": "Some Shop", "remarks": "hello"}).is_unreadable


@pytest.mark.parametrize("field, value", [("total", 5.0), ("document_date", "2026-01-01"), ("customer_name", "A B")])
def test_one_usable_field_is_enough_to_be_readable(field, value):
    empty = dict(vendor=None, document_date=None, total=None, currency=None, customer_name=None, policy_number=None, remarks=None)
    assert not InvoiceExtraction(**{**empty, field: value}).is_unreadable


# --- the document store ----------------------------------------------------------------------------------------

@pytest.fixture
def folder(tmp_path):
    (tmp_path / "doc_01.txt").write_text("INVOICE for Marcus Bell\n", encoding="utf-8")
    (tmp_path / "doc_02.txt").write_text("x" * (MAX_DOCUMENT_CHARS + 1), encoding="utf-8")
    (tmp_path.parent / "secret.txt").write_text("not a document", encoding="utf-8")
    manifest = {
        "doc_01": {"file": "doc_01.txt"},
        "doc_02": {"file": "doc_02.txt"},
        "doc_03": {"file": "../secret.txt"},       # a manifest entry that points outside the folder
        "doc_04": {"file": "missing.txt"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_a_listed_document_is_loaded(folder):
    assert DocumentStore(folder).load("doc_01") == "INVOICE for Marcus Bell\n"


@pytest.mark.parametrize("bad", ["", "doc_1", "DOC_01", "doc_01.txt", "doc_01/../doc_02", "../secret", "doc_00001", "doc_ 01",
                                 "/etc/passwd", "C:\\secret.txt", "doc_01\n", None, 5])
def test_ids_that_do_not_look_like_document_ids_are_refused_before_touching_the_disk(folder, bad):
    with pytest.raises(DocumentError) as raised:
        DocumentStore(folder).load(bad)
    assert raised.value.kind == "invalid_id"


def test_an_id_that_is_not_in_the_manifest_is_not_found(folder):
    with pytest.raises(DocumentError) as raised:
        DocumentStore(folder).load("doc_99")
    assert raised.value.kind == "not_found"


def test_a_manifest_entry_cannot_lead_outside_the_folder_or_to_a_missing_file(folder):
    for doc_id in ("doc_03", "doc_04"):
        with pytest.raises(DocumentError) as raised:
            DocumentStore(folder).load(doc_id)
        assert raised.value.kind == "not_found"


def test_a_document_over_the_size_limit_is_refused_not_cut(folder):
    with pytest.raises(DocumentError) as raised:
        DocumentStore(folder).load("doc_02")
    assert raised.value.kind == "too_large"


def test_the_real_documents_all_load():
    store = DocumentStore()
    assert len(store.manifest) == 22
    for doc_id in store.manifest:
        assert store.load(doc_id).strip()


# --- the OpenAI reader -----------------------------------------------------------------------------------------

class FakeResponsesAPI:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def parsed_response(parsed, status="completed", output=()):
    return SimpleNamespace(output_parsed=parsed, status=status, output=list(output))


def make_reader(**kwargs):
    api = FakeResponsesAPI(**kwargs)
    return OpenAIInvoiceModel(client=SimpleNamespace(responses=api)), api


def test_the_reader_sends_the_prompt_and_the_document_and_returns_the_parsed_reading():
    expected = reading()
    reader, api = make_reader(response=parsed_response(expected))
    assert reader.extract("INVOICE text") == expected
    assert api.kwargs["model"] == INVOICE_MODEL and api.kwargs["text_format"] is InvoiceExtraction
    assert api.kwargs["temperature"] == 0 and api.kwargs["reasoning"] == {"effort": "none"}
    assert api.kwargs["input"] == [{"role": "system", "content": INVOICE_PROMPT}, {"role": "user", "content": "INVOICE text"}]
    assert reader.prompt_version == "invoice-v1"


@pytest.mark.parametrize("error, kind", [
    (RateLimitError("simulated", response=httpx2.Response(429, request=REQUEST), body=None), "rate_limited"),
    (APITimeoutError(request=REQUEST), "timeout"),
    (APIConnectionError(request=REQUEST), "connection"),
    (APIStatusError("simulated", response=httpx2.Response(401, request=REQUEST), body=None), "api_error"),
])
def test_api_errors_are_mapped_to_invoice_errors(error, kind):
    reader, _ = make_reader(error=error)
    with pytest.raises(InvoiceError) as raised:
        reader.extract("text")
    assert raised.value.kind == kind


def test_a_cut_off_a_refusal_and_a_missing_result_are_errors():
    cases = [
        (parsed_response(None, status="incomplete"), "incomplete"),
        (parsed_response(None, output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])]), "refusal"),
        (parsed_response(None), "no_parsed_output"),
    ]
    for response, kind in cases:
        reader, _ = make_reader(response=response)
        with pytest.raises(InvoiceError) as raised:
            reader.extract("text")
        assert raised.value.kind == kind
