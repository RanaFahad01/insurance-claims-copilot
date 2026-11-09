"""The JSONL trace file: what it holds, what it must not hold, and which run IDs may reach the disk."""
import json

import pytest

from app.agent.trace import TraceWriter, Trace, new_run_id, trace_lines, valid_run_id

RUN_ID = "0123456789abcdef"


def sample_trace() -> Trace:
    from app.agent.trace import TurnRecord
    trace = Trace()
    trace.step = 1
    trace.turns.append(TurnRecord(step=1, input_tokens=100, output_tokens=20, latency_ms=900, tool_calls=2))
    trace.record(tool="compare_documents", arguments='{"document_id": "doc_14"}', outcome="ok", result='{"agrees_with_claim": true}',
                 latency_ms=1200, side_effect=False)
    trace.record(tool="get_clause", arguments='{"clause_id": "en-C2.1"}', outcome="ok", result="{}", latency_ms=5, side_effect=False)
    trace.step = 2
    trace.turns.append(TurnRecord(step=2, input_tokens=300, output_tokens=40, latency_ms=700, tool_calls=0))
    trace.step = 0
    trace.record(tool="escalate_to_human", arguments="{}", outcome="ok", result='{"created": true}', latency_ms=3, side_effect=True)
    return trace


SUMMARY = {"status": "completed", "final_route": "escalate"}


# ---- run IDs

def test_new_run_ids_are_valid_and_different():
    ids = {new_run_id() for _ in range(200)}
    assert len(ids) == 200 and all(valid_run_id(i) for i in ids)


@pytest.mark.parametrize("bad", [
    "", "0123456789abcde", "0123456789abcdef0", "0123456789ABCDEF", "0123456789abcdeg", "../0123456789abc", "..\\0123456789ab",
    "0123456789abcdef\n", " 0123456789abcdef", "0123456789abcdef.jsonl", "0123/56789abcdef", None, 12345678901234567, b"0123456789abcdef",
])
def test_only_sixteen_lowercase_hex_characters_are_a_run_id(bad):
    assert valid_run_id(bad) is False


def test_the_good_id_is_valid():
    assert valid_run_id(RUN_ID)


# ---- the lines

def test_lines_are_in_the_order_things_happened_and_numbered():
    lines = trace_lines(RUN_ID, sample_trace(), SUMMARY)
    assert [(l["type"], l.get("step"), l.get("tool")) for l in lines] == [
        ("turn", 1, None), ("tool_call", 1, "compare_documents"), ("tool_call", 1, "get_clause"),
        ("turn", 2, None), ("tool_call", 0, "escalate_to_human"), ("run", None, None),
    ]
    assert [l["seq"] for l in lines] == list(range(6)) and {l["run_id"] for l in lines} == {RUN_ID}


def test_a_turn_line_has_the_tokens_and_latency_and_a_call_line_has_arguments_result_and_latency():
    turn, call = trace_lines(RUN_ID, sample_trace(), SUMMARY)[:2]
    assert (turn["input_tokens"], turn["output_tokens"], turn["latency_ms"], turn["tool_calls"]) == (100, 20, 900, 2)
    assert call["arguments"] == '{"document_id": "doc_14"}' and call["result"] == '{"agrees_with_claim": true}'
    assert call["latency_ms"] == 1200 and call["outcome"] == "ok" and call["side_effect"] is False


def test_the_last_line_is_the_run_summary():
    assert trace_lines(RUN_ID, sample_trace(), SUMMARY)[-1] == {"run_id": RUN_ID, "seq": 5, "type": "run", **SUMMARY}


def test_a_run_with_no_agent_still_has_its_summary_and_its_escalation_call():
    trace = Trace()
    trace.record(tool="escalate_to_human", arguments="{}", outcome="ok", result="{}", latency_ms=1, side_effect=True)
    assert [l["type"] for l in trace_lines(RUN_ID, trace, SUMMARY)] == ["tool_call", "run"]
    assert [l["type"] for l in trace_lines(RUN_ID, Trace(), SUMMARY)] == ["run"]


def test_long_arguments_and_results_are_clipped():
    trace = Trace()
    trace.record(tool="t", arguments="a" * 5000, outcome="ok", result="r" * 5000, latency_ms=1, side_effect=False)
    call = trace_lines(RUN_ID, trace, SUMMARY)[0]
    assert len(call["arguments"]) < 600 and len(call["result"]) < 600


# ---- the file

def test_the_file_has_one_valid_json_object_per_line_and_reads_back_the_same(tmp_path):
    writer = TraceWriter(tmp_path / "traces")
    path = writer.write(RUN_ID, sample_trace(), SUMMARY)
    assert path == tmp_path / "traces" / f"{RUN_ID}.jsonl"
    raw = path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 6 and all(isinstance(json.loads(line), dict) for line in raw)
    assert writer.read(RUN_ID) == trace_lines(RUN_ID, sample_trace(), SUMMARY)


def test_the_folder_is_created_on_first_write_and_no_temporary_file_is_left(tmp_path):
    folder = tmp_path / "a" / "b"
    writer = TraceWriter(folder)
    assert not folder.exists()
    writer.write(RUN_ID, Trace(), SUMMARY)
    assert [p.name for p in folder.iterdir()] == [f"{RUN_ID}.jsonl"]


def test_each_run_gets_its_own_file(tmp_path):
    writer = TraceWriter(tmp_path)
    writer.write("aaaaaaaaaaaaaaaa", sample_trace(), SUMMARY)
    writer.write("bbbbbbbbbbbbbbbb", Trace(), SUMMARY)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["aaaaaaaaaaaaaaaa.jsonl", "bbbbbbbbbbbbbbbb.jsonl"]
    assert len(writer.read("aaaaaaaaaaaaaaaa")) == 6 and len(writer.read("bbbbbbbbbbbbbbbb")) == 1


def test_non_ascii_text_is_kept_as_written(tmp_path):
    trace = Trace()
    trace.record(tool="t", arguments="Peña, Müller", outcome="ok", result="€ 800", latency_ms=1, side_effect=False)
    writer = TraceWriter(tmp_path)
    writer.write(RUN_ID, trace, SUMMARY)
    assert "Peña, Müller" in (tmp_path / f"{RUN_ID}.jsonl").read_text(encoding="utf-8")


def test_an_invalid_id_never_reaches_the_disk(tmp_path):
    writer = TraceWriter(tmp_path / "traces")
    for bad in ["../x", "x", "0123456789ABCDEF", "0123456789abcdef\n"]:
        with pytest.raises(ValueError):
            writer.write(bad, Trace(), SUMMARY)
        with pytest.raises(ValueError):
            writer.path_for(bad)
    assert not (tmp_path / "traces").exists() and list(tmp_path.iterdir()) == []


def test_reading_an_invalid_or_unknown_id_gives_none_and_never_looks_outside_the_folder(tmp_path):
    (tmp_path / "secret.jsonl").write_text('{"type": "run", "secret": true}\n', encoding="utf-8")
    writer = TraceWriter(tmp_path / "traces")
    writer.write(RUN_ID, Trace(), SUMMARY)
    for bad in ["../secret", "..\\secret", "../secret.jsonl", "secret", "0123456789abcdef\n"]:
        assert writer.read(bad) is None
    assert writer.read("ffffffffffffffff") is None


def test_a_corrupt_file_raises_instead_of_returning_half_a_trace(tmp_path):
    writer = TraceWriter(tmp_path)
    (tmp_path / f"{RUN_ID}.jsonl").write_text('{"type": "run"}\n{not json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        writer.read(RUN_ID)
