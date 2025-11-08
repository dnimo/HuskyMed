import json
from datasets import Dataset

from dataset_builder import data_pipeline as dp


def _make_sample_patient():
    return {
        "smrs": [
            {
                "INPUT_DATE": "2024-01-10",
                "EMR_TYPE": "退院時サマリ",
                "SECTION": "内科",
                "smr": [
                    {"EMR_DATA_TYPE": "治療経過", "EMR_TEXT": "治療経過の要約"},
                    {"EMR_DATA_TYPE": "S", "EMR_TEXT": "主観的情報"},
                ],
            }
        ],
        "records": [
            {
                "ts": "20240109",
                "section": "内科",
                "emr": [
                    {"EMR_DATA_TYPE": "治療経過", "EMR_TEXT": "治療記録"},
                    {"EMR_DATA_TYPE": "S", "EMR_TEXT": "患者訴え"},
                ],
            }
        ],
    }


def test_build_instruction_records(monkeypatch, tmp_path):
    sample = _make_sample_patient()

    def fake_load_dataset(name, data_files, split, cache_dir=None):
        return Dataset.from_list([sample])

    monkeypatch.setattr(dp, "load_dataset", fake_load_dataset)

    records = dp.build_instruction_records(["dummy.json"])
    assert len(records) == 1
    merged = records[0].merged
    assert ("治療経過", "治療記録", "治療経過の要約") in merged
    assert any(item[1] == "患者訴え" for item in merged)

    paths = dp.write_instruction_chunks(records, tmp_path.as_posix(), chunk_size=1, prefix="chunk")
    assert len(paths) == 1
    with open(paths[0], "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload[0]["section"] == "内科"
    assert payload[0]["smr_type"] == "退院時サマリ"
    assert payload[0]["merged"] == merged
