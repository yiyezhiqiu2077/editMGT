import gzip
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


spec = importlib.util.spec_from_file_location(
    "raw_schema_audit", Path(__file__).resolve().parents[1] / "scripts/data/audit_raw_dataset_schema.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_gzip_metadata_counts_real_rows_and_booleans(tmp_path):
    path = tmp_path / "train-00000.jsonl.gz"
    rows = [{"edit_type": "Add", "better_data": True}, {"edit_type": "Remove", "better_data": False}]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row) for row in rows))
    result = audit.audit_json([path])
    assert result["row_count"] == 2
    assert result["categorical_counts"]["better_data"] == {"true": 1, "false": 1}
    assert result["schema_variants"] == [{"columns": ["better_data", "edit_type"], "rows": 2}]


def test_parquet_schema_includes_observed_taxonomy_and_nulls(tmp_path):
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"final_task": ["replace", "add", "replace"], "mask__qc_flag": ["OK", None, "OK"]}), path)
    result = audit.audit_parquet([path])
    assert result["row_count"] == 3
    assert result["categorical_counts"]["final_task"] == {'"replace"': 2, '"add"': 1}
    assert result["null_counts"]["mask__qc_flag"] == 1
