#!/usr/bin/env python3
"""Stream canonical pools from an explicitly audited schema mapping."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.explicit_region.canonical import (
    SCHEMA_VERSION, image_from_locator, read_locator_bytes, sample_uid_for, sha256_bytes,
    validate_record,
)
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dataset import _align_to_mask_coordinates
from src.explicit_region.deterministic import canonical_json
from src.explicit_region.geometry import SampleRejected, apply_geometry, sample_geometry
from src.explicit_region.language import contains_han


def locator(mapping, row, file_relative, row_index, role):
    spec = mapping["fields"][role]; storage = spec["storage"]
    if storage == "embedded_parquet":
        return {"backend": "parquet", "file": file_relative, "row_index": row_index, "column": spec["column"]}
    if storage == "file":
        return {"backend": "file", "relative_path": str(row[spec["column"]])}
    if storage == "bbox":
        return {"backend": "derived_bbox", "bbox": list(row[spec["column"]]),
                "reference_width": int(row[spec["width_column"]]), "reference_height": int(row[spec["height_column"]])}
    raise ValueError(f"unsupported explicit storage mapping: {storage}")


def empty_region_status_requirements(empty_policy):
    if not empty_policy:
        return []
    requirements = empty_policy.get("status_requirements")
    if requirements is None:
        return [{"column": empty_policy["status_column"], "values": empty_policy["status_values"]}]
    return requirements


def mapped_columns(mapping):
    fields = mapping["fields"]
    columns = {fields["instruction"], fields["edit_type"]}
    for role in ("source", "target", "region"):
        spec = fields[role]; columns.add(spec["column"])
        if spec["storage"] == "bbox":
            columns.update((spec["width_column"], spec["height_column"]))
    if fields.get("group_id"):
        columns.add(fields["group_id"])
    columns.update(fields.get("group_id_fields") or mapping.get("group_id_fields") or [])
    if mapping.get("better_data_column"):
        columns.add(mapping["better_data_column"])
    empty_policy = mapping.get("empty_region_policy")
    if empty_policy:
        columns.update(empty_policy["zero_columns"])
        columns.update(requirement["column"] for requirement in empty_region_status_requirements(empty_policy))
        columns.update(empty_policy.get("empty_list_columns") or [])
    return sorted(columns)


def iter_rows(root, mapping):
    files = sorted(root.glob(mapping["files_glob"])); fmt = mapping["format"]
    if not files:
        raise RuntimeError("schema mapping glob matched no files")
    columns = mapped_columns(mapping)
    for file in files:
        relative = file.relative_to(root).as_posix()
        if fmt == "parquet":
            import pyarrow.parquet as pq
            with pq.ParquetFile(file, memory_map=False, pre_buffer=False) as parquet:
                missing = set(columns) - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(f"mapped columns absent from {relative}: {sorted(missing)}")
                offset = 0
                for batch in parquet.iter_batches(batch_size=16, columns=columns, use_threads=False):
                    # Python conversion is bounded to 16 relevant-column rows,
                    # never the entire image table or unrelated QC/image columns.
                    for index, row in enumerate(batch.to_pylist(), offset):
                        yield relative, index, row
                    offset += batch.num_rows
        elif fmt == "jsonl":
            with file.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if line.strip():
                        yield relative, index, json.loads(line)
        else:
            raise ValueError("mapping format must be parquet or jsonl")


def validate_mapping(mapping, root):
    if mapping.get("schema_audit_status") != "REAL_SCHEMA_AUDITED":
        raise RuntimeError("mapping must bind a REAL_SCHEMA_AUDITED report")
    name, revision = mapping["dataset_name"], mapping["dataset_revision"]
    if name not in ("crispedit", "scaleedit", "interedit"):
        raise ValueError("schema-mapped builder is for audited non-MagicBrush pools")
    if not isinstance(revision, str) or not revision or "REPLACE" in revision:
        raise ValueError("immutable dataset revision is required")
    binding = mapping.get("schema_audit", {})
    audit_path = Path(binding.get("path", ""))
    if not audit_path.is_absolute() or not audit_path.is_file():
        raise ValueError("mapping requires schema_audit.path to an absolute existing report")
    if sha256_file(audit_path) != binding.get("sha256"):
        raise ValueError("schema audit report hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("status") != "REAL_SCHEMA_AUDITED" or audit.get("dataset_name") != name or
            audit.get("dataset_revision") != revision or binding.get("dataset_revision") != revision or
            Path(audit.get("root", "")).resolve() != root.resolve()):
        raise ValueError("schema audit dataset/revision/root binding mismatch")
    semantics = mapping.get("mask_semantics")
    if not isinstance(semantics, str) or not semantics.strip() or any(
            word in semantics.upper() for word in ("REPLACE", "NOT_RUN", "UNKNOWN")):
        raise ValueError("explicit audited mask_semantics required")
    empty_policy = mapping.get("empty_region_policy")
    if empty_policy is not None:
        requirements = empty_region_status_requirements(empty_policy)
        if (not isinstance(empty_policy, dict) or
                empty_policy.get("action") != "reject_annotated_empty_bytes" or
                not isinstance(empty_policy.get("zero_columns"), list) or
                not empty_policy["zero_columns"] or
                not all(isinstance(x, str) and x for x in empty_policy["zero_columns"]) or
                not isinstance(requirements, list) or not requirements or
                not all(isinstance(requirement, dict) for requirement in requirements) or
                not all(isinstance(requirement.get("column"), str) and requirement["column"] for requirement in requirements) or
                not all(isinstance(requirement.get("values"), list) and requirement["values"] for requirement in requirements) or
                not isinstance(empty_policy.get("empty_list_columns", []), list) or
                not all(isinstance(x, str) and x for x in empty_policy.get("empty_list_columns", []))):
            raise ValueError("invalid explicit empty_region_policy")
    fields = mapping["fields"]
    composite = fields.get("group_id_fields") or mapping.get("group_id_fields")
    if composite and (not isinstance(composite, list) or not all(isinstance(x, str) and x for x in composite)):
        raise ValueError("group_id_fields must be a list of mapped column names")
    if fields.get("group_id") and composite:
        raise ValueError("use either group_id or group_id_fields, not both")
    return audit


def raw_asset_bytes(mapping, raw, loc, root, role):
    if mapping["fields"][role]["storage"] != "embedded_parquet":
        return read_locator_bytes(loc, root)
    # The projected Parquet batch has already supplied exactly the bytes the
    # canonical locator would read. Avoid a second Parquet pass for each row.
    value = raw[mapping["fields"][role]["column"]]
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, bytes):
        raise ValueError(f"embedded {role} image cell must contain bytes")
    return value


def build_pool(root, mapping_path, type_mapping_path, output):
    root, output = Path(root).resolve(), Path(output)
    mapping = yaml.safe_load(Path(mapping_path).read_text(encoding="utf-8"))
    validate_mapping(mapping, root)
    types = yaml.safe_load(Path(type_mapping_path).read_text(encoding="utf-8"))["datasets"]
    name, revision = mapping["dataset_name"], mapping["dataset_revision"]
    if not types.get(name):
        raise RuntimeError(f"formal edit-type mapping for {name} is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    report_path = output.with_suffix(".report.json")
    reject_path = output.with_suffix(".rejections.jsonl")
    counts, rejected = Counter(), Counter()
    report = {"schema": "fixed200k-canonical-pool-audit-v1", "status": "RUNNING", "dataset_name": name,
              "dataset_revision": revision, "schema_audit": mapping["schema_audit"],
              "mapping_sha256": sha256_file(mapping_path), "edit_type_mapping_sha256": sha256_file(type_mapping_path),
              "format": mapping["format"], "batch_rows": 16, "mapped_columns": mapped_columns(mapping)}

    def save_report():
        report["counts"] = dict(counts); report["rejection_counts"] = dict(rejected)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    save_report()
    try:
        with temporary.open("w", encoding="utf-8") as writer, reject_path.open("w", encoding="utf-8") as rejection_writer:
            for file_relative, index, raw in iter_rows(root, mapping):
                counts["input_rows"] += 1
                if counts["input_rows"] % 1000 == 0:
                    save_report()
                    print(json.dumps({"event": "pool_progress", "dataset_name": name,
                                      "counts": dict(counts), "rejection_counts": dict(rejected)}), flush=True)
                better = mapping.get("better_data_column")
                if better and not bool(raw.get(better)):
                    counts["excluded_not_better_data"] += 1
                    continue
                original_type = str(raw[mapping["fields"]["edit_type"]])
                if original_type not in types[name]:
                    raise RuntimeError(f"unmapped {name} edit type: {original_type}")
                locators = {role: locator(mapping, raw, file_relative, index, role)
                            for role in ("source", "target", "region")}
                values = {role: raw_asset_bytes(mapping, raw, loc, root, role) for role, loc in locators.items()}
                hashes = {role: sha256_bytes(value) for role, value in values.items()}
                empty_policy = mapping.get("empty_region_policy")
                if values["region"] == b"" and empty_policy:
                    # Some audited releases encode an absent annotation as b'',
                    # not a PNG. Reject before freeze; never fabricate a mask.
                    evidence = {column: raw[column] for column in empty_policy["zero_columns"]}
                    status_evidence = {requirement["column"]: raw[requirement["column"]]
                                       for requirement in empty_region_status_requirements(empty_policy)}
                    list_evidence = {column: raw[column] for column in empty_policy.get("empty_list_columns", [])}
                    if any(type(value) not in (int, float) or value != 0 for value in evidence.values()):
                        raise ValueError("empty region bytes disagree with audited annotation metadata")
                    if any(status_evidence[requirement["column"]] not in requirement["values"]
                           for requirement in empty_region_status_requirements(empty_policy)):
                        raise ValueError("empty region bytes disagree with audited annotation metadata")
                    if any(not isinstance(value, list) or value for value in list_evidence.values()):
                        raise ValueError("empty region bytes disagree with audited annotation metadata")
                    for role in ("source", "target"):
                        with image_from_locator(locators[role], root, payload=values[role]):
                            pass  # Source/target corruption is still a hard error.
                    rejected["annotated_empty_region"] += 1
                    rejection_writer.write(json.dumps({"file": file_relative, "row_index": index,
                        "reason": "annotated_empty_region", "asset_hashes": hashes,
                        "annotation": evidence, "annotation_status": status_evidence,
                        "annotation_lists": list_evidence}, sort_keys=True) + "\n")
                    continue
                images = {}
                for role, loc in locators.items():
                    try:
                        images[role] = image_from_locator(loc, root, payload=values[role])
                    except Exception as exc:
                        report["failed_asset"] = {"file": file_relative, "row_index": index,
                                                  "role": role, "bytes": len(values[role]),
                                                  "sha256": hashes[role]}
                        for image in images.values():
                            image.close()
                        raise
                counts["decoded_rows"] += 1
                fraction = float((np.asarray(images["region"].convert("L")) > 0).mean())
                original_instruction = raw[mapping["fields"]["instruction"]]
                instruction = "" if original_instruction is None else str(original_instruction).strip()
                group_fields = mapping["fields"].get("group_id_fields") or mapping.get("group_id_fields")
                group_column = mapping["fields"].get("group_id")
                if group_fields:
                    group = canonical_json([[field, raw[field]] for field in group_fields])
                else:
                    group = str(raw[group_column]) if group_column else hashes["source"]
                record = {
                    "schema_version": SCHEMA_VERSION, "sample_uid": "0" * 64, "manifest_index": None,
                    "dataset_name": name, "dataset_revision": revision, "group_id": group,
                    **{f"{role}_locator": loc for role, loc in locators.items()},
                    **{f"{role}_sha256": digest for role, digest in hashes.items()},
                    "instruction_original": instruction, "instruction_en": "" if contains_han(instruction) else instruction,
                    "language_original": "zho_Hans" if contains_han(instruction) else "en",
                    "edit_type_original": original_type, "edit_type_canonical": types[name][original_type],
                    "mask_semantics": mapping["mask_semantics"], "region_fraction": fraction,
                    "translation_status": "pending" if contains_han(instruction) else "passthrough_en",
                    "translation_cache_key": None,
                }
                record["sample_uid"] = sample_uid_for(record)
                reason = "empty_instruction" if not instruction else "empty_region" if fraction == 0 else None
                if reason is None:
                    try:
                        aligned_source, aligned_target, _ = _align_to_mask_coordinates(
                            images["source"], images["target"], images["region"], record["sample_uid"])
                        if aligned_source is not images["source"]:
                            aligned_source.close()
                        if aligned_target is not images["target"]:
                            aligned_target.close()
                        geometry = sample_geometry(images["region"], resolution=1024, base_seed=42,
                            global_sample_index=index, sample_key=record["sample_uid"], epoch=0,
                            max_resample_attempts=8, minimum_mask_retention=0.75, deterministic_fallback=True)
                        if not (np.asarray(apply_geometry(images["region"], geometry, is_mask=True)) > 0).any():
                            reason = "post_geometry_empty_region"
                    except SampleRejected as exc:
                        reason = exc.reason
                    except ValueError as exc:
                        if "unaligned aspect ratio" not in str(exc):
                            raise
                        reason = "unaligned_aspect_ratio"
                for image in images.values():
                    image.close()
                if reason:
                    rejected[reason] += 1
                    rejection_writer.write(json.dumps({"file": file_relative, "row_index": index,
                        "sample_uid": record["sample_uid"], "reason": reason}, sort_keys=True) + "\n")
                    continue
                validate_record(record)
                writer.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                counts["eligible_rows"] += 1
        report.update(status="PASS", output_sha256=sha256_file(temporary),
                      rejections_sha256=sha256_file(reject_path))
        temporary.replace(output)
        save_report()
    except Exception as exc:
        report.update(status="FAIL", failure=f"{type(exc).__name__}: {exc}")
        save_report(); temporary.unlink(missing_ok=True)
        raise
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--edit-type-mapping", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_pool(args.root, args.mapping, args.edit_type_mapping, args.output), indent=2))


if __name__ == "__main__":
    main()
