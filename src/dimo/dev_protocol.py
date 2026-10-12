"""Content-bound native DEV128 protocol and read-only historical result reuse."""
import json
from pathlib import Path
from PIL import Image
from src.explicit_region.contracts import sha256_file

COMMON = {"resolution": 1024, "generation_seeds": [0, 1], "parameters": "fp32",
          "forward": "bf16_autocast", "vq": "fp32", "attention_backend": "math_sdp",
          "tf32": False, "reference_strength": 1., "timestep": "roi_relative",
          "temperature": 1., "top_k": 0, "top_p": 0.,
          "token_mask": {"mode": "any_overlap", "coverage_threshold": .5,
                         "dilation_tokens": 0, "minimum_edit_tokens": 1}}
STUDENT_PROTOCOL = {**COMMON, "cfg": 1., "steps": 1, "r_init": 1., "sigma": .3}
TEACHER_PROTOCOL = {**COMMON, "cfg": 5., "steps": 12}
ROI_CROP_RULE = "native mask bbox + max(8px, round(0.1 * longest bbox side)), clipped to image"


def rows_at(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_native_manifest(path, digest, *, count=128):
    if sha256_file(path) != digest:
        raise RuntimeError("DIMO_DEV_MANIFEST_HASH_MISMATCH")
    rows = rows_at(path)
    if len(rows) != count or len({r["sample_uid"] for r in rows}) != count:
        raise RuntimeError("DIMO_DEV_MANIFEST_COUNT_OR_UID_MISMATCH")
    for row in rows:
        if "test" in str(row.get("split", "")).lower() or not row.get("group_id") or not row.get("cluster_id"):
            raise RuntimeError("DIMO_DEV_SPLIT_OR_GROUP_IDENTITY_MISMATCH")
        sizes = []
        for key, hash_key in (("source", "source_sha256"), ("target", "target_sha256"), ("mask", "region_sha256")):
            if sha256_file(row[key]) != row[hash_key]:
                raise RuntimeError("DIMO_DEV_SOURCE_TARGET_MASK_HASH_MISMATCH")
            with Image.open(row[key]) as img:
                sizes.append(img.size)
                if key == "mask" and img.convert("L").getbbox() is None:
                    raise RuntimeError("DIMO_DEV_EMPTY_MASK")
        if sizes != [(1024, 1024)] * 3:
            raise RuntimeError("DIMO_DEV_NATIVE_GEOMETRY_MISMATCH")
    return rows


def validate_predictions(path, rows, manifest_hash, protocol, *, expected_output_hashes=None):
    expected = {(r["sample_uid"], seed): r for r in rows for seed in protocol["generation_seeds"]}
    predictions = rows_at(path)
    keys = [(r["sample_uid"], r["seed"]) for r in predictions]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise RuntimeError("DIMO_DEV_PREDICTION_SAMPLE_SEED_MISMATCH")
    outputs = {}
    for r in predictions:
        ref = expected[(r["sample_uid"], r["seed"])]
        if (r.get("manifest_sha256") != manifest_hash
                or any(r.get(k) != ref.get(k) for k in ("source_sha256", "target_sha256", "region_sha256",
                                                       "group_id", "cluster_id", "instruction_en"))
                or any(r.get("inference_protocol", {}).get(k) != value for k, value in protocol.items())):
            raise RuntimeError("DIMO_DEV_PREDICTION_PROTOCOL_OR_CONTENT_MISMATCH")
        for key in ("source", "target", "mask"):
            if Path(r[key]).resolve() != Path(ref[key]).resolve():
                raise RuntimeError("DIMO_DEV_PREDICTION_NATIVE_GEOMETRY_MISMATCH")
        with Image.open(r["output"]) as image:
            if image.size != (protocol["resolution"], protocol["resolution"]):
                raise RuntimeError("DIMO_DEV_PREDICTION_RESOLUTION_MISMATCH")
        outputs[str(Path(r["output"]).resolve())] = sha256_file(r["output"])
    if expected_output_hashes is not None and outputs != expected_output_hashes:
        raise RuntimeError("DIMO_DEV_HISTORICAL_OUTPUT_CHANGED")
    return outputs


def validate_plan(plan):
    if (plan.get("schema") != "dimo-v2-dev128-v1" or plan.get("student_protocol") != STUDENT_PROTOCOL
            or plan.get("teacher_protocol") != TEACHER_PROTOCOL or plan.get("teacher_query_cfg") != 4.
            or plan.get("roi_crop_rule") != ROI_CROP_RULE):
        raise RuntimeError("DIMO_V2_DEV_PROTOCOL_CHANGED")
    for key in ("source_plan", "teacher12_plan"):
        if sha256_file(plan[key]) != plan[key + "_sha256"]:
            raise RuntimeError("DIMO_DEV_HISTORICAL_PLAN_CHANGED")
    for path, digest in plan["historical_identity_files"].items():
        if sha256_file(path) != digest:
            raise RuntimeError("DIMO_DEV_HISTORICAL_CHECKPOINT_IDENTITY_CHANGED")
    rows = validate_native_manifest(plan["manifest"], plan["manifest_sha256"])
    for name, entry in plan["historical"].items():
        if sha256_file(entry["predictions"]) != entry["predictions_sha256"]:
            raise RuntimeError("DIMO_DEV_HISTORICAL_PREDICTIONS_CHANGED")
        protocol = TEACHER_PROTOCOL if name == "SFT Teacher" else STUDENT_PROTOCOL
        validate_predictions(entry["predictions"], rows, plan["manifest_sha256"], protocol,
                             expected_output_hashes=entry["output_sha256"])
    if set(plan["historical"]) != {"SFT Teacher", "Old Raw", "Old EMA"}:
        raise RuntimeError("DIMO_DEV_HISTORICAL_MODELS_INCOMPLETE")
    return rows
