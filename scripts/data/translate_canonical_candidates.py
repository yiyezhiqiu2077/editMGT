#!/usr/bin/env python3
"""Translate only candidate/reserve prompts requested by iterative corpus build."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.data.translate_interedit import mock_translate,nllb_translate
from src.explicit_region.language import JsonlTranslationCache,contains_han,translation_cache_key,translation_qa_flags


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--required",required=True);parser.add_argument("--cache",required=True);parser.add_argument("--qa-output",required=True)
    parser.add_argument("--backend",choices=("nllb","mock"),default="nllb");parser.add_argument("--model-path");parser.add_argument("--revision",required=True);parser.add_argument("--batch-size",type=int,default=32);parser.add_argument("--formal",action="store_true");parser.add_argument("--src-lang",default="zho_Hans");parser.add_argument("--tgt-lang",default="eng_Latn")
    args=parser.parse_args()
    if args.formal and args.backend=="mock":raise SystemExit("MOCK_TRANSLATION_FORBIDDEN_IN_FORMAL")
    if args.backend=="nllb" and not args.model_path:raise SystemExit("TRANSLATOR_MODEL_NOT_FOUND")
    decoding={"do_sample":False,"num_beams":1,"max_new_tokens":128}
    requested=[]
    for line in Path(args.required).open(encoding="utf-8"):
        if line.strip():
            text=json.loads(line)["instruction_original"]
            if contains_han(text) and text not in requested:requested.append(text)
    cache=JsonlTranslationCache(args.cache);pending=[];keys=[]
    for text in requested:
        key=translation_cache_key(text,args.backend,args.revision,args.src_lang,args.tgt_lang,decoding)
        if cache.get(key) is None:pending.append(text);keys.append(key)
    translated=(mock_translate(pending) if args.backend=="mock" else nllb_translate(pending,args.model_path,args.revision,decoding,args.batch_size,args.src_lang,args.tgt_lang))
    qa=[]
    for text,value,key in zip(pending,translated,keys):
        flags=translation_qa_flags(text,value);row={"cache_key":key,"source_text":text,"translated_text":value,"backend":args.backend,"revision":args.revision,"source_language":args.src_lang,"target_language":args.tgt_lang,"decoding_config":decoding,"status":"ok" if not flags else "qa_flagged","qa_flags":flags}
        cache.append(row)
        if flags:qa.append(row)
    output=Path(args.qa_output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in qa),encoding="utf-8")
    print(json.dumps({"requested_unique":len(requested),"translated_now":len(pending),"qa_failed":len(qa)}))

if __name__=="__main__":main()
