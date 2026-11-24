#!/usr/bin/env python3
import argparse, importlib.metadata, json, random
from pathlib import Path
from pocket_tts.modules.text_conditioner import SentencePieceTokenizer
from pocket_tts.models.text_chunking import prepare_text_prompt,split_into_best_sentences
parser = argparse.ArgumentParser(description="Dump upstream Pocket frontend corpus")
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
direct = json.loads(importlib.metadata.distribution("pocket-tts").read_text("direct_url.json"))
if direct.get("vcs_info", {}).get("commit_id") != "0c2db3bdea7c991c568989cc11b503f14483fabc":
    raise RuntimeError("Install the pinned Pocket reference")
p=SentencePieceTokenizer(4000, str(args.tokenizer))
corpus=['Hello',' Hello  world ','a\tb','a\nb','a\u00a0b','😊','こんにちは','A\u2581B','ßeta','éclair','hello,','hello—','say "hi"','say "hi!"','a\x00b','3.14 is pi. 2.71 is e.','The value is ٣.١٤ today. Yes.','one  two   three    four','\t hello\nworld \r','Mr. Smith reads 12.05 at 3:40.','Wait... really?! Yes!','One sentence; the second clause: then another, and another.']
r=random.Random(145)
words=['hello','world','nice','Pocket','TTS','five','Café','123','😃','résumé','punctuation,','is','final.']
for i in range(300):corpus.append(' '.join(r.choices(words,k=r.randrange(1,75))))
rows=[]
for s in corpus:
 prepared,tail=prepare_text_prompt(s,False,False)
 chunks=split_into_best_sentences(p,s,50,False,False)
 rows.append(dict(text=s,ids=p(s)[0].tolist(),decoded=p.sp.decode(p(s)[0].tolist()),prepared=prepared,tail=tail+2,chunks=[dict(text=prepare_text_prompt(c,False,False)[0],tail=prepare_text_prompt(c,False,False)[1]+2) for c in chunks]))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
