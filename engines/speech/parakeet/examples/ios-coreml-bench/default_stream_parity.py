#!/usr/bin/env python3
"""Compare default one-second push context against the same native configuration."""
import json
import re
from pathlib import Path
import benchmark as batch
import stream_benchmark as streaming


def transcript(content):
    pieces = []
    for line in content.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if 'text' in value:
            pieces.append(value['text'])
    return ''.join(pieces)


def main():
    batch.configure()
    result_path = batch.OUT / "default-stream-parity.json"
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    for device in batch.DEVICES:
        for wav, language in [('jfk.wav', 'auto'), ('fr-multipiece-16k.wav', 'auto'),
                              ('fr-long.wav', 'auto'), ('jfk.wav', 'other')]:
            for requested in (0, 1000):
                if any(row['device'] == device and row['wav'] == wav and
                       row['language'] == language and row.get('requested_chunk_ms') == requested
                       for row in rows):
                    continue
                texts = []
                for backend in ('cpu', 'coreml'):
                    args = streaming.arguments(device, wav, backend, language, True)
                    index = args.index('--stream-chunk-ms')
                    if requested:
                        args[index + 1] = str(requested)
                    else:
                        del args[index:index + 2]
                    name = f'{device}-default-{Path(wav).stem}-{language}-{requested}-{backend}'
                    content = streaming.invoke(device, args, backend, batch.OUT / (name + '.log'))
                    expected = requested or 2000
                    if f'chunk_ms={expected}' not in content:
                        raise RuntimeError('CLI did not expose the resolved chunk size')
                    texts.append(transcript(content))
                if texts[0] != texts[1]:
                    raise RuntimeError(f'Push differs from native: {device} {wav} {language} {requested}: {texts}')
                row = dict(device=device, wav=wav, language=language, requested_chunk_ms=requested,
                           chunk_ms=requested or 2000, native_equal=True, transcript=texts[1],
                           encoder_placement=re.findall(r'encoder_backend=([^\s]+)', content)[-1])
                batch.score_transcript(row, wav)
                rows.append(row)
                result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
                print(device, wav, language, requested, 'push matches native', flush=True)


if __name__ == '__main__':
    main()
