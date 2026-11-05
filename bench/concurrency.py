"""Latency of the running service under N simultaneous requests: N different short clips (FLEURS Arabic and
LibriSpeech from an Arabic Arena checkout, public data, or the first N audio files of a directory) posted at the same
moment, each from its own thread.

Prints how many succeeded or were turned away (503), the latency percentiles of the successful ones (from the
moment all requests are released to each response), the wall time and the audio seconds served per second.

    cd $ARENA_REPO && python /path/to/cohere-inference/bench/concurrency.py http://127.0.0.1:8086 50
    python bench/concurrency.py http://127.0.0.1:8086 100 --dir /path/to/fleurs/ar_eg/test [--rounds 3]
    python bench/concurrency.py http://127.0.0.1:8086 100 --dir ... --sustain 30   # steady load, 100 in flight
"""

import argparse
import http.client
import statistics
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit


def clips(n, directory=None):
    if directory:
        import soundfile as sf

        files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"))
        return [(f.read_bytes(), sf.info(f).duration) for f in files[:n]]
    import pyarrow.parquet as pq

    out = []
    for lang, name in (("ar", "fleurs_ar"), ("en", "librispeech_clean")):
        t = pq.read_table(f"data/{lang}/{name}/clean.parquet", columns=["audio", "duration"]).slice(0, (n + 1) // 2)
        out += [(r["audio"]["bytes"], r["duration"]) for r in t.to_pylist()]
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("n", type=int)
    ap.add_argument("--dir", help="audio files to send instead of the Arena clips")
    ap.add_argument("--rounds", type=int, default=1, help="repeat the burst (one after the other)")
    ap.add_argument("--sustain", type=float, default=0.0,
                    help="instead of bursts: N clients each sending its next request as soon as the last is answered, "
                         "for this many seconds (steady load of N requests in flight)")
    args = ap.parse_args()
    data = clips(args.n, args.dir)
    if args.sustain:
        sustained(args.url.rstrip("/"), data, args.sustain)
        return
    for _ in range(args.rounds):
        burst(args.url.rstrip("/"), data)


def sustained(url, data, seconds):
    n = len(data)
    u = urlsplit(url)
    forms = [body(audio) for audio, _ in data]
    lat, done_audio, errors = [], [0.0], [0]
    lock = threading.Lock()
    stop = time.perf_counter() + seconds

    def client(i):
        k = i
        while time.perf_counter() < stop:
            payload, ctype = forms[k % n]
            t = time.perf_counter()
            try:
                c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
                c.request("POST", u.path.rstrip("/") + "/transcribe", body=payload, headers={"Content-Type": ctype})
                r = c.getresponse()
                r.read()
                c.close()
                ok = r.status == 200
            except (OSError, http.client.HTTPException):
                ok = False
            with lock:
                if ok:
                    lat.append(time.perf_counter() - t)
                    done_audio[0] += data[k % n][1]
                else:
                    errors[0] += 1
            k += n // 7 + 1

    t0 = time.perf_counter()
    threads = [threading.Thread(target=client, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    lat.sort()
    pct = lambda p: lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))]  # noqa: E731
    print(f"{n} clients for {seconds:.0f} s: {len(lat)} answered ({errors[0]} errors), {len(lat) / wall:.1f} requests/s, "
          f"{done_audio[0] / wall:.0f} audio seconds per second")
    print(f"  latency: p50 {pct(0.5):.2f} s, p90 {pct(0.9):.2f} s, p99 {pct(0.99):.2f} s, max {lat[-1]:.2f} s")


def body(audio: bytes) -> tuple[bytes, str]:
    """A multipart form (file, language=auto) built before the clock starts, so the client's own work is not timed."""
    b = uuid.uuid4().hex
    head = (f"--{b}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nauto\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode()
    return head + audio + f"\r\n--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def burst(url, data):
    n = len(data)
    u = urlsplit(url)
    forms = [body(audio) for audio, _ in data]
    conns = [http.client.HTTPConnection(u.hostname, u.port, timeout=900) for _ in range(n)]
    for c in conns:
        c.connect()
    start = threading.Barrier(n + 1)
    results = [None] * n

    def one(i):
        payload, ctype = forms[i]
        start.wait()
        t = time.perf_counter()
        try:
            conns[i].request("POST", u.path.rstrip("/") + "/transcribe", body=payload, headers={"Content-Type": ctype})
            r = conns[i].getresponse()
            r.read()
            results[i] = (r.status, time.perf_counter() - t)
        except (OSError, http.client.HTTPException) as e:
            results[i] = (repr(e)[:40], time.perf_counter() - t)
        finally:
            conns[i].close()

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    start.wait()
    t0 = time.perf_counter()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    ok = sorted(lat for code, lat in results if code == 200)
    codes = {}
    for code, _ in results:
        codes[code] = codes.get(code, 0) + 1
    audio = sum(d for (_, d), (code, _) in zip(data, results) if code == 200)
    pct = lambda p: ok[min(len(ok) - 1, int(round(p * (len(ok) - 1))))] if ok else float("nan")  # noqa: E731
    print(f"{n} simultaneous requests ({sum(d for _, d in data):.0f} s of audio): status {codes}")
    if ok:
        print(f"  latency of the {len(ok)} answered: p50 {pct(0.5):.2f} s, p90 {pct(0.9):.2f} s, max {ok[-1]:.2f} s, "
              f"mean {statistics.mean(ok):.2f} s")
    print(f"  wall time {wall:.2f} s, {audio / wall:.1f} audio seconds served per second")


if __name__ == "__main__":
    main()
