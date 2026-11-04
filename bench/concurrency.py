"""Latency of the running service under N simultaneous requests: N different short clips (FLEURS Arabic and
LibriSpeech from an Arabic Arena checkout, public data) posted at the same moment, each from its own thread.

Prints how many succeeded or were turned away (503), the latency percentiles of the successful ones (from the
moment all requests are released to each response), the wall time and the audio seconds served per second.

    cd $ARENA_REPO && python /path/to/cohere-inference/bench/concurrency.py http://127.0.0.1:8086 50
"""

import io
import statistics
import sys
import threading
import time

import pyarrow.parquet as pq
import requests


def clips(n):
    out = []
    for lang, name in (("ar", "fleurs_ar"), ("en", "librispeech_clean")):
        t = pq.read_table(f"data/{lang}/{name}/clean.parquet", columns=["audio", "duration"]).slice(0, (n + 1) // 2)
        out += [(r["audio"]["bytes"], r["duration"]) for r in t.to_pylist()]
    return out[:n]


def main():
    url, n = sys.argv[1].rstrip("/"), int(sys.argv[2])
    data = clips(n)
    start = threading.Barrier(n + 1)
    results = [None] * n

    def one(i):
        audio, _ = data[i]
        start.wait()
        t = time.perf_counter()
        try:
            r = requests.post(url + "/transcribe", files={"file": ("clip.wav", io.BytesIO(audio))},
                              data={"language": "auto"}, timeout=900)
            results[i] = (r.status_code, time.perf_counter() - t)
        except requests.RequestException as e:
            results[i] = (repr(e)[:40], time.perf_counter() - t)

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
