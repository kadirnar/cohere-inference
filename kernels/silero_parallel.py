"""Silero VAD speech probabilities of a whole recording, with everything but the LSTM run on all windows at once.

silero_vad.get_speech_timestamps feeds the model one 32 ms window at a time (the previous window's last 64 samples as
context plus 512 new ones) and carries the LSTM state from window to window; through onnxruntime that loop costs about
0.24 ms per window, some 38 s for the 87 minutes of the Audioluna calls. Only the LSTM depends on earlier windows: the
STFT, the four convolutions and the output head each see one window. So here the windows are cut from the recording
exactly as the loop cuts them (zeros before the start and after the end), the STFT and the convolutions run on all of
them as one batch, the LSTM runs over the whole sequence in one call (cuDNN on the GPU), and the head on all of its
outputs. Same weights (the package's silero_vad.jit), fp32 throughout with TF32 off; the probabilities go to
silero_vad's own get_speech_timestamps_from_probs, so the regions are the loop's.

    vad = ParallelSilero("cuda")
    regions = vad.speech_timestamps(x, threshold=0.5, min_speech_duration_ms=200, ...)   # x: 16 kHz mono numpy
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

SR, WINDOW, CONTEXT = 16000, 512, 64
CHUNK = 8192  # windows per call (~4 minutes of audio): bounds the activations and cuDNN's LSTM workspace


class ParallelSilero:
    def __init__(self, device: str = "cuda"):
        import silero_vad
        from silero_vad.utils_vad import get_speech_timestamps_from_probs

        self.from_probs = get_speech_timestamps_from_probs
        path = os.path.join(os.path.dirname(silero_vad.__file__), "data", "silero_vad.jit")
        sd = {k: v.float() for k, v in torch.jit.load(path, map_location="cpu")._model.state_dict().items()}
        self.device = torch.device(device)
        to = lambda t: t.to(self.device)  # noqa: E731
        self.basis = to(sd["stft.forward_basis_buffer"])  # (258, 1, 256): real and imaginary parts, 129 bins
        self.convs = [(to(sd[f"encoder.{i}.reparam_conv.weight"]), to(sd[f"encoder.{i}.reparam_conv.bias"]), stride)
                      for i, stride in enumerate((1, 2, 2, 1))]
        self.lstm = torch.nn.LSTM(128, 128, batch_first=True).to(self.device).eval()
        with torch.no_grad():
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(self.lstm, f"{name}_l0").copy_(sd[f"decoder.rnn.{name}"])
        self.head = (to(sd["decoder.decoder.2.weight"]), to(sd["decoder.decoder.2.bias"]))
        self.lock = threading.Lock()
        if self.device.type == "cuda":
            torch.backends.cudnn.allow_tf32 = False  # fp32 convolutions and LSTM as on the CPU (the ASR model is bf16)
            self.stream = torch.cuda.Stream()  # beside the ASR batches, not queued behind them

    @torch.no_grad()
    def probs(self, x: np.ndarray) -> list[float]:
        """Speech probability of every 512-sample window of x (16 kHz mono), as get_speech_timestamps computes them."""
        n = -(-len(x) // WINDOW)
        if self.device.type == "cuda":
            with self.lock, torch.cuda.stream(self.stream):  # one LSTM state per call; cuDNN handles are per stream
                p = self._probs(x, n)
                return p.cpu().tolist()
        return self._probs(x, n).tolist()

    def _probs(self, x: np.ndarray, n: int) -> torch.Tensor:
        y = torch.zeros(CONTEXT + n * WINDOW)
        y[CONTEXT:CONTEXT + len(x)] = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        windows = y.to(self.device).unfold(0, CONTEXT + WINDOW, WINDOW)  # (windows, 576): context + window
        probs, state = [], None  # zero initial LSTM state, then carried from chunk to chunk
        for i in range(0, n, CHUNK):
            h, state = self.lstm(self._encode(windows[i:i + CHUNK])[None], state)  # (1, windows, 128)
            probs.append(torch.sigmoid(F.conv1d(F.relu(h[0])[:, :, None], *self.head))[:, 0, 0])
        return torch.cat(probs)

    def _encode(self, w: torch.Tensor) -> torch.Tensor:
        """The stateless part, (windows, 576) -> (windows, 128): STFT magnitudes, then the four convolutions."""
        z = F.conv1d(F.pad(w[:, None], (0, 64), mode="reflect"), self.basis, stride=128)  # (windows, 258, 4)
        z = torch.sqrt(z[:, :129] ** 2 + z[:, 129:] ** 2)
        for weight, bias, stride in self.convs:
            z = F.relu(F.conv1d(z, weight, bias, stride=stride, padding=1))
        return z[:, :, 0]

    def speech_timestamps(self, x: np.ndarray, **settings) -> list[dict]:
        """silero_vad.get_speech_timestamps(x, model, sampling_rate=16000, **settings), from these probabilities."""
        return self.from_probs(self.probs(x), sampling_rate=SR, audio_length_samples=len(x), **settings)
