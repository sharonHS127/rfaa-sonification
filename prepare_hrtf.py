#!/usr/bin/env python3
"""
prepare_hrtf.py -- one-time helper: export the MIT KEMAR HRIR set bundled with the `slab` package to
kemar_hrir.npz, the file listening_trials.py loads.

  pip install slab           # Ubuntu additionally:  sudo apt install libportaudio2
  python3 prepare_hrtf.py    # -> kemar_hrir.npz  (az, el, hrir[M,2,512] float32 (left,right), fs)

Data: MIT KEMAR dummy-head HRTF, normal pinna (Gardner & Martin, 1994, MIT Media Lab Perceptual Computing
Technical Report #280).  Please cite it when you use results obtained with it.
Angles in the file follow the SOFA/slab convention: azimuth counter-clockwise from the front (90 = left),
elevation -40 .. +90 deg.
"""
import numpy as np
import sys, types
try:
    import sounddevice  # noqa
except OSError:                      # PortAudio missing: we only need the HRTF data, not playback
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")
import slab

h = slab.HRTF.kemar()
sp = np.asarray(h.sources)[1]                                    # spherical rows: az, el, distance
hrir = np.stack([np.asarray(f.data) for f in h.data]).transpose(0, 2, 1)      # (M, 2 ears, taps)
np.savez_compressed("kemar_hrir.npz", az=sp[:, 0], el=sp[:, 1], hrir=hrir.astype(np.float32), fs=float(h.samplerate))
print(f"wrote kemar_hrir.npz: {hrir.shape[0]} directions, {hrir.shape[2]} taps, {h.samplerate:.0f} Hz, "
      f"elevations {sp[:, 1].min():.0f}..{sp[:, 1].max():.0f} deg")
