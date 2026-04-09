"""Pi classifier supporting BOTH 3-class and 5-class models with runtime switching.

State file: ~/baby_cry/storage/active_model.txt  contains "3cls" or "5cls".
Both interpreters are loaded at startup. predict() reads the active mode each call
(cheap mtime check) so dashboard can switch live without restart.
"""
import numpy as np
import tflite_runtime.interpreter as tflite
import os, json, joblib, librosa
from config import settings


STATE_FILE = os.path.join(settings.BASE_DIR, "storage", "active_model.txt")


class Classifier:
    def __init__(self):
        model_dir = os.path.join(settings.BASE_DIR, 'model')

        # 3-class
        self.t3 = self._load_one(
            os.path.join(model_dir, 'cry_model_v3cls.tflite'),
            os.path.join(model_dir, 'label_encoder_3cls.joblib'),
            os.path.join(model_dir, 'meta_final.json'),
        )

        # 5-class
        self.t5 = self._load_one(
            os.path.join(model_dir, 'cry_model_5cls_aug.tflite'),
            os.path.join(model_dir, 'label_encoder_5cls.joblib'),
            os.path.join(model_dir, 'meta_5cls.json'),
        )

        if self.t3:
            print(f"  3cls classes: {list(self.t3['le'].classes_)}")
        if self.t5:
            print(f"  5cls classes: {list(self.t5['le'].classes_)}")

        self._state_mtime = 0
        self.active = self._read_active()
        self.mode = 'v3cls'  # legacy attr some callers may check

    def _load_one(self, tflite_path, le_path, meta_path):
        if not all(os.path.exists(p) for p in [tflite_path, le_path, meta_path]):
            print(f"  skip (missing): {tflite_path}")
            return None
        interp = tflite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        with open(meta_path) as f:
            meta = json.load(f)
        return {
            'interp': interp,
            'in':  interp.get_input_details()[0],
            'out': interp.get_output_details()[0],
            'le':  joblib.load(le_path),
            'mean': float(meta['feature_mean']),
            'std':  float(meta['feature_std']),
        }

    def _read_active(self):
        try:
            mt = os.path.getmtime(STATE_FILE)
            if mt != self._state_mtime:
                self._state_mtime = mt
                with open(STATE_FILE) as f:
                    val = f.read().strip()
                if val in ('3cls', '5cls'):
                    return val
        except FileNotFoundError:
            pass
        return getattr(self, 'active', '3cls')

    def predict(self, features):
          """features: raw 1D audio OR (64, time) mel spectrogram."""
          import librosa
          if features.ndim == 1:
              target_len = 22050 * 2
              if len(features) < target_len:
                  features = np.pad(features, (0, target_len - len(features)))
              features = features[:target_len]
              mel = librosa.feature.melspectrogram(y=features, sr=22050, n_mels=64)
              mel_spec_db = librosa.power_to_db(mel, ref=np.max)
          else:
              mel_spec_db = features

          # Live switch check
          new_active = self._read_active()
          if new_active != getattr(self, 'active', None):
              print(f"[classifier] switched: {getattr(self,'active',None)} -> {new_active}", flush=True)
              self.active = new_active

          bundle = self.t5 if self.active == '5cls' else self.t3
          if bundle is None:
              bundle = self.t3 or self.t5
              if bundle is None:
                  return "no_cry", 0.0, {}

          x = ((mel_spec_db - bundle['mean']) / bundle['std']).astype(np.float32)
          x = x[None, :, :, None]
          interp = bundle['interp']
          interp.set_tensor(bundle['in']['index'], x)
          interp.invoke()
          logits = interp.get_tensor(bundle['out']['index'])[0]
          # softmax
          e = np.exp(logits - logits.max())
          probs = e / e.sum()
          idx = int(np.argmax(probs))
          label = str(bundle['le'].classes_[idx])
          conf = float(probs[idx])
          all_probs = {str(c): float(probs[i]) for i, c in enumerate(bundle['le'].classes_)}
          return label, conf, all_probs


_classifier = None
def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = Classifier()
    return _classifier


def predict(mel_spec_db):
    return get_classifier().predict(mel_spec_db)
