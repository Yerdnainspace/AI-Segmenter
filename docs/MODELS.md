# Modelle und externe Artefakte

Dieses Repository enthaelt den Anwendungscode, aber keine heruntergeladenen
Modellgewichte, TensorRT-Engines oder ONNX-Exporte.

## Warum diese Dateien nicht versioniert werden

Modellgewichte und generierte Beschleunigungsartefakte sind gross, haeufig
plattformabhaengig und koennen eigenen Lizenzbedingungen unterliegen. Sie
werden deshalb nicht direkt im Git-Repository gepflegt.

Nicht committen:

```text
*.pt
*.pth
*.safetensors
*.tflite
*.onnx
*.engine
assets/yolo_tensorrt/
CorridorKeyModule/
CorridorKeyModule/checkpoints/
```

## Herkunft der Modelle

Der Windows-Installer bereitet die benoetigten Komponenten lokal vor:

- MediaPipe Selfie Segmenter wird von den offiziellen MediaPipe-Modellquellen geladen.
- YOLO-Modelle werden ueber Ultralytics geladen bzw. lokal exportiert.
- BiRefNet wird ueber Hugging Face geladen.
- RVM ByteDance wird ueber Torch Hub geladen.
- CorridorKeyModule und CorridorKey-Checkpoints werden aus den angegebenen Upstream-Quellen geladen.

## Lizenzhinweis

Die MIT-Lizenz in `LICENSE` gilt fuer den Code dieses Projekts. Externe
Bibliotheken, Modellarchitekturen, Modellgewichte und Datensaetze koennen
eigene Lizenzen und Nutzungsbedingungen haben. Vor einer oeffentlichen
Distribution oder kommerziellen Nutzung muessen diese Bedingungen separat
geprueft werden.
