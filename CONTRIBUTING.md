# Contributing

Danke fuer dein Interesse an AI Segmenter.

## Entwicklungsumgebung

1. Repository klonen oder herunterladen.
2. Installer ausfuehren:

   ```bat
   installer_starten.bat
   ```

3. Anwendung starten:

   ```bat
   start_programm.bat
   ```

Direktstart:

```powershell
.venv\Scripts\python.exe -m ai_segmenter
```

Fuer reine Code-/Testentwicklung:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Code-Stil

- Halte Hardware-spezifische Logik in eigenen Modulen.
- GUI-Orchestrierung gehoert in `ai_segmenter/app.py` oder passende Mixins.
- Modelle gehoeren nach `ai_segmenter/models/`.
- Generierte Dateien nicht committen.
- Keine `.venv`, Logs, TensorRT-Engines, ONNX-Dateien oder Modellgewichte committen.

## Tests vor Pull Requests

Mindestens ausfuehren:

```powershell
$files = @("script.py", "install_windows.py") + `
  (Get-ChildItem ai_segmenter -Filter *.py).FullName + `
  (Get-ChildItem ai_segmenter\models -Filter *.py).FullName
.venv\Scripts\python.exe -m py_compile $files
.venv\Scripts\python.exe -c "import script; from ai_segmenter.app import FoolproofSyncApp; import ai_segmenter.models; print('ok')"
.venv\Scripts\python.exe -m pytest
```

Bei UI-, Kamera-, DeckLink- oder Modell-Aenderungen bitte zusaetzlich manuell testen:

- Programmstart
- Kamera-Suche
- Kamera starten/stoppen
- Modellwechsel
- Postproduktion mit Bild
- Postproduktion mit Video
- DeckLink Input/Output, falls Hardware verfuegbar

## Issues

Bei Fehlern bitte angeben:

- Windows-Version
- Python-Version
- GPU und Treiberversion
- verwendetes Modell
- Live-Quelle oder Datei
- vollstaendige Fehlermeldung
- Schritte zum Reproduzieren

Sicherheitsrelevante Probleme bitte gemaess [SECURITY.md](SECURITY.md) melden.
