# Architektur

AI Segmenter ist als Python-Package aufgebaut. Der Einstiegspunkt liegt in
`ai_segmenter.__main__`, die Windows-Starter rufen diesen Paketstart auf.

## Aktuelle Modulgrenzen

Bereits getrennt:

- `ai_segmenter/models/`: Modelladapter fuer MediaPipe, BiRefNet, RVM, YOLO und CorridorKey
- `ai_segmenter/decklink.py`: DeckLink Input/Output und Blackmagic-API-Anbindung
- `ai_segmenter/live_output.py`: Fill/Matte-Ausgabe, Output-Sync und Delay-Logik
- `ai_segmenter/postprocessing.py`: Datei-/Videoverarbeitung und Export
- `ai_segmenter/metrics.py`: Live-Metriken und Profiler-Integration
- `ai_segmenter/yolo_controls.py`: YOLO-Laden, Objektauswahl, ROI-Logik und Overlay
- `ai_segmenter/preview_renderer.py`: Alpha-Nachbearbeitung, Hintergrundauswahl und Preview-Rendering
- `ai_segmenter/ui/app_layout.py`: GUI-Aufbau, Control Panel, Preview-Layout und UI-Hilfsmethoden
- `ai_segmenter/pipeline/camera_lifecycle.py`: Live-Quellenwechsel, Kamera-/DeckLink-Start, Stop und Shutdown
- `ai_segmenter/pipeline/live_pipeline.py`: Live-Pipeline-Threads, Frame-Buffer, AI-Loop und Output-Loop
- `ai_segmenter/image_utils.py`: Bild- und Alpha-Helfer
- `ai_segmenter/camera.py`: Kameraauswahl, Parsing und Capture-Helfer
- `ai_segmenter/config.py`: zentrale Optionen und Konstanten

## Bewertung

Die Struktur ist fuer eine erste Open-Source-Version brauchbar und klar besser
als ein einzelnes Skript. Die wichtigsten technischen Domänen sind bereits in
eigene Module ausgelagert. YOLO-Auswahl, Preview-Rendering, GUI-Layout,
Kamera-Lifecycle und Live-Pipeline-Threads sind inzwischen aus `app.py`
herausgezogen.

`ai_segmenter/app.py` bleibt der zentrale App-Orchestrator. Die Datei enthaelt
noch wenige zentrale Verantwortlichkeiten:

- UI-State
- Modellwechsel
- CorridorKey-Lifecycle
- Hintergrundbild-Laden
- App-Lifecycle

Das ist fuer eine erste GitHub-Version akzeptabel. Fuer laengerfristige Wartung
sollte als naechstes die Mixin-Struktur spaeter durch explizitere
Controller-Klassen ersetzt werden.

## Empfohlene Refactor-Schritte

1. `ai_segmenter/ui/app_layout.py` bei weiterem Wachstum in kleinere UI-Module teilen:
   - `preview_panel.py`
   - `model_controls.py`
   - `output_controls.py`
   - `postproduction_controls.py`

2. Pipeline-State weiter typisieren:
   - `pipeline/frame_buffers.py`
   - `pipeline/state.py`

3. Tests fuer die ausgelagerten Module ergaenzen:
   - YOLO Detection-Signaturen
   - ROI-Erzeugung und Alpha-Kombination
   - Checker, Alpha-Preview, Greenscreen und Custom Background

4. App-State typisieren:
   - `dataclasses` fuer Model-State, Output-State, Postprocessing-State und Pipeline-State

Diese Schritte sollten inkrementell passieren. Ein grosser Komplettumbau vor
der ersten Veroeffentlichung waere riskanter als ein dokumentierter, getesteter
Refactor in mehreren kleinen Pull Requests.
