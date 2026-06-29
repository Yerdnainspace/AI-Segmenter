# Testplan

Dieser Testplan beschreibt die wichtigsten manuellen Tests vor einer Veroeffentlichung.

## Installation

- Frischen Ordner verwenden
- `installer_starten.bat` ausfuehren
- Pruefen, ob `.venv` erstellt wird
- Pruefen, ob MediaPipe-Modell geladen wird
- Pruefen, ob Startdateien geschrieben werden

## Start

```powershell
.venv\Scripts\python.exe -m ai_segmenter
.venv\Scripts\python.exe script.py
```

Beide Starts muessen die Anwendung oeffnen.

## Live-Modus

- Automatische Kamerasuche beim Start
- `Kameras neu suchen`
- Kamera starten
- Kamera stoppen
- Kamera wechseln
- Programm waehrend aktiver Kamera schliessen

## Modelle

- MediaPipe Selfie laden
- BiRefNet laden
- BiRefNet TensorRT laden, falls CUDA/TensorRT vorhanden
- RVM ByteDance laden
- YOLO laden
- YOLO TensorRT laden, falls CUDA/TensorRT vorhanden
- Modellwechsel bei gestoppter Kamera
- Modellwechsel bei laufender Kamera

## YOLO

- YOLO-Nachbearbeitung aktivieren
- Confidence aendern
- Sync-Modus testen
- Async-Modus testen
- Einzelne Objekte auswaehlen
- Alle Objekte automatisch auswaehlen

## CorridorKey

- CorridorKey aktivieren
- Hardware-Modus wechseln
- Despill aendern
- Despeckle aendern
- Deaktivieren und Pipeline weiterlaufen lassen

## DeckLink

Nur mit passender Hardware:

- DeckLink-Geraete suchen
- DeckLink Input als Quelle starten
- Fill Output starten
- Alpha Matte Output starten
- Fill/Matte auf demselben Ausgang testen, Fehler erwartet
- Output-Sync-Overlay testen
- Fill/Matte Delay testen

## Postproduktion

- Bild importieren und PNG exportieren
- Video importieren und MP4 exportieren
- Transparent-Modus mit Bild
- Transparent-Modus mit MOV/ProRes 4444
- Ungueltige Quelldatei testen
- Fehlendes FFmpeg testen

## Performance

- 10 Minuten Live-Betrieb
- Mehrfach starten/stoppen
- Logs unter `logs/` pruefen
- Keine haengenden Python-Prozesse nach Programmende

