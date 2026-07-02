import ctypes
import threading
import time

import cv2
import numpy as np

from .camera import get_available_cameras
from .config import DECKLINK_OUTPUT_MODES, DECKLINK_SDK_DLL_PATHS


def load_decklink_api():
    try:
        from comtypes import client
    except ImportError as exc:
        raise RuntimeError("DeckLink-Ausgabe benötigt comtypes. Installiere es mit pip install comtypes.") from exc

    errors = []
    for path in ["DeckLinkAPI.dll", *DECKLINK_SDK_DLL_PATHS]:
        try:
            client.GetModule(path)
            import comtypes.gen.DeckLinkAPI as decklink_api
            return decklink_api
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    raise RuntimeError(
        "DeckLinkAPI.dll konnte nicht geladen werden. Blackmagic Desktop Video muss installiert sein.\n"
        + "\n".join(errors[-3:])
    )


def get_decklink_output_devices():
    try:
        return DeckLinkLiveOutput.list_devices()
    except Exception:
        return ["Keine DeckLink-Ausgabe gefunden"]


def get_decklink_input_devices():
    try:
        return DeckLinkLiveInput.list_devices()
    except Exception:
        return []


def get_live_input_sources():
    decklink_inputs = get_decklink_input_devices()
    decklink_sources = [f"DeckLink: {name}" for name in decklink_inputs]
    camera_sources = get_available_cameras()
    if camera_sources == ["Keine Kamera gefunden"]:
        camera_sources = []
    sources = decklink_sources + camera_sources
    return sources if sources else ["Keine Live-Quelle gefunden"]


class DeckLinkLiveOutput:
    def __init__(self, device_name, mode_label, status_callback=None):
        self.device_name = device_name
        self.mode_label = mode_label
        self.status_callback = status_callback
        _, self.width, self.height, self.fps = DECKLINK_OUTPUT_MODES[mode_label]
        self._lock = threading.Lock()
        self._latest_rgb_or_rgba = None
        self._last_rgb_or_rgba = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = None
        self._error = None

    @staticmethod
    def list_devices():
        import comtypes
        from comtypes.client import CreateObject

        comtypes.CoInitialize()
        try:
            decklink_api = load_decklink_api()
            iterator = CreateObject(decklink_api.CDeckLinkIterator, interface=decklink_api.IDeckLinkIterator)
            devices = []
            while True:
                try:
                    decklink = iterator.Next()
                except Exception:
                    break
                if not decklink:
                    break

                try:
                    attributes = decklink.QueryInterface(decklink_api.IDeckLinkProfileAttributes)
                    io_support = int(attributes.GetInt(decklink_api.BMDDeckLinkVideoIOSupport))
                    if not (io_support & int(decklink_api.bmdDeviceSupportsPlayback)):
                        continue
                    devices.append(str(decklink.GetDisplayName()))
                except Exception:
                    continue

            return devices if devices else ["Keine DeckLink-Ausgabe gefunden"]
        finally:
            comtypes.CoUninitialize()

    def start(self):
        self._thread = threading.Thread(target=self._run, name="DeckLinkLiveOutput", daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=3.0)
        if self._error:
            raise self._error

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def write(self, frame_rgb_or_rgba):
        if self._thread is None or not self._thread.is_alive():
            return
        if frame_rgb_or_rgba is None:
            return
        with self._lock:
            self._latest_rgb_or_rgba = np.ascontiguousarray(frame_rgb_or_rgba)

    def _set_status(self, text):
        if self.status_callback:
            self.status_callback(text)

    def _select_device(self, decklink_api):
        from comtypes.client import CreateObject

        iterator = CreateObject(decklink_api.CDeckLinkIterator, interface=decklink_api.IDeckLinkIterator)
        first_playback_device = None
        while True:
            try:
                decklink = iterator.Next()
            except Exception:
                break
            if not decklink:
                break

            try:
                attributes = decklink.QueryInterface(decklink_api.IDeckLinkProfileAttributes)
                io_support = int(attributes.GetInt(decklink_api.BMDDeckLinkVideoIOSupport))
                if not (io_support & int(decklink_api.bmdDeviceSupportsPlayback)):
                    continue
                display_name = str(decklink.GetDisplayName())
            except Exception:
                continue

            if first_playback_device is None:
                first_playback_device = decklink
            if display_name == self.device_name:
                return decklink, display_name

        if first_playback_device is not None:
            return first_playback_device, str(first_playback_device.GetDisplayName())
        raise RuntimeError("Keine DeckLink-Ausgabekarte gefunden.")

    def _frame_to_bgra(self, frame):
        frame = np.asarray(frame)
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise RuntimeError("DeckLink-Ausgabe erwartet RGB oder RGBA Frames.")

        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)

    def _write_decklink_frame(self, decklink_output, frame):
        bgra = np.ascontiguousarray(self._frame_to_bgra(frame), dtype=np.uint8)
        video_frame = decklink_output.CreateVideoFrame(
            self.width,
            self.height,
            self.width * 4,
            self._decklink_api.bmdFormat8BitBGRA,
            self._decklink_api.bmdFrameFlagDefault,
        )
        buffer_ptr = video_frame.GetBytes()
        if hasattr(buffer_ptr, "value"):
            buffer_ptr = buffer_ptr.value
        ctypes.memmove(buffer_ptr, bgra.ctypes.data, bgra.nbytes)
        decklink_output.DisplayVideoFrameSync(video_frame)

    def _run(self):
        import comtypes

        comtypes.CoInitialize()
        decklink_output = None
        try:
            self._decklink_api = load_decklink_api()
            decklink, actual_device_name = self._select_device(self._decklink_api)
            decklink_output = decklink.QueryInterface(self._decklink_api.IDeckLinkOutput_v14_2_1)
            mode_name, _, _, _ = DECKLINK_OUTPUT_MODES[self.mode_label]
            display_mode = getattr(self._decklink_api, mode_name)

            _actual_mode, supported = decklink_output.DoesSupportVideoMode(
                self._decklink_api.bmdVideoConnectionUnspecified,
                display_mode,
                self._decklink_api.bmdFormat8BitBGRA,
                self._decklink_api.bmdNoVideoOutputConversion,
                self._decklink_api.bmdSupportedVideoModeDefault,
            )
            if not supported:
                raise RuntimeError(f"{actual_device_name} unterstuetzt {self.mode_label} mit BGRA nicht.")

            decklink_output.EnableVideoOutput(display_mode, self._decklink_api.bmdVideoOutputFlagDefault)
            self._set_status(f"DeckLink aktiv: {actual_device_name}\n{self.mode_label}")
            self._ready_event.set()

            black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self._last_rgb_or_rgba = black
            period = 1.0 / max(1.0, float(self.fps))
            next_frame_time = time.perf_counter()

            while not self._stop_event.is_set():
                with self._lock:
                    frame = self._latest_rgb_or_rgba
                    self._latest_rgb_or_rgba = None
                if frame is None:
                    frame = self._last_rgb_or_rgba
                else:
                    self._last_rgb_or_rgba = frame

                self._write_decklink_frame(decklink_output, frame)
                next_frame_time += period
                sleep_time = next_frame_time - time.perf_counter()
                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)
                else:
                    next_frame_time = time.perf_counter()
        except Exception as exc:
            self._error = exc
            self._set_status(f"DeckLink Fehler: {exc}")
            self._ready_event.set()
        finally:
            if decklink_output is not None:
                try:
                    decklink_output.DisableVideoOutput()
                except Exception:
                    pass
            comtypes.CoUninitialize()


def create_decklink_input_callback(owner, decklink_api):
    from comtypes import COMObject

    class _DeckLinkInputCallback(COMObject):
        _com_interfaces_ = [decklink_api.IDeckLinkInputCallback_v14_2_1]

        def VideoInputFormatChanged(self, notificationEvents, newDisplayMode, detectedSignalFlags):
            owner._on_video_format_changed(newDisplayMode, detectedSignalFlags)
            return 0

        def VideoInputFrameArrived(self, videoFrame, audioPacket):
            if videoFrame:
                owner._on_video_frame(videoFrame)
            return 0

    return _DeckLinkInputCallback()


class DeckLinkLiveInput:
    def __init__(self, device_name, mode_label):
        self.device_name = device_name
        self.mode_label = mode_label
        _, self.width, self.height, self.fps = DECKLINK_OUTPUT_MODES[mode_label]
        self._decklink_api = None
        self._decklink_input = None
        self._callback = None
        self._lock = threading.Lock()
        self._format_lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._latest_rgb = None
        self._latest_capture_actual_s = 0.0
        self._opened = False
        self._com_initialized = False
        self._input_pixel_format = None
        self._input_flags = 0
        self._received_frames = 0
        self._overwritten_frames = 0
        self._last_reported_received = 0
        self._last_reported_overwritten = 0

    @staticmethod
    def list_devices():
        import comtypes
        from comtypes.client import CreateObject

        comtypes.CoInitialize()
        try:
            decklink_api = load_decklink_api()
            iterator = CreateObject(decklink_api.CDeckLinkIterator, interface=decklink_api.IDeckLinkIterator)
            devices = []
            while True:
                try:
                    decklink = iterator.Next()
                except Exception:
                    break
                if not decklink:
                    break

                try:
                    attributes = decklink.QueryInterface(decklink_api.IDeckLinkProfileAttributes)
                    io_support = int(attributes.GetInt(decklink_api.BMDDeckLinkVideoIOSupport))
                    if not (io_support & int(decklink_api.bmdDeviceSupportsCapture)):
                        continue
                    devices.append(str(decklink.GetDisplayName()))
                except Exception:
                    continue

            return devices
        finally:
            comtypes.CoUninitialize()

    def open(self):
        import comtypes
        from comtypes.client import CreateObject

        if self._opened:
            return True

        comtypes.CoInitialize()
        self._com_initialized = True
        self._decklink_api = load_decklink_api()
        iterator = CreateObject(self._decklink_api.CDeckLinkIterator, interface=self._decklink_api.IDeckLinkIterator)

        decklink = None
        while True:
            try:
                candidate = iterator.Next()
            except Exception:
                break
            if not candidate:
                break
            try:
                name = str(candidate.GetDisplayName())
                attributes = candidate.QueryInterface(self._decklink_api.IDeckLinkProfileAttributes)
                io_support = int(attributes.GetInt(self._decklink_api.BMDDeckLinkVideoIOSupport))
                if name == self.device_name and (io_support & int(self._decklink_api.bmdDeviceSupportsCapture)):
                    decklink = candidate
                    break
            except Exception:
                continue

        if decklink is None:
            raise RuntimeError(f"DeckLink Input nicht gefunden: {self.device_name}")

        self._decklink_input = decklink.QueryInterface(self._decklink_api.IDeckLinkInput_v14_2_1)
        mode_name, _, _, _ = DECKLINK_OUTPUT_MODES[self.mode_label]
        display_mode = getattr(self._decklink_api, mode_name)
        pixel_format = self._decklink_api.bmdFormat8BitYUV
        _actual_mode, supported = self._decklink_input.DoesSupportVideoMode(
            self._decklink_api.bmdVideoConnectionUnspecified,
            display_mode,
            pixel_format,
            self._decklink_api.bmdNoVideoInputConversion,
            self._decklink_api.bmdSupportedVideoModeDefault,
        )
        if not supported:
            raise RuntimeError(f"{self.device_name} unterstuetzt {self.mode_label} als DeckLink Input nicht.")

        self._callback = create_decklink_input_callback(self, self._decklink_api)
        self._decklink_input.SetCallback(self._callback)
        self._input_pixel_format = pixel_format
        self._input_flags = 0
        self._decklink_input.EnableVideoInput(
            display_mode,
            pixel_format,
            self._input_flags,
        )
        self._decklink_input.StartStreams()
        self._opened = True
        return True

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def read(self, timeout=1.0):
        ret, frame, _capture_actual_s = self.read_with_timing(timeout=timeout)
        return ret, frame

    def read_with_timing(self, timeout=1.0):
        deadline = time.perf_counter() + float(timeout)
        with self._frame_ready:
            while self._opened and self._latest_rgb is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return False, None, 0.0
                self._frame_ready.wait(remaining)
            if self._latest_rgb is None:
                return False, None, 0.0
            frame = self._latest_rgb
            capture_actual_s = float(self._latest_capture_actual_s or 0.0)
            self._latest_rgb = None
        convert_start = time.perf_counter()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        capture_actual_s += time.perf_counter() - convert_start
        return True, frame_bgr, capture_actual_s

    def consume_frame_stats(self):
        with self._frame_ready:
            received_delta = self._received_frames - self._last_reported_received
            overwritten_delta = self._overwritten_frames - self._last_reported_overwritten
            self._last_reported_received = self._received_frames
            self._last_reported_overwritten = self._overwritten_frames
            return received_delta, overwritten_delta

    def release(self):
        self._opened = False
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._decklink_input is not None:
            try:
                self._decklink_input.StopStreams()
            except Exception:
                pass
            try:
                self._decklink_input.SetCallback(None)
            except Exception:
                pass
            try:
                self._decklink_input.DisableVideoInput()
            except Exception:
                pass
        self._decklink_input = None
        self._callback = None
        if self._com_initialized:
            try:
                import comtypes
                comtypes.CoUninitialize()
            except Exception:
                pass
            self._com_initialized = False

    def _on_video_frame(self, video_frame):
        capture_actual_start = time.perf_counter()
        try:
            try:
                no_signal_flag = getattr(self._decklink_api, "bmdFrameHasNoInputSource", 0)
                if int(video_frame.GetFlags()) & int(no_signal_flag):
                    return
            except Exception:
                pass
            width = int(video_frame.GetWidth())
            height = int(video_frame.GetHeight())
            row_bytes = int(video_frame.GetRowBytes())
            try:
                pixel_format = video_frame.GetPixelFormat()
            except Exception:
                pixel_format = self._input_pixel_format
            buffer_ptr = video_frame.GetBytes()
            if hasattr(buffer_ptr, "value"):
                buffer_ptr = buffer_ptr.value
            raw = ctypes.string_at(buffer_ptr, row_bytes * height)
            if pixel_format == self._decklink_api.bmdFormat8BitBGRA:
                bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, row_bytes // 4, 4))[:, :width, :]
                rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB).copy()
            elif pixel_format == self._decklink_api.bmdFormat8BitYUV:
                uyvy = np.frombuffer(raw, dtype=np.uint8).reshape((height, row_bytes // 2, 2))[:, :width, :]
                rgb = cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY).copy()
            else:
                return
        except Exception:
            return

        capture_actual_s = time.perf_counter() - capture_actual_start
        with self._frame_ready:
            self._received_frames += 1
            if self._latest_rgb is not None:
                self._overwritten_frames += 1
            self._latest_rgb = rgb
            self._latest_capture_actual_s = capture_actual_s
            self._frame_ready.notify()

    def _on_video_format_changed(self, new_display_mode, detected_signal_flags):
        if self._decklink_input is None or self._decklink_api is None:
            return

        with self._format_lock:
            try:
                display_mode = new_display_mode.GetDisplayMode()
                self.width = int(new_display_mode.GetWidth())
                self.height = int(new_display_mode.GetHeight())
                try:
                    frame_duration, time_scale = new_display_mode.GetFrameRate()
                    if frame_duration:
                        self.fps = float(time_scale) / float(frame_duration)
                except Exception:
                    pass

                rgb_flag = getattr(self._decklink_api, "bmdDetectedVideoInputRGB444", 0)
                if int(detected_signal_flags) & int(rgb_flag):
                    pixel_format = self._decklink_api.bmdFormat8BitBGRA
                else:
                    pixel_format = self._decklink_api.bmdFormat8BitYUV

                self._decklink_input.StopStreams()
                try:
                    self._decklink_input.FlushStreams()
                except Exception:
                    pass
                self._decklink_input.DisableVideoInput()
                self._decklink_input.EnableVideoInput(display_mode, pixel_format, self._input_flags)
                self._input_pixel_format = pixel_format
                with self._frame_ready:
                    self._latest_rgb = None
                    self._frame_ready.notify_all()
                self._decklink_input.StartStreams()
            except Exception:
                pass

