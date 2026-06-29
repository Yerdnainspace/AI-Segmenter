import threading


def run_with_timeout(func, fallback, timeout=6.0):
    result = {"value": fallback}

    def worker():
        try:
            result["value"] = func()
        except Exception:
            result["value"] = fallback

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=float(timeout))
    return result["value"] if not thread.is_alive() else fallback

