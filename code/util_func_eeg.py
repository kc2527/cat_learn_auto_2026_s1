class EEGPort:

    def __init__(self,
                 win,
                 address=None,
                 enabled=False,
                 default_ms=10):
        self.win = win
        self.enabled = enabled
        self.default_ms = default_ms
        self._port = None
        self._clear_at = None
        if not self.enabled:
            return
        try:
            from psychopy import parallel  # type: ignore
            self._port = parallel.ParallelPort(address=address)
        except Exception as e:
            print(
                f"[EEG] Parallel port unavailable ({e}). Running without triggers."
            )
            self.enabled = False
            self._port = None

    def flip_pulse(self, code, width_ms=None, global_clock=None):
        """Schedule a flip-locked pulse: set code on next win.flip, clear after width_ms."""
        if not (self.enabled and self._port):
            return
        width_ms = self.default_ms if width_ms is None else width_ms
        # rising edge exactly on next flip:
        self.win.callOnFlip(self._port.setData, int(code) & 0xFF)
        # schedule a timed clear to 0 after the flip:
        if global_clock is not None:
            # record when to clear (relative to global clock)
            self._clear_at = global_clock.getTime() + (width_ms / 1000.0)

    def pulse_now(self, code, width_ms=None, global_clock=None):
        """Immediate pulse (not flip-locked) -- useful for response events."""
        if not (self.enabled and self._port):
            return
        width_ms = self.default_ms if width_ms is None else width_ms
        self._port.setData(int(code) & 0xFF)
        if global_clock is not None:
            self._clear_at = global_clock.getTime() + (width_ms / 1000.0)

    def update(self, global_clock=None):
        """Call every frame: clears the port to 0 if a pulse has expired."""
        if not (self.enabled and self._port):
            return
        if self._clear_at is not None and global_clock is not None:
            if global_clock.getTime() >= self._clear_at:
                self._port.setData(0)
                self._clear_at = None

    def close(self):
        try:
            if self._port:
                self._port.setData(0)
        except Exception:
            pass
