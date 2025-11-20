import rumps
from pathlib import Path
from threading import Thread

class RecorderApp(rumps.App):
    def __init__(self, out_base: Path, cycle: int, capture_once):
        super().__init__("Rec")
        self.out_base = out_base
        self.cycle = cycle
        self.capture_once = capture_once
        self.timer = rumps.Timer(self.tick, self.cycle)
        self.menu = ["Pause", "Resume"]
        self.title = "Rec"

    def tick(self, _):
        Thread(target=self._do_capture, daemon=True).start()

    def _do_capture(self):
        try:
            img_path = self.capture_once(self.out_base)
        finally:
            self.title = "Rec"

    @rumps.clicked("Pause")
    def pause(self, _):
        self.timer.stop()
        self.title = "Paused"

    @rumps.clicked("Resume")
    def resume(self, _):
        self.timer.start()
        self.title = "Rec"

def run_menubar(out_base: Path, cycle: int, capture_once):
    app = RecorderApp(out_base, cycle, capture_once)
    app.timer.start()
    app.run()