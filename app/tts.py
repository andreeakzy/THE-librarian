# toate comentariile sunt in romana, fara diacritice si cu litere mici
import pyttsx3
from pathlib import Path
from datetime import datetime

def speak(text: str, save_to_wav: bool = False) -> str | None:
    """
    foloseste pyttsx3 (sapi5 pe windows) pentru a reda textul.
    daca save_to_wav = True, salveaza si intr-un fisier .wav si returneaza calea.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 185)
    out_path = None
    if save_to_wav:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = Path(f"tts-output-{ts}.wav").resolve()
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        return str(out_path)
    else:
        engine.say(text)
        engine.runAndWait()
        return None
