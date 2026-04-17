import sys
sys.setrecursionlimit(5000)   # This fixes the recursion error

from cx_Freeze import setup, Executable

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="EmotionRecognition",
    version="1.0",
    description="Multimodal Emotion Recognition Prototype",
    executables=[Executable("app.py", base=base, target_name="Emotion_Recognition.exe")]
)