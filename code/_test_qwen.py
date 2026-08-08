import _runtime_guard  # noqa
from PIL import Image
import io, base64, json, urllib.request, urllib.error, sys, traceback

img = Image.open(r"C:\Users\TS\WorkBuddy\HydroGelNet\figures\Figure3_dataset.png")
img.thumbnail((1400, 1400))
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
print("img size:", img.size, "b64 len:", len(b64), flush=True)

payload = {
    "model": "qwen36:latest",
    "messages": [{
        "role": "user",
        "content": "Describe this scientific figure briefly. Is text overlapping?",
        "images": [b64],
    }],
    "think": False,
    "stream": False,
}
try:
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=600))
    print("RESPONSE OK:", resp.get("message", {}).get("content", "")[:600], flush=True)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
