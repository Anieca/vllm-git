"""Caption an image through the OpenAI-compatible API of `vllm serve`."""

import argparse
import base64
import io

from openai import OpenAI
from vllm.assets.image import ImageAsset

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:8000/v1")
parser.add_argument("--prompt", default="")
parser.add_argument("--image", default="stop_sign")
args = parser.parse_args()

buf = io.BytesIO()
ImageAsset(args.image).pil_image.convert("RGB").save(buf, format="PNG")
image_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

client = OpenAI(base_url=args.base_url, api_key="EMPTY")
model = client.models.list().data[0].id
content = [{"type": "image_url", "image_url": {"url": image_url}}]
if args.prompt:
    content.append({"type": "text", "text": args.prompt})
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": content}],
    max_tokens=20,
    temperature=0.0,
)
print(repr(resp.choices[0].message.content))
