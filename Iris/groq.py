import base64
import json
import os
import urllib.error
import urllib.request

GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')


def _encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _call_groq(prompt: str, image_path: str = None) -> str:
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        print('Groq fallback unavailable: GROQ_API_KEY is not set')
        return ''

    content = [{'type': 'text', 'text': prompt}]
    if image_path:
        content.append({
            'type': 'image_url',
            'image_url': {
                'url': 'data:image/png;base64,{}'.format(_encode_image(image_path))
            },
        })

    payload = {
        'model': GROQ_MODEL,
        'messages': [{'role': 'user', 'content': content}],
        'temperature': 0.2,
    }
    request = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(api_key),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode('utf-8'))
            choices = payload.get('choices') or []
            if not choices:
                return ''
            message = choices[0].get('message') or {}
            return (message.get('content') or '').strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print('Groq error: {}'.format(e))
        return ''


def ProcessImage(image_path: str, prompt: str) -> str:
    return _call_groq(prompt, image_path)


def ProcessText(prompt: str) -> str:
    return _call_groq(prompt)


def ProcessImageAndText(image_path: str, prompt: str) -> str:
    return _call_groq(prompt, image_path)