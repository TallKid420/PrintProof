import base64
import json
import urllib.error
import urllib.request

OLLAMA_URL = 'http://127.0.0.1:11434/api/generate'
MODEL = 'gemma3:4b'


def _encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _call_ollama(prompt: str, image_path: str = None) -> str:
    payload = {'model': MODEL, 'prompt': prompt, 'stream': False}
    if image_path:
        payload['images'] = [_encode_image(image_path)]

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode('utf-8')).get('response', '').strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print('Ollama error: {}'.format(e))
        return ''


def ProcessImage(image_path: str) -> str:
    """Send image directly to Gemma3:4b — no OCR, LLM reads the text itself."""
    prompt = (
        'Read all text visible in this image exactly as it appears. '
        'Preserve paragraph breaks and line structure. '
        'Do not summarize, interpret, or add anything not visible in the image.'
    )
    result = _call_ollama(prompt, image_path)
    if result:
        print('\nLLM extracted text:\n{}\n'.format(result))
    return result


def ProcessFracturedText(text: str) -> str:
    """Use Gemma3:4b to repair and clean fractured OCR text."""
    prompt = (
        'Clean this OCR output into readable text. '
        'Repair broken words and line wraps. '
        'Do not add information or rewrite meaning. '
        'Preserve paragraph breaks.\n\n' + text
    )
    result = _call_ollama(prompt)
    if result:
        print('\nCleaned text:\n{}\n'.format(result))
    return result


def ProcessImageAndText(image_path: str, text: str) -> str:
    """Send image alongside OCR text; LLM reconciles and corrects errors."""
    prompt = (
        'The following is raw OCR output from the image. '
        'Use the image to correct OCR errors and repair fractured or garbled text. '
        'Return the corrected, readable text only.\n\nOCR output:\n' + text
    )
    result = _call_ollama(prompt, image_path)
    if result:
        print('\nLLM corrected text:\n{}\n'.format(result))
    return result