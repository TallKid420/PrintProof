import base64
import json
import os
import urllib.error
import urllib.request
from datetime import datetime

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')


def encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def normalize_order_info(order_info: dict | None) -> dict:
    product_rows = order_info.get('product', []) if isinstance(order_info, dict) else []
    first_row = product_rows[0] if product_rows and isinstance(product_rows[0], dict) else {}
    return {
        'Name': first_row.get('Name') or first_row.get('name') or '',
        'Title': first_row.get('Title') or first_row.get('title') or '',
        'Date': first_row.get('Date') or first_row.get('date') or '',
    }


def build_document_template(document_template: list[str] | None = None) -> list[str]:
    if document_template:
        return document_template

    return [
        'IEEE',
        'In recognition of professional standing',
        'the Officers and Board of Directors of',
        'the IEEE certify that',
        '<expected name>',
        'has been elected to the grade of',
        '<expected title>',
        '<expected date>',
    ]


def analyze_image_against_orders(
    order_info: dict | None,
    image_path: str,
    api_key: str | None = None,
    model: str = GROQ_MODEL,
    save_dir: str = 'results',
    save_json: bool = False,
    temperature: float = 1,
    max_completion_tokens: int = 1024,
    top_p: float = 1,
    document_template: list[str] | None = None,
) -> dict:
    resolved_api_key = api_key or GROQ_API_KEY
    if not resolved_api_key:
        raise ValueError('GROQ_API_KEY is missing. Pass api_key or set GROQ_API_KEY in env.')

    def _error_result(code: str, message: str, details: str | None = None) -> dict:
        payload = {'error': {'code': code, 'message': message, 'details': details}}
        print('Groq error [{}]: {}'.format(code, message))
        if details:
            print(details)

        return {
            'response_text': '',
            'response_json': payload,
            'output_file': None,
        }

    base64_image = encode_image(image_path)
    slot_template = build_document_template(document_template)
    requested_fields = normalize_order_info(order_info)

    requested_fields_json = json.dumps(list(requested_fields.keys()), ensure_ascii=True)
    slot_template_json = json.dumps(slot_template, ensure_ascii=True)
    prompt = f"""Read all visible text from this image.
Use DOCUMENT_TEMPLATE to understand where the variable fields should appear in the document.

Return strict JSON only (no markdown fences) using EXACTLY this object shape and key casing:
{{
    \"exact_text_read\": \"<verbatim text read from the image>\",
    \"Name\": \"<text found where <expected name> appears>\",
    \"Title\": \"<text found where <expected title> appears>\",
    \"Date\": \"<text found where <expected date> appears>\"
}}

Rules:
1) Do not compare against expected values.
2) Use the fixed phrases in DOCUMENT_TEMPLATE to infer where Name, Title, and Date belong.
3) Return only the text actually found in the image for each slot.
4) Preserve spelling exactly as read from the image. Do not normalize or correct.
5) If a field cannot be found, return an empty string for that field.
6) Do not return matched, confidence, reason, pass/fail, or any extra keys.

REQUESTED_FIELDS:\n {requested_fields_json}

DOCUMENT_TEMPLATE:\n {slot_template_json}"""

    payload = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': 'data:image/jpeg;base64,{}'.format(base64_image),
                        },
                    },
                ],
            }
        ],
        'temperature': temperature,
        'max_completion_tokens': max_completion_tokens,
        'top_p': top_p,
        'stream': False,
        'response_format': {'type': 'json_object'},
        'stop': None,
    }

    request = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'PrintProof-Iris/1.0',
            'Authorization': 'Bearer {}'.format(resolved_api_key),
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            completion_payload = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as err:
        details = err.read().decode('utf-8', errors='replace')
        if err.code == 403:
            return _error_result(
                'http_403',
                'Access denied by Groq API. Verify GROQ_API_KEY and account/network access.',
                details,
            )
        return _error_result('http_{}'.format(err.code), 'Groq request failed.', details)
    except urllib.error.URLError as err:
        return _error_result('url_error', 'Groq request failed.', str(err.reason))

    choices = completion_payload.get('choices') or []
    message = choices[0].get('message', {}) if choices else {}
    response_text = message.get('content') or ''

    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        response_json = {'raw_response': response_text}

    return {
        'response_text': response_text,
        'response_json': response_json,
        'output_file': None,
    }
