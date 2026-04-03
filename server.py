import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import runner


BASE_DIR = Path(__file__).resolve().parent
BATCH_DIR = Path(os.environ.get("IRIS_BATCH_DIR", str(BASE_DIR / "Batch"))).resolve()
UPLOADS_DIR = BATCH_DIR / "uploads"
OUTPUT_JSON = BATCH_DIR / "orders.json"
HOST = os.environ.get("IRIS_SERVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("IRIS_SERVER_PORT", "8000"))


def _safe_filename(name: str) -> str:
	cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name or "orders.xlsx")
	if not cleaned.lower().endswith(".xlsx"):
		cleaned += ".xlsx"
	return cleaned


def _extract_uploaded_xlsx(handler: BaseHTTPRequestHandler) -> tuple[bytes | None, str | None, str]:
	content_type = handler.headers.get("Content-Type", "")
	content_length = int(handler.headers.get("Content-Length", "0"))
	if content_length <= 0:
		return None, "Request body is empty.", "orders.xlsx"

	body = handler.rfile.read(content_length)

	if "multipart/form-data" in content_type:
		boundary_match = re.search(r"boundary=([^;]+)", content_type)
		if not boundary_match:
			return None, "Missing multipart boundary.", "orders.xlsx"

		boundary = boundary_match.group(1).strip().strip('"').encode("utf-8")
		parts = body.split(b"--" + boundary)
		for part in parts:
			if b"Content-Disposition" not in part or b"filename=" not in part:
				continue

			header_end = part.find(b"\r\n\r\n")
			if header_end == -1:
				continue

			headers_blob = part[:header_end]
			data = part[header_end + 4 :]
			data = data.rstrip(b"\r\n")

			filename_match = re.search(br'filename="([^"]+)"', headers_blob)
			filename = (
				filename_match.group(1).decode("utf-8", errors="replace")
				if filename_match
				else "orders.xlsx"
			)

			if not filename.lower().endswith(".xlsx"):
				return None, "Only .xlsx uploads are supported.", filename

			return data, None, filename

		return None, "No file part with filename found in multipart body.", "orders.xlsx"

	filename = handler.headers.get("X-Filename", "orders.xlsx")
	if not filename.lower().endswith(".xlsx"):
		return None, "Only .xlsx uploads are supported.", filename

	return body, None, filename


def _process_uploaded_file(xlsx_path: Path) -> dict:
	print("Received upload: {}".format(xlsx_path))
	print("Press Enter in this terminal to start processing...")
	input()

	runner_json = runner.run_all(str(xlsx_path), use_autofocus=False)
	if runner_json is None:
		runner_json = {
			"error": "runner returned no result",
			"input_file": str(xlsx_path),
		}

	OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
	with OUTPUT_JSON.open("w", encoding="utf-8") as f:
		json.dump(runner_json, f, indent=2, ensure_ascii=True, default=str)

	print("Saved server output to {}".format(OUTPUT_JSON))
	return runner_json


class UploadHandler(BaseHTTPRequestHandler):
	def _write_json(self, status_code: int, payload: dict) -> None:
		response = json.dumps(payload, ensure_ascii=True, default=str).encode("utf-8")
		self.send_response(status_code)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(response)))
		self.end_headers()
		self.wfile.write(response)

	def do_GET(self) -> None:
		if self.path in ("/", "/health"):
			self._write_json(
				200,
				{
					"status": "ok",
					"message": "POST .xlsx to /upload",
				},
			)
			return

		self._write_json(404, {"error": "Not found"})

	def do_POST(self) -> None:
		if self.path != "/upload":
			self._write_json(404, {"error": "Use POST /upload"})
			return

		file_bytes, error, filename = _extract_uploaded_xlsx(self)
		if error:
			self._write_json(400, {"error": error})
			return

		if file_bytes is None:
			self._write_json(400, {"error": "No file uploaded."})
			return

		UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
		safe_name = _safe_filename(filename)
		upload_path = UPLOADS_DIR / safe_name
		upload_path.write_bytes(file_bytes)

		result = _process_uploaded_file(upload_path)
		self._write_json(
			200,
			{
				"message": "Processed uploaded XLSX",
				"uploaded_file": str(upload_path),
				"output_file": str(OUTPUT_JSON),
				"result": result,
			},
		)


def main() -> None:
	server = HTTPServer((HOST, PORT), UploadHandler)
	print("Listening on http://{}:{}".format(HOST, PORT))
	print("Send POST /upload with an .xlsx file.")
	print("For each upload, processing starts only after Enter is pressed in this terminal.")
	server.serve_forever()


if __name__ == "__main__":
	main()