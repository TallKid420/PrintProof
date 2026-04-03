# PrintProof Iris

Iris is the plaque inspection service for Jetson Nano + ArduCam.
It receives an XLSX order file, captures plaque images, extracts fields with Groq vision, and returns pass/fail against expected values.

## What It Does

- Accepts uploaded XLSX files over HTTP.
- Waits for operator confirmation (press Enter in terminal) before processing each upload.
- Captures photos from camera.
- Extracts Name, Title, and Date from plaque image.
- Compares extracted values against expected spreadsheet fields.
- Produces JSON results for each order and batch summary totals.

## Folder Layout

- Autofocus.py: camera autofocus flow.
- Focuser.py: lens focus controls.
- takepicture.py: camera UI capture helper.
- groq.py: image-to-JSON extraction with Groq API.
- runner.py: XLSX parsing, capture + evaluate pipeline.
- server.py: HTTP upload listener.
- Batch/: uploaded files and server output JSON.
- Json/: active normalized orders JSON.
- photos/: captured image files.
- results/: per-run artifacts.

## Requirements

- Python 3.10+ recommended.
- Jetson: OpenCV should be installed with apt for GStreamer camera support.
- Valid Groq API key.

### Jetson system packages

sudo apt update
sudo apt install -y python3-pip python3-opencv

## Python Dependencies

Install from this Iris directory:

pip3 install --user -r requirements.txt

## Configuration

Environment variables:

- GROQ_API_KEY: required for Groq requests.
- GROQ_MODEL: optional, defaults to meta-llama/llama-4-scout-17b-16e-instruct.
- IRIS_SERVER_HOST: optional, defaults to 0.0.0.0.
- IRIS_SERVER_PORT: optional, defaults to 8000.
- IRIS_BATCH_DIR: optional override for Batch path.
- IRIS_JSON_DIR: optional override for Json path.

## Run The Server

From the Iris directory:

python3 server.py

Health endpoint:

- GET /health

Upload endpoint:

- POST /upload with .xlsx file

Server behavior on upload:

1. Saves uploaded XLSX to Batch/uploads.
2. Prompts in terminal: press Enter to start processing.
3. Runs batch capture + verification.
4. Writes final JSON to Batch/orders.json.
5. Returns JSON response with processing result.

## Upload Examples

Multipart upload:

curl -X POST "http://JETSON_IP:8000/upload" -F "file=@orders.xlsx"

Raw body upload:

curl -X POST "http://JETSON_IP:8000/upload" \
	-H "Content-Type: application/octet-stream" \
	-H "X-Filename: orders.xlsx" \
	--data-binary "@orders.xlsx"

## Output Files

- Batch/orders.json: last server processing output.
- Json/orders.json: normalized active orders from spreadsheet.

## Notes

- Current pass/fail check in runner.py is strict exact match on Name, Title, Date.
- If capture is canceled during batch, processing stops early and returns partial progress.
