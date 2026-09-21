"""HTTP wrapper so migrate_data_to_gcs.py can run as a Cloud Run *Service*.

Unlike a Cloud Run Job, a Service must bind $PORT and respond within the startup timeout or
the deployment is marked failed. This binds the port immediately so the deploy always
succeeds, and only runs the (potentially long-running) migration when explicitly triggered:

- GET  /  -> fast health check, no work done, used by Cloud Run's probes.
- POST /  -> runs the migration synchronously and returns a JSON summary.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from migrate_data_to_gcs import DEFAULT_DATA_DIR, migrate


class Handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._respond(200, {"status": "ok"})

    def do_POST(self) -> None:
        bucket = os.environ.get("EXPERT_HUB_GCS_BUCKET", "").strip()
        if not bucket:
            self._respond(500, {"error": "EXPERT_HUB_GCS_BUCKET is required"})
            return
        try:
            summary = migrate(DEFAULT_DATA_DIR, bucket)
        except Exception as error:  # noqa: BLE001 - report the failure to the caller
            self._respond(500, {"error": str(error)})
            return
        self._respond(200, summary)

    def log_message(self, log_format: str, *args: object) -> None:
        print(f"{self.address_string()} - {log_format % args}", flush=True)


def main() -> int:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"listening on :{port} (GET = health check, POST = run migration)", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
