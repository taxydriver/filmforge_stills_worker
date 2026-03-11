from __future__ import annotations

import os
import time
import json
import uuid
from typing import Any, Dict, Optional
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ComfyRuntime:
    def __init__(self, base_url: Optional[str] = None, timeout_seconds: int = 20):
        self.base_url = (base_url or os.getenv("COMFY_BASE") or "http://127.0.0.1:8188").rstrip("/")
        self.timeout_seconds = int(timeout_seconds)

    def _json_request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                if not body:
                    return {}
                return json.loads(body)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail[:300]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Connection failed {method} {path}: {exc}") from exc

    def wait_until_ready(self, timeout_seconds: int = 180) -> None:
        deadline = time.time() + timeout_seconds
        last_error: Optional[str] = None
        while time.time() < deadline:
            try:
                _ = self._json_request("GET", "/object_info")
                if _ is not None:
                    return
            except Exception as exc:  # pragma: no cover - runtime dependent
                last_error = str(exc)
            time.sleep(1.0)
        raise RuntimeError(f"Comfy runtime not ready at {self.base_url}: {last_error}")

    def get_object_info(self) -> Dict[str, Any]:
        data = self._json_request("GET", "/object_info")
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected /object_info payload type: {type(data)}")
        return data

    def submit_workflow(self, workflow: Dict[str, Any], client_id: str = "filmforge-stills-worker") -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        data = self._json_request("POST", "/prompt", payload=payload)
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected /prompt response type: {type(data)}")
        prompt_id = data.get("prompt_id") or data.get("promptId")
        if not prompt_id:
            raise RuntimeError(f"No prompt_id in /prompt response: {data}")
        return str(prompt_id)

    def poll_history_entry(
        self,
        prompt_id: str,
        timeout_seconds: int = 300,
        poll_interval_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout_seconds
        last_payload: Any = None
        while time.time() < deadline:
            try:
                data = self._json_request("GET", f"/history/{quote(prompt_id)}")
            except Exception as exc:
                last_payload = str(exc)
                time.sleep(poll_interval_seconds)
                continue

            last_payload = data
            if not isinstance(data, dict):
                time.sleep(poll_interval_seconds)
                continue

            entry = data.get(prompt_id)
            if not isinstance(entry, dict) and data:
                maybe_entry = next(iter(data.values()))
                entry = maybe_entry if isinstance(maybe_entry, dict) else None

            if isinstance(entry, dict) and isinstance(entry.get("outputs"), dict):
                return entry

            time.sleep(poll_interval_seconds)

        raise TimeoutError(
            f"Timeout waiting for prompt_id={prompt_id}. Last history payload={last_payload}"
        )

    def fetch_image_bytes(self, filename: str, *, subfolder: str = "", image_type: str = "output") -> bytes:
        url = f"{self.base_url}/view?filename={quote(filename)}&type={quote(image_type)}"
        if subfolder:
            url += f"&subfolder={quote(subfolder)}"
        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=max(30, self.timeout_seconds)) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code} GET /view: {detail[:300]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Connection failed GET /view: {exc}") from exc

    def upload_input_image(
        self,
        *,
        image_bytes: bytes,
        filename: str = "ref.png",
        overwrite: bool = True,
    ) -> str:
        url = f"{self.base_url}/upload/image"
        fields = {
            "type": "input",
            "overwrite": "true" if overwrite else "false",
        }
        boundary = f"----filmforge-{uuid.uuid4().hex}"
        body_chunks = []

        for key, value in fields.items():
            body_chunks.append(f"--{boundary}\r\n".encode("utf-8"))
            body_chunks.append(
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
            )
            body_chunks.append(str(value).encode("utf-8"))
            body_chunks.append(b"\r\n")

        body_chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        body_chunks.append(
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body_chunks.append(b"Content-Type: image/png\r\n\r\n")
        body_chunks.append(image_bytes)
        body_chunks.append(b"\r\n")
        body_chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(body_chunks)

        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        req = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=max(30, self.timeout_seconds)) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code} POST /upload/image: {detail[:300]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Connection failed POST /upload/image: {exc}") from exc

        try:
            payload = json.loads(raw) if raw else {}
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON from /upload/image: {raw[:200]}") from exc

        uploaded_name = payload.get("name") or payload.get("filename") or filename
        return str(uploaded_name)
