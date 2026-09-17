import os
import socket

import uvicorn


def find_available_port(host: str, start_port: int, max_tries: int = 20) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError("No available application port")


if __name__ == "__main__":
    host = os.getenv("CARDIAC_APP_HOST", "127.0.0.1")
    port = find_available_port(host, int(os.getenv("CARDIAC_APP_PORT", "8010")))
    print(f"Open http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
