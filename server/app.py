# app.py
from __future__ import annotations

import os

from fastapi.responses import HTMLResponse, RedirectResponse

try:
    from ..models import AdSpendAction, AdSpendObservation
except ImportError:
    from models import AdSpendAction, AdSpendObservation

try:
    from .adspend_env_environment import AdSpendEnvironment
except ImportError:
    from server.adspend_env_environment import AdSpendEnvironment

import uvicorn
from openenv.core.env_server import create_app

# create_app registers:
#   /ws       — WebSocket endpoint for persistent sessions
#   /health   — liveness probe
#   /reset    — POST to reset episode
#   /step     — POST to advance one slot
#   /state    — GET current environment state
#   /web      — browser-based interactive UI
app = create_app(
    AdSpendEnvironment,
    AdSpendAction,
    AdSpendObservation,
    env_name="adspend-personalizer",
)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>AdSpend OpenEnv</title>
        <style>
            :root {
                color-scheme: light;
                --bg: #f4f7fb;
                --panel: #ffffff;
                --text: #102033;
                --muted: #58708a;
                --accent: #0f62fe;
                --border: #d9e2ec;
            }
            body {
                margin: 0;
                font-family: "Segoe UI", Tahoma, sans-serif;
                background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 100%);
                color: var(--text);
            }
            .wrap {
                max-width: 760px;
                margin: 48px auto;
                padding: 24px;
            }
            .card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 28px;
                box-shadow: 0 18px 40px rgba(16, 32, 51, 0.08);
            }
            h1 {
                margin: 0 0 10px;
                font-size: 32px;
            }
            p {
                margin: 0 0 18px;
                color: var(--muted);
                line-height: 1.5;
            }
            .status {
                display: inline-block;
                padding: 8px 12px;
                border-radius: 999px;
                background: #e8f0ff;
                color: var(--accent);
                font-weight: 600;
                margin-bottom: 18px;
            }
            .links {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 12px;
                margin-top: 18px;
            }
            a {
                text-decoration: none;
                color: var(--accent);
                background: #f8fbff;
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 14px 16px;
                font-weight: 600;
                text-align: center;
            }
            a:hover {
                background: #eef5ff;
            }
            .footer {
                margin-top: 18px;
                font-size: 14px;
                color: var(--muted);
            }
        </style>
    </head>
    <body>
        <div class="wrap">
            <div class="card">
                <div class="status">AdSpend OpenEnv is running</div>
                <h1>AdSpend Personalizer</h1>
                <p>
                    The backend is live and ready for OpenEnv interaction. Use the
                    links below to inspect the API, check health, or interact with
                    the environment endpoints.
                </p>
                <div class="links">
                    <a href="/docs">API Docs</a>
                    <a href="/health">Health</a>
                    <a href="/state">State</a>
                    <a href="/web">OpenEnv Web</a>
                </div>
                <div class="footer">
                    Core endpoints: <code>/reset</code>, <code>/step</code>, <code>/state</code>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/web")
def web_home() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=307)


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
