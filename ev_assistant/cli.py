"""Command-line entry point for E.V. (the `ev` command).

This is also the SSH-facing surface: `ev ask "..."` from any shell on the
machine talks to the already-running daemon over its localhost control API.
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from ev_assistant.config import Config, load_config, write_default_config


def _client_url(cfg: Config, path: str) -> str:
    return f"http://{cfg.control_host}:{cfg.control_port}{path}"


def _auth_headers(cfg: Config) -> dict:
    if not cfg.control_token:
        print(
            "EV_CONTROL_TOKEN is not set in this shell - export the same value the "
            "daemon is running with (see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)
    return {"Authorization": f"Bearer {cfg.control_token}"}


def cmd_init(args: argparse.Namespace) -> None:
    path = write_default_config()
    print(f"Config file: {path}")
    print("Edit it to change personality sliders, wake phrases, news/weather feeds, etc.")
    print()
    print("Still needed before `ev daemon` will start, as environment variables:")
    print("  ANTHROPIC_API_KEY  - https://console.anthropic.com/settings/keys")
    print("  EV_CONTROL_TOKEN   - any random string, e.g.:")
    print('                       python -c "import secrets; print(secrets.token_hex(32))"')
    print()
    print("See .env.example for a template, and the README for Fedora/Windows autostart setup.")


def cmd_daemon(args: argparse.Namespace) -> None:
    from ev_assistant.daemon import run_daemon

    run_daemon()


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = load_config()
    text = " ".join(args.text)
    try:
        response = httpx.post(
            _client_url(cfg, "/ask"),
            json={"text": text, "speak": not args.quiet},
            headers=_auth_headers(cfg),
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.ConnectError:
        print(
            f"Can't reach E.V. at {cfg.control_host}:{cfg.control_port} - "
            "is `ev daemon` running on this machine?",
            file=sys.stderr,
        )
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"E.V. returned an error: {e.response.status_code} {e.response.text}", file=sys.stderr)
        sys.exit(1)
    print(response.json()["reply"])


def cmd_status(args: argparse.Namespace) -> None:
    cfg = load_config()
    try:
        response = httpx.get(_client_url(cfg, "/status"), headers=_auth_headers(cfg), timeout=10.0)
        response.raise_for_status()
    except httpx.ConnectError:
        print(f"E.V. is not running (no response on {cfg.control_host}:{cfg.control_port}).")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"E.V. returned an error: {e.response.status_code} {e.response.text}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(response.json(), indent=2))


def cmd_stop(args: argparse.Namespace) -> None:
    cfg = load_config()
    try:
        httpx.post(_client_url(cfg, "/stop"), headers=_auth_headers(cfg), timeout=10.0).raise_for_status()
    except httpx.ConnectError:
        print("E.V. is not running.")
        return
    print("Stop signal sent.")


def cmd_devices(args: argparse.Namespace) -> None:
    import sounddevice as sd

    for idx, dev in enumerate(sd.query_devices()):
        marker = "in " if dev["max_input_channels"] > 0 else "out"
        print(f"[{idx}] ({marker}) {dev['name']}  (default samplerate: {dev['default_samplerate']:.0f})")
    print("\nSet `input_device` under [audio] in the config to an index or name above if the default mic is wrong.")


def cmd_voices(args: argparse.Namespace) -> None:
    from ev_assistant.audio.tts import Voice

    for v in Voice.list_voices():
        print(f"{v['id']}  -  {v['name']}")
    print("\nSet `voice_id` under [voice] in the config to one of the ids above.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ev", description="E.V. - your voice-activated assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Write the default config file").set_defaults(func=cmd_init)
    sub.add_parser("daemon", help="Run E.V. in the foreground (wake word + control API)").set_defaults(
        func=cmd_daemon
    )

    p_ask = sub.add_parser("ask", help="Send a text question to a running E.V. (works over SSH)")
    p_ask.add_argument("text", nargs="+", help="What to ask")
    p_ask.add_argument(
        "-q", "--quiet", action="store_true", help="Don't speak the reply aloud on the host machine"
    )
    p_ask.set_defaults(func=cmd_ask)

    sub.add_parser("status", help="Show whether E.V. is running and what she knows").set_defaults(
        func=cmd_status
    )
    sub.add_parser("stop", help="Stop a running E.V. daemon").set_defaults(func=cmd_stop)
    sub.add_parser("devices", help="List audio input devices").set_defaults(func=cmd_devices)
    sub.add_parser("voices", help="List available TTS voices").set_defaults(func=cmd_voices)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
