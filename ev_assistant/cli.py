"""Command-line entry point for E.V. (the `ev` command).

Also the SSH-facing surface: `ev ask "..."` from any shell talks to the
running daemon over its localhost (or a configured remote) control API.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import webbrowser

import httpx

from ev_assistant.config import (
    Config,
    config_path,
    env_file_path,
    load_config,
    write_default_config,
)


def _client_url(cfg: Config, path: str) -> str:
    return f"http://{cfg.api_host}:{cfg.api_port}{path}"


def _auth_headers(cfg: Config) -> dict:
    if not cfg.control_token:
        print(
            f"EV_CONTROL_TOKEN isn't set. Run `ev init` to generate one into {env_file_path()}, "
            "or export it in this shell.",
            file=sys.stderr,
        )
        sys.exit(1)
    return {"Authorization": f"Bearer {cfg.control_token}"}


# -- setup ------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> None:
    cfg_file = write_default_config()
    env_file = env_file_path()
    env_file.parent.mkdir(parents=True, exist_ok=True)

    created_token = False
    existing = {}
    if env_file.exists():
        from ev_assistant.config import read_env_file

        existing = read_env_file(env_file)
    if not existing.get("EV_CONTROL_TOKEN"):
        token = secrets.token_hex(32)
        api_key = existing.get("ANTHROPIC_API_KEY", "sk-ant-your-key-here")
        env_file.write_text(
            f"ANTHROPIC_API_KEY={api_key}\nEV_CONTROL_TOKEN={token}\n", encoding="utf-8"
        )
        try:
            env_file.chmod(0o600)
        except OSError:
            pass
        created_token = True

    print(f"Config file: {cfg_file}")
    print(f"Secrets file: {env_file}")
    if created_token:
        print("  Generated a fresh EV_CONTROL_TOKEN.")
    print()
    print(f"Before starting, edit {env_file} and set ANTHROPIC_API_KEY")
    print("  (get a key at https://console.anthropic.com/settings/keys).")
    print("  E.V. can also run fully offline - set offline.mode = \"offline\" in the config.")
    print()
    print("Then: systemctl --user start ev-assistant   (or just `ev daemon`)")


def cmd_daemon(args: argparse.Namespace) -> None:
    from ev_assistant.daemon import run_daemon

    run_daemon()


# -- talking to a running daemon --------------------------------------


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = load_config()
    text = " ".join(args.text)
    try:
        response = httpx.post(
            _client_url(cfg, "/ask"),
            json={"text": text, "speak": not args.quiet, "allow_destructive": args.yes},
            headers=_auth_headers(cfg),
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.ConnectError:
        print(
            f"Can't reach E.V. at {cfg.api_host}:{cfg.api_port} - is `ev daemon` running?",
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
        print(f"E.V. is not running (no response on {cfg.api_host}:{cfg.api_port}).")
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


def cmd_gui(args: argparse.Namespace) -> None:
    cfg = load_config()
    if not cfg.control_token:
        print("No EV_CONTROL_TOKEN - run `ev init` first.", file=sys.stderr)
        sys.exit(1)
    url = f"http://{cfg.api_host}:{cfg.api_port}/?token={cfg.control_token}"
    print(f"Opening E.V.'s console: {url}")
    if not args.no_open:
        webbrowser.open(url)


# -- diagnostics ------------------------------------------------------


def cmd_devices(args: argparse.Namespace) -> None:
    import sounddevice as sd

    for idx, dev in enumerate(sd.query_devices()):
        marker = "in " if dev["max_input_channels"] > 0 else "out"
        print(f"[{idx}] ({marker}) {dev['name']}  (default samplerate: {dev['default_samplerate']:.0f})")
    print("\nSet `input_device` under [audio] in the config to an index or name above.")


def cmd_mic_test(args: argparse.Namespace) -> None:
    from ev_assistant.audio.model_setup import ensure_model
    from ev_assistant.audio.stt import record_command
    from ev_assistant.audio.wake_word import match_wake
    from vosk import Model

    cfg = load_config()
    print("Preparing speech model (first run downloads it)...")
    ensure_model(cfg.vosk_model_dir)
    model = Model(str(cfg.vosk_model_dir))

    print("Say something now (E.V. will show the input level and transcribe it).")
    print("Try: \"E.V., can you hear me\"\n")

    def bar(level: float) -> None:
        filled = int(level * 40)
        sys.stdout.write("\r  [" + "#" * filled + "-" * (40 - filled) + f"] {level:4.2f}")
        sys.stdout.flush()

    text = record_command(
        model,
        silence_timeout_s=1.5,
        max_duration_s=12.0,
        lead_grace_s=6.0,
        device=cfg.input_device,
        on_level=bar,
    )
    print("\n")
    if not text:
        print("Heard no speech. If the level bar stayed near 0.00, the wrong mic is selected -")
        print("run `ev devices` and set `input_device` in the config, then try again.")
        sys.exit(1)
    print(f"Transcribed: {text!r}")
    match = match_wake(text, cfg.wake_names, cfg.wake_prefixes)
    if match is not None:
        got = f" with command {match.command!r}" if match.command else ""
        print(f"Wake word DETECTED{got}. Microphone and wake word are working.")
    else:
        print("Wake word not detected in that phrase - but the mic works. Say one of:")
        print("  " + ", ".join(cfg.wake_names))


# -- voices -----------------------------------------------------------


def cmd_voices(args: argparse.Namespace) -> None:
    if args.online:
        from ev_assistant.audio.tts import list_online_voices

        locale = None if args.all else "en-AU"
        voices = list_online_voices(filter_locale=locale)
        if not voices:
            print("No online voices found (is edge-tts installed and are you online?).")
            return
        header = "Australian" if locale else "all"
        print(f"Online ({header}) neural voices - set voice.edge_voice in the config:")
        for v in voices:
            print(f"  {v}")
        return
    if args.install_piper:
        _install_piper_voice()
        return
    # Default: what's active + how to see more.
    cfg = load_config()
    print(f"Active voice engine: {cfg.voice_engine}")
    print(f"  online voice (edge): {cfg.edge_voice}")
    print(f"  offline fallback (espeak): {cfg.espeak_voice}")
    print("\nSee Australian online voices:  ev voices --online")
    print("Install an offline neural voice: ev voices --install-piper")


def _install_piper_voice() -> None:
    import httpx as _httpx

    cfg = load_config()
    cfg.piper_dir.mkdir(parents=True, exist_ok=True)
    base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium"
    files = {
        "en_GB-jenny_dioco-medium.onnx": f"{base}/en_GB-jenny_dioco-medium.onnx",
        "en_GB-jenny_dioco-medium.onnx.json": f"{base}/en_GB-jenny_dioco-medium.onnx.json",
    }
    print("Downloading an offline neural voice (British female - no Australian offline model exists).")
    for name, url in files.items():
        dest = cfg.piper_dir / name
        if dest.exists():
            continue
        try:
            with _httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 16):
                        f.write(chunk)
        except Exception as e:
            print(f"  Failed to download {name}: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"Installed to {cfg.piper_dir}. Set voice.engine = \"piper\" (or leave \"auto\") to use it.")
    print("You also need the piper binary: pip install piper-tts")


# -- learning ---------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> None:
    """Bundle E.V.'s brain (config, memory, knowledge) into one file to move."""
    import tarfile
    from pathlib import Path

    cfg = load_config()
    out = Path(args.path)
    members = [
        (config_path(), "config.toml"),
        (env_file_path(), "env"),
        (cfg.db_path, "memory.sqlite3"),
        (cfg.knowledge_path, "knowledge.sqlite3"),
    ]
    with tarfile.open(out, "w:gz") as tar:
        for src, arcname in members:
            if src.exists():
                tar.add(src, arcname=arcname)
    print(f"Exported E.V.'s brain to {out}.")
    print("Copy it to the other PC and run:  ev import " + str(out))
    print("Note: the `env` file holds your API key and token - keep the archive private.")


def cmd_import(args: argparse.Namespace) -> None:
    import tarfile
    from pathlib import Path

    cfg = load_config()
    src = Path(args.path)
    if not src.is_file():
        print(f"No such file: {src}", file=sys.stderr)
        sys.exit(1)
    targets = {
        "config.toml": config_path(),
        "env": env_file_path(),
        "memory.sqlite3": cfg.db_path,
        "knowledge.sqlite3": cfg.knowledge_path,
    }
    config_path().parent.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src, "r:gz") as tar:
        for member in tar.getmembers():
            dest = targets.get(member.name)
            if dest is None or not member.isfile():
                continue  # ignore anything unexpected in the archive
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            dest.write_bytes(extracted.read())
    try:
        env_file_path().chmod(0o600)
    except OSError:
        pass
    print("Imported E.V.'s brain. Restart the daemon (or start it) to use it.")


def cmd_learn(args: argparse.Namespace) -> None:
    from pathlib import Path

    from ev_assistant import ingest
    from ev_assistant.knowledge import Knowledge

    cfg = load_config()
    kb = Knowledge(cfg.knowledge_path)
    try:
        if args.wikipedia:
            title, text = ingest.from_wikipedia(args.wikipedia)
        elif args.url:
            title, text = ingest.from_url(args.url)
        elif args.file:
            title, text = ingest.from_file(Path(args.file))
        else:
            print("Give one of --wikipedia, --url, or --file.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Couldn't learn that: {e}", file=sys.stderr)
        sys.exit(1)
    added = kb.add_document(title, args.wikipedia or args.url or args.file, text)
    print(f"Learned \"{title}\" - {added} passages added. Total: {kb.passage_count()}.")


# -- argument parsing -------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ev", description="E.V. - your voice-activated assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Write default config and generate the control token").set_defaults(func=cmd_init)
    sub.add_parser("daemon", help="Run E.V. in the foreground (wake word + API + GUI)").set_defaults(func=cmd_daemon)

    p_ask = sub.add_parser("ask", help="Send a text question/command to a running E.V. (works over SSH)")
    p_ask.add_argument("text", nargs="+", help="What to ask or tell her to do")
    p_ask.add_argument("-q", "--quiet", action="store_true", help="Don't speak the reply aloud on the host")
    p_ask.add_argument("-y", "--yes", action="store_true", help="Pre-approve destructive actions this request")
    p_ask.set_defaults(func=cmd_ask)

    sub.add_parser("status", help="Show whether E.V. is running and what she knows").set_defaults(func=cmd_status)
    sub.add_parser("stop", help="Stop a running E.V. daemon").set_defaults(func=cmd_stop)

    p_gui = sub.add_parser("gui", help="Open E.V.'s cyberpunk console in your browser")
    p_gui.add_argument("--no-open", action="store_true", help="Print the URL but don't open a browser")
    p_gui.set_defaults(func=cmd_gui)

    sub.add_parser("devices", help="List audio input devices").set_defaults(func=cmd_devices)
    sub.add_parser("mic-test", help="Check the microphone and wake word are working").set_defaults(func=cmd_mic_test)

    p_voices = sub.add_parser("voices", help="Show/list/install TTS voices")
    p_voices.add_argument("--online", action="store_true", help="List online neural voices")
    p_voices.add_argument("--all", action="store_true", help="With --online, list every locale, not just en-AU")
    p_voices.add_argument("--install-piper", action="store_true", help="Download an offline neural voice")
    p_voices.set_defaults(func=cmd_voices)

    p_learn = sub.add_parser("learn", help="Teach E.V. something for her offline knowledge base")
    g = p_learn.add_mutually_exclusive_group()
    g.add_argument("--wikipedia", metavar="TITLE", help="Learn a Wikipedia article")
    g.add_argument("--url", metavar="URL", help="Learn a web page")
    g.add_argument("--file", metavar="PATH", help="Learn a local .txt/.md/.pdf file")
    p_learn.set_defaults(func=cmd_learn)

    p_export = sub.add_parser("export", help="Bundle E.V.'s config/memory/knowledge to move to another PC")
    p_export.add_argument("path", help="Output file, e.g. ev-brain.tar.gz")
    p_export.set_defaults(func=cmd_export)

    p_import = sub.add_parser("import", help="Restore an exported E.V. brain on this PC")
    p_import.add_argument("path", help="The exported .tar.gz to import")
    p_import.set_defaults(func=cmd_import)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
