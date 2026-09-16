# E.V.

A voice-activated personal assistant that runs on your own machine. Say
"Hey E.V." and she listens, thinks (via Claude), and answers out loud -
no screen, no app, just a voice, in the spirit of a dry-witted, blunt
sci-fi assistant (think TARS' candor) rather than a hype-machine chatbot.
She's software only: no camera, no physical form, no moving parts.

Read this whole page before installing - the "What 'getting smarter'
actually means" and "SSH access, honestly" sections below set expectations
that the rest of the setup depends on.

## What this actually is

- **Wake word:** an offline speech model (Vosk) listens continuously and
  watches for "Hey E.V." (and a few phonetic variants). Nothing you say is
  sent anywhere until the wake phrase is heard.
- **Brain:** once triggered, your command is transcribed locally and sent
  as text to Claude (Anthropic's API) to generate a reply, which is then
  spoken back to you via offline text-to-speech.
- **Memory:** conversations and background-fetched facts (news headlines,
  weather) are stored in a local SQLite database and fed back to Claude as
  context on later questions.
- **Control API:** a small localhost-only web server lets you talk to E.V.
  as text from any shell on the machine - including over SSH.

### What "getting smarter over time" actually means

This does **not** retrain or fine-tune any model on your machine - that
would be a bad idea for a personal project (expensive, slow, and easy to
get subtly wrong) and this repo doesn't pretend to do it. What actually
happens: a background loop periodically pulls fresh data (RSS feeds you
configure, optionally local weather) into E.V.'s memory, and everything
you tell her gets remembered too. Claude then draws on whatever's
relevant from that growing memory when it answers. That's a real and
useful effect - she'll know about a headline from an hour ago, or
something you told her yesterday - just not literal model retraining.

### SSH access, honestly

`ev ask "..."` from an SSH session talks to the same running brain and
memory as the voice loop, and she'll still speak the answer out loud *on
the physical machine* (use `--quiet` to suppress that). What SSH does
**not** give you is your remote terminal's microphone or speakers - audio
hardware access only works for whoever's physically at that machine (or
logged into its desktop session). The control API is bound to
`127.0.0.1` only, on purpose - see [Security](#security) below.

## Prerequisites

- An Anthropic API key: <https://console.anthropic.com/settings/keys>
- Python 3.11+ (Fedora: usually already installed; Windows: only needed
  for the one-time build step, not to run the final `.exe`)

## Fedora setup

```bash
git clone <this repo> && cd specter
bash scripts/install-fedora.sh
```

This installs `portaudio`, `espeak-ng`, and `alsa-utils` via `dnf`
(you'll be prompted for `sudo`), creates a virtualenv under
`~/.local/share/ev-assistant`, writes a default config, generates a
control-API token, and registers a `systemd --user` service so E.V.
starts automatically whenever you log in (she needs your desktop's audio
session, so this is a **user** service, not a system-wide one - running
her as root or via a system unit won't have access to your microphone).

Then:

1. Edit `~/.config/ev-assistant/env` and set `ANTHROPIC_API_KEY`.
2. `systemctl --user start ev-assistant`
3. `journalctl --user -u ev-assistant -f` to watch it start (the first
   run downloads a ~40MB offline speech model).
4. Say "Hey E.V., I need help" and wait for her acknowledgement.

To remove everything later: `bash scripts/uninstall-fedora.sh`.

## Windows setup

PyInstaller can't cross-compile, so the `.exe` has to be built **on your
Windows machine** - there's no way around that from here. From PowerShell,
with Python 3.11+ installed:

```powershell
git clone <this repo>; cd specter
.\scripts\build-windows.ps1     # builds dist\ev\ev.exe
.\scripts\install-windows.ps1   # installs it + registers a Task Scheduler autostart entry
```

Then edit the printed `ev.env` file to set `ANTHROPIC_API_KEY`, and either
log off/on or run `Start-ScheduledTask -TaskName EV-Assistant`.

**Honesty note:** the packaging scripts (`ev.spec`, `build-windows.ps1`,
`install-windows.ps1`) were written in a Linux container with no Windows
machine available to actually build and run on, based on well-documented
PyInstaller behavior for this dependency set. See
[Troubleshooting](#troubleshooting) below for the likely failure modes and
how to fix them if the build doesn't come out clean on the first try.

To remove everything later: `.\scripts\uninstall-windows.ps1`.

## Talking to her

| From | How |
|---|---|
| Voice, at the machine | Say "Hey E.V." (or "...I need help"), wait for "Go ahead.", then speak your question. |
| SSH, or any local shell | `ev ask "what's the weather like"` (add `-q`/`--quiet` to skip speaking the reply aloud) |
| Check she's alive | `ev status` |
| Shut her down | `ev stop` (the systemd/Task Scheduler entry will restart her unless you also `systemctl --user stop ev-assistant` / stop the scheduled task) |
| Pick the right mic | `ev devices`, then set `input_device` in the config |
| Pick a TTS voice | `ev voices`, then set `voice_id` under `[voice]` in the config |

`ev ask`/`status`/`stop` all need `EV_CONTROL_TOKEN` set in your shell to
the same value the daemon is running with (it's in the `env`/`ev.env`
file the install script generated).

## Configuration

Edit the config file directly - `ev init` (run automatically by both
install scripts) writes a commented default at:

- Fedora: `~/.config/ev-assistant/config.toml`
- Windows: `%LOCALAPPDATA%\ev-assistant\config.toml`

Restart the service/task after editing it. Notable settings:

- `[brain] model` - defaults to `claude-opus-5`. Switch to
  `claude-sonnet-5` if the voice round-trip feels slow; it's cheaper and
  faster, at somewhat less depth.
- `[brain] effort` - `low` by default, tuned for a snappy live
  conversation. Raise it (`medium`/`high`) if you want more careful
  answers and don't mind a longer pause before she replies.
- `[personality] humor` / `honesty` (0-100) - shapes E.V.'s tone; see
  `ev_assistant/personality.py` for exactly how. High honesty means she'll
  tell you plainly when she doesn't know something instead of guessing.
- `[wake_word] phrases` - the list of phrases that trigger her (a few
  phonetic spellings of "Hey E.V." are included by default, since speech
  recognizers vary in how they spell it out).
- `[data_feeds] feeds` / `weather_location` - what she pulls into memory
  in the background, and how often (`interval_minutes`).

## Security

- The control API only binds to `127.0.0.1` and requires the
  `EV_CONTROL_TOKEN` bearer token on every request - reaching it requires
  a shell on the machine itself (SSH counts) or a port-forward you set up
  yourself. It is never exposed to your network by default, and that's
  deliberate: turning that around would let anyone on your LAN issue
  commands to (and hear responses from) your assistant.
- Feed content (RSS headlines, etc.) is passed to Claude as labeled data
  ("Things E.V. currently knows"), never as instructions, and this version
  of E.V. has no tool-use or action-taking capability - so even a
  malicious feed attempting a prompt-injection-style headline has nothing
  to actually do beyond producing a weird spoken sentence. Still, only
  point `feeds` at sources you trust.
- Your `.env` / `env` / `ev.env` file holds your Anthropic API key in
  plaintext - it's `.gitignore`d here and the install scripts `chmod 600`
  it on Fedora, but treat it like any other credential.

## Troubleshooting

**"PortAudio library not found"** (Fedora) - `sudo dnf install portaudio`.

**No sound / `RuntimeError: ... eSpeak ... not installed`** (Fedora) -
`sudo dnf install espeak-ng alsa-utils` (both are installed by
`install-fedora.sh`, but if you're running outside that script, install
them yourself).

**Wake word never triggers, or triggers constantly** - run `ev devices`
and set `input_device` in the config to your actual microphone; the
system default device is sometimes the wrong one (e.g. a webcam mic vs. a
headset). You can also loosen/tighten detection by adjusting the phrase
list in `[wake_word] phrases`, and endpointing (when she decides you've
stopped talking) via `silence_timeout_s`.

**"Invalid sample rate" from sounddevice** - your microphone doesn't
support 16kHz capture directly; try a different `input_device`, or a USB
headset, which almost always supports it.

**`ev.exe` fails at startup with `ModuleNotFoundError`** (Windows) - a
PyInstaller hidden import is missing for your exact dependency versions.
Open `scripts/ev.spec`, and add the missing module name to
`hiddenimports`, then rebuild. `uvicorn`, `pyttsx3`'s driver loading, and
`vosk`/`sounddevice`'s native libraries are the most likely spots (the
spec already covers the known cases - see the comment at the top of that
file).

**A console window briefly flashes when E.V. auto-starts on Windows** -
this is the console-subsystem `ev.exe` process; the scheduled task
launches it through a hidden PowerShell wrapper specifically to suppress
this, but Windows' behavior here varies by version. If it bothers you,
you can rebuild with `console=False` in `ev.spec` - but then `ev.exe ask`
run manually from a terminal won't print output either, so you'd lose
normal CLI usability for the sake of a cosmetic flash.

**Claude API errors** (`ANTHROPIC_API_KEY` missing/invalid, rate limits,
network) - E.V. will say so out loud rather than going silent; check
`journalctl --user -u ev-assistant -f` (Fedora) for the full error.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The test suite covers config loading, memory/retrieval, the personality
prompt builder, the Claude error-handling paths (mocked, no real API
calls), the control API's auth and routes, and RSS/weather ingestion - all
without needing a microphone, speakers, or a real API key. Also verified
directly in the sandbox this was built in: every module imports cleanly,
the CLI's non-network commands behave correctly end-to-end, and TTS
actually speaks (via `espeak-ng`) once the right system packages are
installed. Two things that sandbox couldn't reach to verify, both for
infrastructure reasons rather than known bugs: the first-run speech-model
download in `audio/model_setup.py` (that sandbox's network policy blocks
the download host entirely; the download/extract code follows standard,
unremarkable `httpx` + `zipfile` patterns) and the wake-word/STT audio
loop itself (no microphone in a container). The two install scripts need
a real Fedora/Windows machine to exercise end-to-end, which is why this
project was built and tested to that boundary rather than claimed beyond
it.

## Project layout

```
ev_assistant/
  cli.py           entry point (`ev ...`)
  daemon.py        wires everything together: wake→listen→think→speak + feeds + control API
  server.py        localhost control API (FastAPI)
  brain.py         Claude client
  personality.py   TARS-inspired, adjustable system prompt
  memory.py        SQLite conversation + fact store
  data_feeds.py    background RSS/weather ingestion
  config.py        TOML config + env secrets
  audio/
    wake_word.py   continuous "Hey E.V." spotting (Vosk)
    stt.py         records + transcribes your command
    tts.py         speaks the reply (pyttsx3)
scripts/           Fedora (systemd) and Windows (PyInstaller + Task Scheduler) packaging
tests/             unit tests (no audio/network/API-key required)
```
