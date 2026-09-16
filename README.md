# E.V.

A voice-activated personal assistant that runs on your own machine. Say
"Hey E.V." (or "Yo E.V.", or just "E.V.") and she listens, answers out
loud in an Australian voice, and can act on your computer - open and close
apps, open websites, control volume and media, tweak GNOME, launch your
Jegeo console, and run commands. She has a cyberpunk console GUI with a
live audio visualizer, a growing memory, and an offline mode with a
survivalist knowledge base for when you have no internet. Personality
(dry, blunt, TARS-ish) is yours to tune.

She is software only: no camera, no body, no moving parts - just a voice.

Read the [Security](#security) section before turning on full system
control.

## Contents

- [What she can do](#what-she-can-do)
- [Fedora setup](#fedora-setup)
- [Talking to her](#talking-to-her)
- [The GUI](#the-gui)
- [System control & permissions](#system-control--permissions)
- [Voice](#voice)
- [Offline mode & the knowledge base](#offline-mode--the-knowledge-base)
- [How she finds things](#how-she-finds-things)
- [Jegeo](#jegeo)
- [Moving E.V. to another PC](#moving-ev-to-another-pc)
- [Configuration](#configuration)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [What was and wasn't verified](#what-was-and-wasnt-verified)

## What she can do

- **Wake on her name** - "Hey E.V.", "Yo E.V.", "E.V., can you..." or a bare
  "E.V." Say the command in the same breath ("E.V., open Firefox") and she
  acts on it directly.
- **Answer with Claude** when online, in an Australian voice.
- **Act on your machine** - open/close apps, open websites, volume/media,
  GNOME settings and extensions, launch Jegeo, run shell commands - gated by
  a permission tier you choose, with spoken confirmation before anything
  destructive.
- **Remember** - conversations and background-pulled news/weather.
- **Work offline** - rule-based voice commands still run, and questions are
  answered from a local knowledge base you build with `ev learn` (Wikipedia,
  web pages, files) - a survivalist/reference brain for no-signal situations.
- **A cyberpunk GUI** - a reactive audio visualizer, a text box to type to
  her, a live transcript, and panels to tune her personality, voice, and
  permissions.

## Choosing E.V.'s brain

E.V. can run on any of three brains, set in `[brain] provider` (or the GUI
**BRAIN** tab). She automatically falls back to a local Ollama model, then to
reading her offline notes, if the chosen one can't be reached.

| Provider | Key? | Cost | Notes |
|---|---|---|---|
| **ollama** (default) | none | free | A model on *your* PC. Private, works offline. Install [Ollama](https://ollama.com) + `ollama pull llama3.1`. Use a tool-capable model (llama3.1, qwen2.5, mistral) so she can control your computer. |
| **claude** | paid | paid | Anthropic Claude - best quality. `ev set-key sk-ant-...`, then set provider to `claude`. |
| **openai** | usually free | free tier | Any OpenAI-compatible endpoint: **Groq** (fast, free), **Google Gemini**, **OpenRouter**, a local server. `ev set-key --openai KEY`, set `openai_base_url` + `openai_model`. |

She ships defaulting to **Ollama** so she works with no API key at all - install
Ollama (the Fedora installer offers to do this for you) and she can converse,
online or off. Switch to Claude whenever you buy credits; nothing else changes.

Example base URLs for the `openai` provider:
- Groq: `https://api.groq.com/openai/v1`
- Gemini: `https://generativelanguage.googleapis.com/v1beta/openai`
- OpenRouter: `https://openrouter.ai/api/v1`

## Fedora setup

```bash
git clone https://github.com/voidless026-design/specter.git
cd specter
git checkout claude/ev-voice-activated-assistant-lrf1z7
bash scripts/install-fedora.sh
```

The installer adds the system packages she needs (`portaudio`, `espeak-ng`,
`alsa-utils`, `ffmpeg-free`, `playerctl`, `xdg-utils`), creates a virtualenv
under `~/.local/share/ev-assistant`, writes a default config, generates her
control token, and registers a `systemd --user` service so she starts at
login. It runs as **your user**, not root - she needs your desktop's audio
session, and running as root would both break her audio and be dangerous
(see [Security](#security)).

The installer offers to install Ollama and pull a model, so out of the box she
has a free local brain (no API key). Then:

1. `systemctl --user start ev-assistant`
2. **Run `ev doctor`** - it checks the active brain (Ollama running + model
   pulled, or a live key test), audio, the Australian voice, the mic, and
   system tools, and tells you exactly what to fix. Fastest path to a working
   setup.
3. `ev mic-test` to confirm the microphone and wake word.
4. Say "Hey E.V." and wait for "Go ahead."

To use Claude instead (when you have credits): `ev set-key sk-ant-...`, set
`[brain] provider = "claude"` (or the GUI BRAIN tab), restart, `ev doctor`.

> **"my brain isn't set up to converse"** means the chosen brain isn't ready:
> for Ollama, install it and `ollama pull llama3.1`; for Claude/cloud, add the
> key. `ev doctor` pinpoints it. **Robotic (not Australian) voice** means the
> neural voice isn't reaching Microsoft's servers or there's no audio player -
> `ev doctor` says which.

Uninstall with `bash scripts/uninstall-fedora.sh`.

## Talking to her

| From | How |
|---|---|
| Voice | "Hey E.V.", "Yo E.V.", or "E.V." — then your request. Or all at once: "E.V., open Firefox." |
| The GUI | `ev gui` (opens the cyberpunk console in your browser) |
| Any shell / SSH | `ev ask "what's the weather"` (`-q` to not speak it aloud; `-y` to pre-approve a destructive action) |
| Check she's alive | `ev status` |
| Stop her | `ev stop` |
| Search her notes directly | `ev search "which knot won't slip"` (no brain call) |
| See what she knows | `ev stats` |

`ev ask`/`status`/`stop`/`gui` read her token automatically from
`~/.config/ev-assistant/env`, so they work from any shell, including a fresh
SSH session, with nothing to export.

**Wake names** are configurable (`[wake_word] names`). Several spellings of
"E.V." ship by default because speech recognizers render it inconsistently.
If she wakes too easily or not enough, add or trim spellings there.

## The GUI

`ev gui` opens a Cyberpunk-2077-styled console served by the daemon:

- A central **audio visualizer** that pulses with what she hears and
  animates while she thinks and speaks (driven live over a WebSocket).
- A **text box** to type commands/questions (with a speak-aloud toggle).
- A **live transcript** log.
- A **SETTINGS** panel - personality (humor/honesty/verbosity/custom
  instructions), voice engine and rate, and permission tier - saved back to
  your config.
- An **OFFLINE** panel - teach her from Wikipedia, a web page, or a typed
  note, and see how many passages her offline brain holds.

It's served on `127.0.0.1` only and needs her token (which `ev gui` puts in
the URL for you).

## System control & permissions

When you ask her to do something, Claude picks a tool and E.V. runs it under
your chosen **permission tier** (`[permissions] tier`):

- **safe** — open apps, open websites, launch Jegeo. Nothing else.
- **standard** (default) — the above + close apps, volume/media, GNOME
  settings and extensions.
- **full** — the above + arbitrary shell commands.

Two guards apply on top:

- **Spoken confirmation** before anything destructive (`rm`, `sudo`, `kill`,
  disk operations, and similar). She asks out loud and waits for you to say
  "yes." Toggle with `[permissions] confirm_destructive`.
- **A blocklist** (`[permissions] forbidden_patterns`) that is refused
  outright at every tier, no confirmation offered (e.g. `rm -rf /`, `mkfs`).

Tiers are enforced twice - the tool list Claude is even offered is filtered
by tier, and every action is re-checked before it runs.

Examples: "E.V., open Firefox" · "close Spotify" · "set volume to 30" ·
"next track" · "enable the Dash to Dock extension" · "set my GNOME theme to
dark" · (full tier) "how much disk space is left."

## Voice

E.V. speaks with an Australian female voice online and falls back gracefully
offline. Engine is `[voice] engine`:

- **auto** (default) — `en-AU-NatashaNeural` (Microsoft Edge neural, free, no
  key) when online; best offline voice otherwise.
- **edge** — always the online Australian neural voice.
- **piper** — a local neural voice (offline). Install one with
  `ev voices --install-piper` (note: no Australian Piper model exists
  upstream, so this is British/US neural).
- **espeak** — `espeak-ng`, robotic but always works, fully offline, no setup.

See every online voice with `ev voices --online` (add `--all` for all
locales), then set `[voice] edge_voice`.

## Offline mode & the knowledge base

`[offline] mode`:

- **auto** (default) — Claude when the network is up, local brain when it's
  down. She also falls back to offline automatically if a request drops
  mid-network.
- **online** — always Claude (errors if offline).
- **offline** — never uses the network.

Offline, two things still work: **rule-based voice commands** (open/close
apps, volume, media, launch Jegeo) and **question answering from a local
knowledge base** you build up in advance:

```bash
ev learn "Water purification"              # bare topic -> Wikipedia
ev learn https://example.com/survival      # URL -> web page
ev learn ~/notes/wilderness.pdf            # path -> .txt, .md, or .pdf
```

The source type is auto-detected; `--wikipedia` / `--url` / `--file` force it.
To pull in a whole reference site, follow links to a bounded depth:

```bash
ev learn https://example.com/survival --depth 1              # + everything it links to
ev learn "Wilderness survival" --depth 1 --max-pages 50      # crawl Wikipedia links
ev learn https://example.com/docs --depth 2 --any-domain     # off-site links too
```

Crawling is deliberately bounded: `--max-pages` (default 20) is a hard budget,
links stay on the starting domain unless you pass `--any-domain`, already-seen
pages are skipped so cycles terminate, and links pointing at private/loopback
addresses are refused. Re-learning the same page is a no-op - content is
de-duplicated by hash - so you can safely re-run a crawl to pick up new pages.
Use `--force` to re-ingest a page whose content changed.

Everything you teach her goes into a single-file SQLite store with both a
keyword index and a vector index. When you ask something, the question is
planned, searched across both, reranked, and the best passages are handed to
her brain with source labels she can cite - or, when nothing is relevant
enough, she's told the store had nothing and says so instead of guessing.
See [How she finds things](#how-she-finds-things) below.

For real generated answers offline (not just passage lookup), install
[Ollama](https://ollama.com), pull a small model, and set
`[offline] ollama_model` (e.g. `llama3.2`).

## How she finds things

Asking E.V. a question does not just grep her notes. The pipeline is:

1. **Decide whether to search at all.** "Thanks", "open Firefox" and
   "what's 17 times 3" never touch the knowledge base. Retrieving on
   everything is how an assistant gets worse.
2. **Rewrite and expand.** "What about its melting point?" becomes a
   standalone question using the last few turns, then fans out into a
   natural-language form, a keyword-only form for BM25, and a hypothetical
   answer paragraph (which matches document prose better than a question
   does).
3. **Search both indexes at once** - vector similarity and keyword BM25 -
   plus a direct lookup of any names the question mentioned.
4. **Fuse** the rankings, then **rerank** the shortlist with a cross-encoder
   that reads the question and each passage together. This step is the
   single biggest quality difference in the whole system.
5. **Apply a relevance floor.** If nothing clears it, she returns nothing,
   and says the answer is from general knowledge rather than your notes.
   An empty result is a real answer.
6. **Assemble the context**: best passages at the start *and* end of the
   block (attention is weakest in the middle), each tagged `[S1]`, `[S2]`
   so she can cite them. Tags appear in text and are dropped from speech.

### Looking inside it

```bash
ev search "which knot won't slip"     # ranked results, no brain call
ev retrieve --explain "..."           # every stage: plan, scores, what survived the floor
ev retrieve "..." --json              # the same, machine-readable
ev stats                              # documents, chunks, coverage, namespaces
ev reindex                            # build any missing vectors (resumable)
ev reindex --all --model-changed      # rebuild after changing the embedding model
ev eval                               # measure retrieval quality (see evals/README.md)
```

`ev doctor` checks all of it: sqlite-vec loaded, embedding model present,
reranker present, dimension match, unembedded chunks, orphaned rows.

### Namespaces

Documents are filed as `personal`, `code`, `reference`, `news`, or a
`domain:yours`. A coding question is boosted toward `code` without being
locked out of everything else - a misrouted question that finds nothing is
worse than one that ranks imperfectly. `personal` is always searched and
always ranked above the rest: what she knows about *you* outranks what she
read on the internet.

### The models, and what happens without them

| | Default | Without it |
|---|---|---|
| Embeddings | `BAAI/bge-m3`, local, ~2GB | Lexical hashing vectors - matches words, not meaning |
| Reranking | `BAAI/bge-reranker-v2-m3`, local | Word overlap - noticeably worse, and worse at saying "I don't know" |

Both download once and then work entirely offline. Install them with:

```bash
pip install 'ev-assistant[local-embeddings]'
```

Without them she still works — keyword search, vector search on hashed
lexical vectors, and abstention all function — but the quality difference is
real and `ev doctor` will tell you so. `evals/README.md` measures exactly
what the gap costs.

## Jegeo

E.V. can launch your Jegeo security-operator console
([dig_atk](https://github.com/voidless026-design/dig_atk)) on command -
"E.V., launch Jegeo", or "open the network scanner in Jegeo." Install Jegeo
first (its own repo has a Fedora installer); E.V. runs `jegeo` (or
`python -m jegeo`). Deeper step-by-step automation of Jegeo's modules
depends on Jegeo exposing a command-line/scriptable interface; today E.V.
launches it (optionally naming a module to open). Tell me if you want E.V.
to drive specific Jegeo workflows and I'll wire those to whatever interface
Jegeo exposes.

## Moving E.V. to another PC

To use a beefier PC's brainpower, you have two options:

**Move her whole brain (config + memory + knowledge):**

```bash
# On this PC:
ev export ev-brain.tar.gz
# copy the file across, then on the other PC (with E.V. installed):
ev import ev-brain.tar.gz
```

**Point a laptop at another PC's running daemon** (text/GUI - the laptop
sends questions to the desktop's brain): set `[control_api] remote_host` and
`remote_port` in the laptop's config to the desktop, and use SSH port
forwarding since the API is localhost-only:

```bash
ssh -L 8765:127.0.0.1:8765 you@desktop     # forward the desktop's daemon
ev ask "..."                               # laptop now uses the desktop's brain
```

**Build a standalone binary bundle** (no Python needed on the target Fedora
PC): `bash scripts/build-bundle-fedora.sh` produces `dist/ev/` - copy the
folder over, `dnf install portaudio espeak-ng` there, and run `./ev/ev`.

## Configuration

Config lives at `~/.config/ev-assistant/config.toml` (secrets in the
sibling `env` file). Edit and `systemctl --user restart ev-assistant`, or
use the GUI SETTINGS panel. Key sections: `[brain]` (model, effort),
`[personality]` (humor, honesty, verbosity, custom_instructions),
`[wake_word]` (names, prefixes), `[voice]`, `[permissions]`, `[offline]`,
`[control_api]` (incl. remote brain), `[data_feeds]`. Every field is
commented in the file.

## Security

- **Permission tiers gate what she can do.** `full` lets anything that
  reaches your microphone - a video, someone in the room, even a malicious
  news headline she ingested - potentially run commands as you. Destructive
  actions still need spoken confirmation, but treat `full` with the same
  respect as an open root shell. `standard` is the default for good reason.
- **Don't run her as root.** She's a `systemd --user` service on purpose:
  audio lives in your user session, and root + always-on mic is a bad combo.
  When a task genuinely needs elevation she'll use `sudo` for that command
  (and destructive `sudo` triggers confirmation).
- **The control API is localhost-only** and bearer-token protected. Reaching
  it needs a shell on the machine (SSH counts) or a tunnel you set up. It's
  never exposed to your network by default.
- **Feed and knowledge content is data, not instructions** - E.V. is told to
  treat it as reference material. Still, only point `feeds` and `ev learn` at
  sources you trust.
- **Your `env` file holds your API key and token in plaintext** - it's
  `chmod 600` and `.gitignore`d. `ev export` bundles it, so keep exported
  archives private.

## Troubleshooting

**"EV_CONTROL_TOKEN is not set"** - run `ev init` (the installer does this).
The token lives in `~/.config/ev-assistant/env` and is read automatically.

**She doesn't hear me / nothing happens when I speak** - run `ev mic-test`.
It shows a live input-level bar and transcribes what you say. If the bar
stays near zero, the wrong microphone is selected: `ev devices`, then set
`[audio] input_device` to the right index/name and restart. If the bar moves
but the wake word isn't detected, say one of the configured names clearly, or
add a spelling to `[wake_word] names`.

**She hears me but doesn't speak back** - test the OS voice directly:
`espeak-ng "test"`. Silent? Install `sudo dnf install alsa-utils espeak-ng`.
For the Australian neural voice you also need an mp3 player and internet:
`sudo dnf install ffmpeg-free` (or mpv). If `espeak-ng` speaks but E.V.
doesn't, it's usually the `systemd --user` service starting before your audio
session - `systemctl --user restart ev-assistant` after you're logged in.

**"No audio player found"** - `sudo dnf install ffmpeg-free` (provides
`ffplay`, used to play the neural voice).

**Claude errors (bad key, rate limit, offline)** - she says so out loud
rather than going silent; `journalctl --user -u ev-assistant -f` has details.
If offline, she automatically uses her local brain.

## What was and wasn't verified

Built and tested in a Linux container with no microphone, speaker, GNOME
desktop, or Windows machine, and with restricted network egress. So:

- **Verified here:** every module imports cleanly; the full unit-test suite
  passes (627 tests - config/env-file secrets, memory, wake-word matching,
  the permission-tier and confirmation gates on the executor, the offline
  command parser, the settings editor, the provider tool-use loop against a
  real local stub server, the control API incl. auth, `/ask`, `/learn`,
  settings and the WebSocket, and the whole retrieval layer: schema and
  index triggers, chunking, embeddings, namespaces, query planning, hybrid
  search, reranking, the relevance floor, context assembly and citations);
  `ev learn` / `search` / `retrieve` / `reindex` / `stats` / `eval` / `doctor`
  run end-to-end against a real store; the server serving the GUI and
  streaming state; and TTS actually speaking via `espeak-ng`.
- **Needs your real machine to confirm** (standard patterns, not verified
  end-to-end here): the live microphone wake/record loop, the Australian
  edge-tts voice (blocked network here), Piper install, the GNOME/app/media
  system actions (no desktop here), launching Jegeo, the first-run speech
  model download, and the Fedora/Windows install and bundle scripts.
- **Retrieval quality specifically.** `huggingface.co` was blocked here, so
  **neither `BAAI/bge-m3` nor `BAAI/bge-reranker-v2-m3` could be downloaded
  or measured.** Everything was built behind interfaces and verified against
  the fallbacks (lexical hashing vectors, word-overlap reranking), and the
  sentence-transformers adapter itself is tested against the real library
  using a weightless model. The checked-in eval baseline is a
  *fallback-quality* number, not a target number — see `evals/README.md`,
  which says exactly which numbers are real and which need your machine.
  Run `ev eval` there and tune `relevance_floor` against your own results.

Run `ev mic-test` first on your machine - it's the fastest way to confirm the
audio path end-to-end.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Project layout

```
ev_assistant/
  cli.py           the `ev` command (init, daemon, ask, gui, learn, search, stats, eval, ...)
  daemon.py        wake -> listen -> think/act -> speak loop + feeds + API/GUI
  server.py        localhost control API + GUI host + WebSocket state stream
  brain.py         provider routing, retrieval integration, citations
  providers.py     Claude / Ollama / OpenAI-compatible brains, one interface
  offline.py       offline rule-based commands
  personality.py   adjustable system prompt, incl. the citation rules
  memory.py        SQLite conversation + fact store

  -- the retrieval layer --
  store.py         one SQLite file: documents, chunks, facts, FTS5 + sqlite-vec
  chunking.py      structure-aware chunking (headings, sentences, code)
  embeddings.py    bge-m3 / API / hashing backends, resumable indexing
  rerank.py        cross-encoder / Cohere / word-overlap, and the floor
  query.py         should we search at all, rewriting, variants, filters
  namespaces.py    routing at ingest, soft preference at query time
  retrieval.py     hybrid search, RRF fusion, rerank, floor, diversify
  context.py       budgeted context block with citable source tags
  enrich.py        background summaries and atomic facts
  library.py       the one ingest path (CLI, API, feeds all use it)
  migrate.py       carries an old knowledge.sqlite3 into the store
  evaluate.py      the `ev eval` harness
  probe.py         remembers which models aren't available, so CLI stays fast

  ingest.py        `ev learn` fetchers (Wikipedia / URL / file)
  knowledge.py     the pre-retrieval knowledge base (migrated from, kept for offline)
  data_feeds.py    background news/weather ingestion
  config.py        TOML config + env-file secrets
  settings.py      in-place config editing for the GUI
  net.py / bus.py  connectivity check / live-state bus for the visualizer
  tools/           system-control executor + tool schemas
  audio/           wake word (Vosk), STT, TTS (edge/piper/espeak)
  gui/index.html   the cyberpunk console
evals/             golden question set + corpus for `ev eval`
scripts/           Fedora + Windows install/bundle packaging
tests/             627 unit tests (no audio/network/API key needed)
```
