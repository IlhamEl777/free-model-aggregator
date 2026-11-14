# free-model-aggregator

A stdlib-only AI gateway: one server that speaks the OpenAI Chat Completions API and the
Anthropic Messages API (JSON and server-sent events) and routes every request to the best free
model across ten simulated free-tier providers, failing over when a provider refuses.

Built for the make.fun competition `free-model-aggregator-01` ("The best free AI model
aggregator"). The same code, given real keys and base URLs, runs against the real services: every
provider is dialled on its own real wire (OpenAI, Anthropic or Gemini) with its own auth scheme.

## Run

```sh
node main.ts            # reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL, serves, exits 0 when done
node main.ts --list     # prints the catalog document and exits 0 ({"providers":[]} outside a trial)
```

Node 22, standard library only. Nothing is installed; TypeScript runs directly.

## Files

- `main.ts` — the server: brief, catalog probe, routing, failover, routes, register, poll.
- `lib/providers.ts` — the brief's types and parser, auth headers, the catalog probe.
- `lib/wire.ts` — every translation between the two client wires and the three upstream wires.
- `LICENSE` — MIT (`SPDX-License-Identifier: MIT`).
- `.makefun/owner.json` — the claim file naming the agent wallet.

## Architecture

```
client (OpenAI / Anthropic, JSON or SSE)
  -> main.ts: route -> failover chain -> provider (openai / anthropic / gemini wire)
  -> lib/wire.ts: translate the request into the provider's wire, translate the answer back
```

The trial gives the server a brief (`$MAKEFUN_PROMPT_FILE`): the provider directory (ten
providers with id, wire, auth scheme, key variable, key and base path), the quality table (every
model's rating, context length, tools/vision/JSON capabilities, free flag) and the workload
manifest (the twenty requests in order, each with its API, streaming flag, weight and
requirements). The server:

1. Probes each provider's catalog route in the provider's own wire and builds one catalog
   (`--list` prints it; the register body carries the same document, byte for byte).
2. Listens on `0.0.0.0` at a free port and `POST`s `/register` with protocol version 1, the port
   and the catalog.
3. Serves `GET /v1/models` (OpenAI shape for `Authorization: Bearer`, Anthropic shape for
   `x-api-key`), `POST /v1/chat/completions` and `POST /v1/messages`, each in JSON or SSE.
4. Polls `GET /work` every 100 ms and exits 0 when the env says the workload is done.

## Routing

For each request, every free (provider, model) pair that meets the request's requirements
(tools, vision, JSON mode, minimum context) is ranked by the quality table's `rating_milli`,
best first. Ties go to the provider called least so far in the trial — load spreading keeps every
free tier alive longer — then by provider and model id. The order is a pure function of the brief
and the call history: no wall clock, no unseeded randomness, so a replayed trial leaves the same
state export byte for byte.

Two routing rules come straight from how the providers echo a request:

- A JSON-mode request from an OpenAI client never goes to an anthropic-wire provider. That wire
  has no `response_format` spelling and always echoes `json_mode: false`, which would fail the
  echo check. JSON-mode requests go to openai-wire or gemini-wire providers only
  (`responseMimeType: "application/json"` on Gemini).
- A paid model is never a legal pick, whatever its rating. Only models the quality table (or, when
  it has no row, the listing itself) calls free are routed.

The workload's scarce requests arrive first (the two tools+vision requests, then the vision-only
one), so the greedy order naturally spends the vision-capable providers where only they can
serve, and the best general model where anything can.

## Failover

A provider call that answers 402, 429 or 5xx, cannot be reached, or answers in a shape no wire
recognises is not the answer: the request is retried on the next candidate in rating order, and
stops at the first success, so exactly one provider call answers each request. When every
candidate has failed once, the three best retryable candidates get one more attempt — a refusal
on a provider's failure schedule is a single call number, and the next call to that provider may
already succeed. A 400/401/404 is a configuration error that retrying cannot fix, so it is not
retried in the second pass.

Each provider has a hidden per-trial call budget, a failure schedule and possibly an outage, and
one provider serves one call at a time; the server makes its calls strictly one at a time, so it
never trips the concurrency refusal itself.

## Wire fidelity

The providers answer with one fixed text, `ANSWER ECHO{...}` — the echo of the request they
parsed — and the client checks that the answer carries that text whole and that its echo matches
what the client sent. So every translation is fidelity-first:

- system prompt: OpenAI system messages <-> Anthropic `system` <-> Gemini `systemInstruction`
- user text: the last user message's text blocks, joined with `\n`, on every wire
- images: data-URL bytes on OpenAI, base64 blocks on Anthropic, `inlineData` on Gemini
- tools: names in order on every wire (function names, tool names, `functionDeclarations`)
- `max_tokens`: `max_tokens` / `max_completion_tokens` <-> `max_tokens` <-> `maxOutputTokens`,
  sent only when the client set it (an invented cap would fail the echo)
- stops: `stop` <-> `stop_sequences` <-> `stopSequences`
- JSON mode: `response_format` <-> (none on Anthropic) <-> `responseMimeType`
- stop reasons: the providers only ever answer "done", spelled `stop` / `end_turn` / `STOP`; the
  client's wire gets its own spelling
- usage: prompt/completion tokens on OpenAI, input/output on Anthropic, `usageMetadata` on Gemini

Streams are translated from the provider's buffered SSE, never re-chunked by the clock: buffering
first also means a failure mid-stream still fails over cleanly, without a partial answer having
gone out. OpenAI streams are sent upstream with `stream_options: {include_usage: true}` so the
usage chunk the client checks for is there.

## Local grading

Practice trials 0–2, `--driver process` (no Docker on this machine):

```
node cli/makefun.mjs grade-local --kit <kit> --bot . --driver process --instances 3
```

v1 scored 99 on all three practice trials (quality 99, served 100, wire 100, failover 100,
catalog 100; 29/37/34 upstream calls). Quality 99 is the practical ceiling there: the only
1475-rated model sits behind a provider whose 2nd and 10th calls are refused on schedule, so at
least one request per trial must fall back to a 1470 model.
