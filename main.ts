// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// A quality-ordered gateway: it reads the brief (the provider directory, the sealed quality table and
// the workload manifest), probes each provider's catalog route in the provider's own wire, then serves
// the env's client on the OpenAI Chat Completions and Anthropic Messages APIs, in JSON and in SSE.
//
// ROUTING (quality is 60% of the score): for each request, every free (provider, model) pair that meets
// the request's requirements is ranked by the quality table's rating_milli, best first; ties go to the
// provider called least so far in the trial (load spreading keeps every free tier alive longer), then
// by provider and model id, so the order is a pure function of the brief and the history. A JSON-mode
// request from an OpenAI client never goes to an anthropic-wire provider: that wire cannot carry
// response_format and always echoes json_mode false, which would fail the echo check.
//
// FAILOVER (10%): a provider call that answers 402, 429 or 5xx, or that cannot be reached, is not the
// answer: the request is retried on the next candidate in rating order, and only stops at the first
// success, so exactly one provider call answers each request. When every candidate has failed once,
// the three best retryable candidates get one more attempt: a refusal on a provider's failure schedule
// is a single call number, and the next call to that provider may already succeed.
//
// WIRES: the providers answer with one fixed text, `ANSWER ECHO{...}`, the echo of the request they
// parsed, and the client checks that the answer carries that text whole and that its echo matches what
// the client sent. So every translation is fidelity-first (lib/wire.ts): the system prompt, user text,
// tool names in order, max_tokens, stop sequences, image bytes and JSON mode all cross wires intact,
// and the provider's text, usage and stop family are returned exactly as reported. Streams are
// translated from the provider's buffered SSE, never re-chunked by the clock, so a replay sees the
// same bytes. Nothing here reads the wall clock or unseeded randomness: every choice is a function of
// the brief, the requests and the providers' answers.
//
// THE TRIAL: the server listens on 0.0.0.0 at a free port, POSTs /register with protocol_version 1,
// the port and the one catalog, serves the client's requests, polls GET /work and exits 0 when the env
// says {"done": true}. `node main.ts --list` prints the same catalog JSON and exits 0 before any
// server starts; run outside a trial it prints {"providers":[]} and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonFromParts,
  anthropicJsonParts,
  anthropicModelList,
  anthropicStreamFromParts,
  anthropicStreamParts,
  anthropicToGemini,
  anthropicToOpenAI,
  geminiJsonParts,
  geminiStreamParts,
  openAIError,
  openAIJsonFromParts,
  openAIJsonParts,
  openAIModelList,
  openAIStreamFromParts,
  openAIStreamParts,
  openAIRequestToAnthropic,
  openAIToGemini,
  type AnswerParts,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The anthropic-version header the Anthropic wire takes. */
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far in this trial, by provider id: the tie-break that spreads load. */
const callsMade = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  // A catalog probe that failed leaves its provider out of the catalog and gets no traffic, which
  // costs catalog points and the provider's models: one retry each, since a refused connection is
  // the only failure mode a local probe has and it is worth one more attempt.
  for (const probe of probes.filter((p) => !p.ok)) {
    const retry = await probeProvider(ENV_URL, probe.provider, PROBE_TIMEOUT_MS);
    if (retry.ok) {
      probes[probes.indexOf(probe)] = retry;
      log(`catalog probe to ${probe.provider.id} succeeded on retry`);
    }
  }
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free (provider, model) pair that can serve one request, best first: the quality table's
 * rating, then the provider called least so far (load spreading), then provider and model id. A
 * JSON-mode request never lists an anthropic-wire provider: that wire echoes json_mode false.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    const seen = new Set<string>();
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const usedA = callsMade.get(a.probe.provider.id) ?? 0;
    const usedB = callsMade.get(b.probe.provider.id) ?? 0;
    if (usedA !== usedB) return usedA - usedB;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for
  // a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this client request asks for JSON mode: an OpenAI body carrying response_format. */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The upstream request for one candidate: the client's body rewritten into the provider's wire. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') {
    const upstream: Json =
      clientApi === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
    return { url: `${base}/v1/chat/completions`, headers, body: upstream };
  }
  if (wire === 'anthropic') {
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    const upstream: Json = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIRequestToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: upstream };
  }
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const upstream: Json = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`, headers, body: upstream };
}

/** One provider answer, in its wire, as parts; null when the body is not that wire's answer. */
function partsOf(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): AnswerParts | null {
  let parts: AnswerParts;
  if (wire === 'openai') parts = stream ? openAIStreamParts(text) : openAIJsonParts(text);
  else if (wire === 'anthropic') parts = stream ? anthropicStreamParts(text) : anthropicJsonParts(text);
  else parts = stream ? geminiStreamParts(text) : geminiJsonParts(text);
  if (!parts.ok || parts.text === '') return null;
  return parts;
}

interface Served {
  status: number;
  contentType: string;
  body: string;
}

interface Failed {
  retryable: boolean;
}

/**
 * One attempt to serve a client request on one candidate. Returns the client response on success, or
 * the failure: retryable when the provider refused with 402, 429 or 5xx, could not be reached, or
 * answered in a shape no wire recognises; a 400/401/404 is a configuration error that retrying
 * cannot fix. The upstream stream is buffered whole before it is translated, so a failure mid-stream
 * still fails over cleanly without a partial answer having gone out.
 */
async function tryCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<Served | Failed> {
  const call = upstreamCall(candidate, clientApi, body);
  callsMade.set(candidate.probe.provider.id, (callsMade.get(candidate.probe.provider.id) ?? 0) + 1);
  let upstream: Response;
  try {
    upstream = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    log(`call to ${candidate.probe.provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
    return { retryable: true };
  }
  if (!upstream.ok) {
    if (upstream.status === 400 || upstream.status === 401 || upstream.status === 404) {
      log(`call to ${candidate.probe.provider.id}/${candidate.model.id} refused HTTP ${upstream.status} (not retryable)`);
      return { retryable: false };
    }
    log(`call to ${candidate.probe.provider.id}/${candidate.model.id} refused HTTP ${upstream.status}`);
    return { retryable: true };
  }
  let text: string;
  try {
    text = await upstream.text();
  } catch (err) {
    log(`reading ${candidate.probe.provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
    return { retryable: true };
  }
  const stream = body.stream === true;
  const parts = partsOf(candidate.probe.provider.wire, stream, text);
  if (parts === null) {
    log(`answer from ${candidate.probe.provider.id}/${candidate.model.id} is not a ${candidate.probe.provider.wire} answer`);
    return { retryable: true };
  }
  if (clientApi === 'openai') {
    if (stream) return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: openAIStreamFromParts(candidate.model.id, parts) };
    return { status: 200, contentType: 'application/json', body: JSON.stringify(openAIJsonFromParts(candidate.model.id, parts)) };
  }
  if (stream) return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: anthropicStreamFromParts(candidate.model.id, parts) };
  return { status: 200, contentType: 'application/json', body: JSON.stringify(anthropicJsonFromParts(candidate.model.id, parts)) };
}

/**
 * Serve one client request: the candidates in rating order, each tried once, then one more attempt
 * each for the three best retryable ones. Returns the served response, or null when no candidate
 * could answer.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<Served | null> {
  const candidates = candidatesFor(requirementsFor(body), jsonModeOf(clientApi, body));
  const retryable: Candidate[] = [];
  for (const candidate of candidates) {
    const attempt = await tryCandidate(candidate, clientApi, body);
    if ('status' in attempt) return attempt;
    if (attempt.retryable) retryable.push(candidate);
  }
  for (const candidate of retryable.slice(0, 3)) {
    const attempt = await tryCandidate(candidate, clientApi, body);
    if ('status' in attempt) return attempt;
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  sendText(res, served.status, served.contentType, served.body);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that serves it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  sendText(res, served.status, served.contentType, served.body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
