// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. Every provider answers in its
// own wire (openai, anthropic or gemini), and this file translates between all six combinations.
//
// The providers answer with one fixed text: `ANSWER ECHO{...}`, the echo of the request they parsed.
// The env's client checks that the entry's answer carries that text whole and unchanged, and that the
// echo inside it matches what the client sent (system prompt, user text, tool names, max_tokens, stop
// sequences, image bytes, JSON mode). So every translation here is fidelity-first: it moves each of
// those fields across wires without rewriting or dropping it, and it returns the provider's text,
// usage and stop family exactly as the provider reported them.
//
//   - system: OpenAI system messages <-> Anthropic `system` <-> Gemini `systemInstruction`
//   - user text: the last user message's text blocks, joined with "\n", on every wire
//   - images: data-URL bytes on OpenAI, base64 blocks on Anthropic, inlineData on Gemini
//   - tools: names in order on every wire (function names, tool names, functionDeclarations)
//   - max_tokens: max_tokens / max_completion_tokens <-> max_tokens <-> maxOutputTokens
//   - stops: stop <-> stop_sequences <-> stopSequences
//   - JSON mode: response_format json_object/json_schema <-> (Anthropic has none: it always echoes
//     json_mode false, so JSON-mode requests must never go to an anthropic-wire provider)
//     <-> generationConfig.responseMimeType "application/json"
//   - stop: provider families stop / end_turn / STOP all mean "done"; the client's wire gets its own
//     spelling ("stop" for OpenAI, "end_turn" for Anthropic)
//   - usage: prompt/completion tokens on OpenAI, input/output on Anthropic, usageMetadata on Gemini
//
// Streams are translated from the provider's buffered SSE, never re-chunked by the clock: the same
// upstream bytes always give the same client bytes. The model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Any provider's stop spelling as an OpenAI finish_reason (the providers only ever answer "done"). */
export function openAIFinishOf(stop: unknown): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Any provider's stop spelling as an Anthropic stop_reason (the providers only ever answer "done"). */
export function anthropicStopOf(stop: unknown): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// Cross-wire translation: every provider answer, in any wire, reduced to one
// shape ({text, stop, input, output}) and rebuilt in either client wire.
// The text is carried whole and unchanged: the env's client checks that the
// answer carries the provider's ANSWER text byte for byte.
// ---------------------------------------------------------------------------

/** One parsed server-sent event frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split buffered SSE text into frames, tolerating CRLF. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** A provider answer reduced to its text, stop spelling and usage counts. */
export interface AnswerParts {
  text: string;
  /** The provider's stop spelling ("stop" | "end_turn" | "STOP" | ...), or null when unknown. */
  stop: string | null;
  inputTokens: number;
  outputTokens: number;
  /** False when the body could not be parsed as that wire's answer at all. */
  ok: boolean;
}

function jsonOf(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A Chat Completions JSON answer as parts. */
export function openAIJsonParts(bodyText: string): AnswerParts {
  const bad: AnswerParts = { text: '', stop: null, inputTokens: 0, outputTokens: 0, ok: false };
  const body = jsonOf(bodyText);
  if (body === null) return bad;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return bad;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null || message.role !== 'assistant') return bad;
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    ok: true,
  };
}

/** A Chat Completions SSE stream as parts: deltas joined, the finish chunk's reason, the usage chunk. */
export function openAIStreamParts(bodyText: string): AnswerParts {
  let text = '';
  let stop: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let sawChunk = false;
  for (const frame of parseSseFrames(bodyText)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOf(payload);
    if (chunk === null) continue;
    sawChunk = true;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text, stop, inputTokens, outputTokens, ok: sawChunk };
}

/** An Anthropic Messages JSON answer as parts. */
export function anthropicJsonParts(bodyText: string): AnswerParts {
  const bad: AnswerParts = { text: '', stop: null, inputTokens: 0, outputTokens: 0, ok: false };
  const body = jsonOf(bodyText);
  if (body === null || body.type !== 'message' || !Array.isArray(body.content)) return bad;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: answerText(body.content),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    ok: true,
  };
}

/** An Anthropic Messages SSE stream as parts: text deltas joined, the message_delta stop and usage. */
export function anthropicStreamParts(bodyText: string): AnswerParts {
  let text = '';
  let stop: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let sawEvent = false;
  for (const frame of parseSseFrames(bodyText)) {
    const data = jsonOf(frame.data);
    if (data === null) continue;
    sawEvent = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      text += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text, stop, inputTokens, outputTokens, ok: sawEvent };
}

/** The text of an Anthropic content block array (text blocks joined, nothing else). */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** A Gemini generateContent JSON answer as parts. */
export function geminiJsonParts(bodyText: string): AnswerParts {
  const bad: AnswerParts = { text: '', stop: null, inputTokens: 0, outputTokens: 0, ok: false };
  const body = jsonOf(bodyText);
  if (body === null || !Array.isArray(body.candidates) || body.candidates.length === 0) return bad;
  const candidate = isRecord(body.candidates[0]) ? body.candidates[0] : null;
  if (candidate === null) return bad;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    ok: true,
  };
}

/** A Gemini streamGenerateContent SSE stream as parts: candidate text joined, the final frame's stop and usage. */
export function geminiStreamParts(bodyText: string): AnswerParts {
  let text = '';
  let stop: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let sawFrame = false;
  for (const frame of parseSseFrames(bodyText)) {
    const data = jsonOf(frame.data);
    if (data === null || !Array.isArray(data.candidates) || data.candidates.length === 0) continue;
    sawFrame = true;
    const candidate = isRecord(data.candidates[0]) ? data.candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    text += parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, stop, inputTokens, outputTokens, ok: sawFrame };
}

/** Parts as a Chat Completions JSON object naming the serving model. */
export function openAIJsonFromParts(model: string, parts: AnswerParts): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: parts.text }, finish_reason: openAIFinishOf(parts.stop) }],
    usage: {
      prompt_tokens: parts.inputTokens,
      completion_tokens: parts.outputTokens,
      total_tokens: parts.inputTokens + parts.outputTokens,
    },
  };
}

/** Parts as a Chat Completions SSE stream naming the serving model, ending with a usage chunk and [DONE]. */
export function openAIStreamFromParts(model: string, parts: AnswerParts): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  if (parts.text !== '') frames.push(sseEvent(null, chunk({ content: parts.text }, null)));
  frames.push(sseEvent(null, chunk({}, openAIFinishOf(parts.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: parts.inputTokens,
        completion_tokens: parts.outputTokens,
        total_tokens: parts.inputTokens + parts.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** Parts as an Anthropic Messages object naming the serving model. */
export function anthropicJsonFromParts(model: string, parts: AnswerParts): Json {
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: parts.text }],
    stop_reason: anthropicStopOf(parts.stop),
    stop_sequence: null,
    usage: { input_tokens: parts.inputTokens, output_tokens: parts.outputTokens },
  };
}

/** Parts as an Anthropic SSE stream naming the serving model: message_start, one text block, message_delta, message_stop. */
export function anthropicStreamFromParts(model: string, parts: AnswerParts): string {
  return [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: parts.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ...(parts.text !== '' ? [sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: parts.text } })] : []),
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(parts.stop), stop_sequence: null },
      usage: { output_tokens: parts.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ].join('');
}

// ---------------------------------------------------------------------------
// Upstream request builders: the client's request rewritten for a provider's
// wire, keeping every echo field (system, user text, tools, max_tokens, stops,
// image bytes, JSON mode) intact across the translation.
// ---------------------------------------------------------------------------

/** The system text of an OpenAI request: its system messages' text, joined with "\n". */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'system') continue;
    const text = textOf(message.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** One OpenAI message's image_url part as a Gemini part; null when it carries no image. */
function geminiImagePart(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { inlineData: { mimeType: data[1] !== undefined && data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  }
  return { fileData: { fileUri: url } };
}

/** One Anthropic content block as a Gemini part: text or an inline image; null for the rest. */
function geminiPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** OpenAI tool definitions as Gemini functionDeclarations (names in order). */
function geminiFunctionDeclarations(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tool definitions as Gemini functionDeclarations (names in order). */
function geminiFunctionDeclarationsFromAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

/** The generationConfig for a Gemini call: only the caps the client actually set. */
function geminiGenerationConfig(maxTokens: unknown, stops: unknown, jsonMode: boolean, temperature: unknown, topP: unknown): Json {
  const config: Json = {};
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (Array.isArray(stops) && stops.length > 0) config.stopSequences = stops.filter((s) => typeof s === 'string');
  if (jsonMode) config.responseMimeType = 'application/json';
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. System messages become the
 * systemInstruction, user messages become user contents (text and image parts), assistant messages
 * become model contents, tool messages become user contents carrying the result text, and tools become
 * functionDeclarations; max_tokens, stops, JSON mode, temperature and top_p land in generationConfig.
 */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') continue;
    if (role === 'assistant') {
      const text = textOf(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      const parts: Json[] = [];
      if (text !== '') parts.push({ text });
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ text: `tool call ${typeof call.function.name === 'string' ? call.function.name : ''}` });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const text = toolResultText(raw.content);
      contents.push({ role: 'user', parts: [{ text }] });
      continue;
    }
    if (role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else {
        const image = geminiImagePart(block);
        if (image !== null) parts.push(image);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  const system = openAISystemText(body);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = geminiFunctionDeclarations(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const config = geminiGenerationConfig(
    typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined,
    body.stop,
    format === 'json_object' || format === 'json_schema',
    body.temperature,
    body.top_p,
  );
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The system prompt becomes the
 * systemInstruction, user messages become user contents (text and image parts; tool_result blocks are
 * dropped, since the echo counts only text blocks), assistant messages become model contents, and
 * tools become functionDeclarations; max_tokens, stop sequences, temperature and top_p land in
 * generationConfig. Anthropic has no JSON mode, so none is set.
 */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const text = textOf(content);
      if (text !== '') contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    if (raw.role !== 'user') continue;
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') continue;
      const part = geminiPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = geminiFunctionDeclarationsFromAnthropic(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config = geminiGenerationConfig(body.max_tokens, body.stop_sequences, false, body.temperature, body.top_p);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request: system messages become the
 * system string, messages and tool calls are rewritten, and the output cap, temperature, top_p, stop
 * sequences and streaming pass through. JSON mode has no Anthropic spelling and is dropped: a
 * JSON-mode request must never be routed to an anthropic-wire provider (see main.ts).
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const system = openAISystemText(body);
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url = block.image_url.url;
        const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
        if (data !== null) {
          blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== undefined && data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        } else {
          blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}
