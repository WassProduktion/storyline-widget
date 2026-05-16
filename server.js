import express from 'express';
import { createClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';
import 'dotenv/config';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();

app.use((req, res, next) => {
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith('Basic ')) {
    res.setHeader('WWW-Authenticate', 'Basic realm="Storyline Generator"');
    return res.status(401).send('Authentication required');
  }
  const decoded = Buffer.from(auth.slice(6), 'base64').toString();
  const colon = decoded.indexOf(':');
  const user = decoded.slice(0, colon);
  const pass = decoded.slice(colon + 1);
  if (user !== process.env.AUTH_USER || pass !== process.env.AUTH_PASS) {
    res.setHeader('WWW-Authenticate', 'Basic realm="Storyline Generator"');
    return res.status(401).send('Invalid credentials');
  }
  next();
});

app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SECRET_KEY);
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

async function embed(text) {
  const res = await fetch('https://api.voyageai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.VOYAGE_API_KEY}`,
    },
    body: JSON.stringify({ model: 'voyage-3', input: [text] }),
  });
  if (!res.ok) throw new Error(`Voyage API error ${res.status}`);
  const data = await res.json();
  return data.data[0].embedding;
}

async function getProfile(company) {
  const { data } = await supabase
    .from('company_profiles')
    .select('profile')
    .eq('company', company)
    .single();
  return data?.profile || null;
}

async function search(embedding, company, count = 10) {
  const { data, error } = await supabase.rpc('match_documents', {
    query_embedding: embedding,
    match_count: count,
    filter_company: company || null,
  });
  if (error) throw new Error(`Supabase error: ${error.message}`);
  return data;
}

const CHANNEL_TONE = {
  instagram: 'No warm-up, no context. The first second is the only second that matters. Write for someone mid-scroll who hasn\'t decided to watch yet. One emotion, one idea, one reason to stay.',
  linkedin:  'Thoughtful and direct. Written for people who think for a living. One clear idea, developed with precision. No buzzwords, no empty inspiration. Ends with something worth sitting with.',
  youtube:   'Narrative first, message second. The viewer chose to be here — earn that with a story that unfolds, not a pitch that performs. Specific details over grand statements. Show the work, not the idea of the work.',
  website:   'This is the definitive version. No trending format, no platform pressure. Write as if it will be read five years from now. The viewer came here on purpose — they are already interested. Reward that with depth, not a sales pitch. Quiet confidence over performance. The company at its most honest.',
};

const CHANNEL_LABEL = {
  instagram: 'Instagram',
  linkedin:  'LinkedIn',
  youtube:   'YouTube',
  website:   'Corporate Website / Brand Film',
};

const MODEL_INSTRUCTIONS = {
  berettermodellen: `Structure the narrative across these 7 phases (label each):
ANSLAG: A brief, attention-grabbing opening that hints at the film's theme, tone and premise.
PRÆSENTATION: Introduce the key people, context and set up the central conflict.
UDDYBNING: Deepen our understanding. Conflicts begin to emerge. Emotional investment grows.
POINT OF NO RETURN: The decisive moment — something changes irreversibly. No way back.
KONFLIKTOPTRAPNING: Tensions escalate. Complications arise. The protagonist faces real obstacles.
KLIMAKS: The peak — the decisive confrontation or resolution of the central conflict.
UDTONING: The new reality. What has changed? Where do we land emotionally?`,

  abt: `Structure the narrative as three clearly labeled sections:
AND: Set the scene — who is this about, what is the context? Keep it brief. Only enough to make BUT land with full force.
BUT: The heart of the story — what disrupts, challenges or is at stake? Concrete and human. The viewer should think: "I know that feeling."
THEREFORE: The logical and emotional conclusion. Must feel like a direct answer to BUT — a consequence that invites action.`,

  hcr: `Structure the narrative as three clearly labeled sections:
HOOK (first 3-8 seconds): Open a loop in the viewer's mind. Do NOT open with the company name or a generic introduction.
CONFLICT: Show the tension between two states — what is vs. what should be. Give this space. The viewer should actively want to know what happens next.
RESOLUTION: Land on three levels — practical (what changed, in facts), emotional (how it feels for the person in the story), and meaningful (why this matters beyond the immediate situation). CTA should feel like the natural next step in the story's logic.`,

  threeact: `Structure the narrative across three acts (approximate proportions: 20% / 50% / 30%):
ACT 1 — SETUP: Establish the world. Present a human protagonist — not the company. Plant the dramatic question. End with the inciting incident.
ACT 2 — CONFRONTATION: Raise the stakes and keep them there. Include a midpoint escalation and a darkest moment just before Act 3. No product features or benefit lists in this act.
ACT 3 — RESOLUTION: Show the transformation. Three layers: concrete results, the human emotional experience, and the larger meaning. CTA as invitation into the same transformation.`,

  psr: `Structure the narrative as three clearly labeled sections:
PROBLEM (approximately 50-60% of total length): A human situation with a face and real emotion. Must stand alone — do not hint at the solution. The viewer must arrive at the recognition that this is serious, on their own.
SOLUTION: The mechanism of change — not a product demo. The client is the hero. The company is the helper. No feature lists.
RESULT: Three levels — quantitative (numbers), qualitative (how daily life changed), strategic (what became possible). Let the protagonist speak in their own words.`,

  pixar: `Write the narrative as exactly six labeled sentences — no more:
ONCE UPON A TIME: Establish the world and the protagonist — one human we follow.
EVERY DAY: Describe the status quo — the normal before the change.
UNTIL ONE DAY: The decisive turning point — what disrupts everything.
BECAUSE OF THAT: The first consequence — directly and causally linked.
BECAUSE OF THAT: The second consequence. This must be cause and effect, never "and then."
UNTIL FINALLY: The resolution — what ends up happening and what has fundamentally changed.`,
};

function getDurationGuidance(sec) {
  if (sec <= 30) return 'IMPORTANT: Each phase must be 1-2 sentences maximum. The entire piece should be sharp and fast.';
  if (sec <= 90) return 'Keep each element focused — 2-4 sentences per section. Economy of language is essential.';
  return 'You have room to develop each element with specific detail and human texture. Use it.';
}

function getMaxTokens(sec) {
  if (sec < 60) return 600;
  if (sec < 180) return 1200;
  if (sec < 300) return 2000;
  return 3000;
}

function formatDuration(sec) {
  if (sec < 60) return `${sec} seconds`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return s > 0 ? `${m} min ${s} sec` : `${m} minutes`;
}

function buildPrompt({ topic, company, model, channel, duration, context, feedback, original }) {
  const companyLabel = company || 'the company';
  const modelInstructions = MODEL_INSTRUCTIONS[model] || MODEL_INSTRUCTIONS.berettermodellen;
  const durationGuidance = getDurationGuidance(duration);
  const durationLabel = formatDuration(duration);
  const channelLabel = CHANNEL_LABEL[channel] || channel;
  const tone = CHANNEL_TONE[channel] || '';

  const base = `You are an experienced video producer specialising in corporate storytelling.

${profile ? `COMPANY PROFILE — always reflect these values and commitments in the narrative:\n${profile}\n\n` : ''}Below is information about ${companyLabel}, sourced directly from their own materials:

${context}

`;

  if (feedback && original) {
    return base + `The following narrative was written for a ${channelLabel} video (${durationLabel}) about "${topic}":

---
${original}
---

Refine it based on this feedback: "${feedback}"

Keep the same model structure. Output only the narrative text — no stage directions, no camera notes. Write in English.

${durationGuidance}`;
  }

  return base + `Write a narrative for a ${channelLabel} video about: "${topic}"
Target duration: ${durationLabel}
Tone and register: ${tone}

Output ONLY the narrative text — no stage directions, no camera instructions, no scene descriptions. Only what is spoken or written on screen.

Structure it using this model:

${modelInstructions}

${durationGuidance}

Use specific facts, names and language from the source material. Avoid generic phrases and clichés. Write in English.`;
}

async function streamGenerate(res, promptArgs) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  try {
    const [embedding, profile] = await Promise.all([
      embed(promptArgs.topic),
      getProfile(promptArgs.company),
    ]);
    const chunks = await search(embedding, promptArgs.company);

    if (!chunks || chunks.length === 0) {
      res.write(`data: ${JSON.stringify({ error: 'No data found for this company. Try ingesting more content first.' })}\n\n`);
      return res.end();
    }

    const context = chunks.map((c, i) => `[${i + 1}] ${c.content}`).join('\n\n');
    const prompt = buildPrompt({ ...promptArgs, context, profile });
    const maxTokens = getMaxTokens(promptArgs.duration);

    const stream = anthropic.messages.stream({
      model: 'claude-sonnet-4-6',
      max_tokens: maxTokens,
      messages: [{ role: 'user', content: prompt }],
    });

    for await (const event of stream) {
      if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
        res.write(`data: ${JSON.stringify({ text: event.delta.text })}\n\n`);
      }
    }

    res.write('data: [DONE]\n\n');
    res.end();
  } catch (err) {
    res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  }
}

app.get('/api/companies', async (req, res) => {
  try {
    const { data, error } = await supabase.from('documents').select('company').not('company', 'is', null);
    if (error) throw error;
    const companies = [...new Set(data.map(d => d.company))].sort();
    res.json({ companies });
  } catch (err) {
    console.error('companies error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/generate', async (req, res) => {
  const { topic, company, model, channel, duration } = req.body;
  if (!topic || !company || !model || !channel || !duration) {
    return res.status(400).json({ error: 'Missing required fields' });
  }
  await streamGenerate(res, { topic, company, model, channel, duration });
});

app.post('/api/refine', async (req, res) => {
  const { topic, company, model, channel, duration, original, feedback } = req.body;
  if (!topic || !company || !model || !feedback || !original) {
    return res.status(400).json({ error: 'Missing required fields' });
  }
  await streamGenerate(res, { topic, company, model, channel, duration, original, feedback });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Storyline Widget running at http://localhost:${PORT}`));
