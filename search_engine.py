"""
Codeforces Semantic Search Engine
Run: python search_engine.py
Then open: http://localhost:5000
"""

import os, json, pickle, re, csv, math, http.server, threading, urllib.parse, webbrowser
from pathlib import Path

# ── Optional numpy for faster cosine similarity ──────────────────────────────
try:
    import numpy as np
    NUMPY = True
except ImportError:
    NUMPY = False

CACHE_PATH   = "cf_embeddings.pkl"
EMBED_MODEL  = "voyage-code-2"   # best for code/algo content; falls back below
ANTHROPIC_EMBED_AVAILABLE = False  # Anthropic embeddings beta flag

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    problems = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse tags from Python list literal string e.g. "['dp', 'graphs']"
            raw_tags = row.get("tags", "") or ""
            tags = re.findall(r"'([^']+)'", raw_tags)

            rating_raw = row.get("rating", "") or ""
            try:
                rating = int(float(rating_raw))
            except (ValueError, TypeError):
                rating = 0

            contest_id = row.get("contestId", "") or ""
            index      = row.get("index", "") or ""
            url = f"https://codeforces.com/problemset/problem/{contest_id}/{index}" if contest_id else ""

            problems.append({
                "id":      row.get("id", ""),
                "title":   row.get("title", "Untitled"),
                "rating":  rating,
                "tags":    tags,
                "tags_str": ", ".join(tags),
                "url":     url,
                "search_text": f"{row.get('title','')} {' '.join(tags)}",
            })
    return problems


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING  (Claude / Anthropic API)
# ─────────────────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set.\n"
            "Export it: export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    return key


def embed_texts_anthropic(texts: list[str], api_key: str) -> list[list[float]]:
    """Call Anthropic embeddings endpoint."""
    import urllib.request
    BATCH = 96
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        payload = json.dumps({
            "model": "voyage-code-2",
            "input": batch,
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/embeddings",
            data=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "anthropic-beta": "embeddings-2024-01-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                for item in data["data"]:
                    all_vecs.append(item["embedding"])
            print(f"  Embedded {min(i+BATCH, len(texts))}/{len(texts)}", end="\r")
        except Exception as e:
            raise RuntimeError(f"Embedding API error: {e}")
    print()
    return all_vecs


def cosine_sim(a: list[float], b: list[float]) -> float:
    if NUMPY:
        va, vb = np.array(a), np.array(b)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom else 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  KEYWORD FALLBACK SEARCH  (works without embeddings / API)
# ─────────────────────────────────────────────────────────────────────────────

def keyword_search(query: str, problems: list[dict],
                   min_rating: int, max_rating: int,
                   tag_filter: str, top_k: int) -> list[dict]:
    tokens = re.findall(r'\w+', query.lower())
    results = []
    for p in problems:
        r = p["rating"]
        if r and not (min_rating <= r <= max_rating):
            continue
        if tag_filter and tag_filter.lower() not in p["tags_str"].lower():
            continue
        haystack = p["search_text"].lower()
        score = sum(1 for t in tokens if t in haystack)
        if score > 0:
            results.append({**p, "score": score / len(tokens)})
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
#  SEMANTIC SEARCH (embedding-based)
# ─────────────────────────────────────────────────────────────────────────────

def semantic_search(query: str, problems: list[dict], embeddings: list,
                    api_key: str, min_rating: int, max_rating: int,
                    tag_filter: str, top_k: int) -> list[dict]:
    import urllib.request
    payload = json.dumps({
        "model": "voyage-code-2",
        "input": [query.strip() or "algorithm problem"],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/embeddings",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-beta": "embeddings-2024-01-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            q_vec = data["data"][0]["embedding"]
    except Exception:
        return keyword_search(query, problems, min_rating, max_rating, tag_filter, top_k)

    candidates = []
    for p, vec in zip(problems, embeddings):
        r = p["rating"]
        if r and not (min_rating <= r <= max_rating):
            continue
        if tag_filter and tag_filter.lower() not in p["tags_str"].lower():
            continue
        sim = cosine_sim(q_vec, vec)
        candidates.append({**p, "score": sim})

    candidates.sort(key=lambda x: -x["score"])
    return candidates[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD / LOAD EMBEDDING CACHE
# ─────────────────────────────────────────────────────────────────────────────

def load_or_build_cache(problems: list[dict], api_key: str, force: bool = False):
    cache = Path(CACHE_PATH)
    if not force and cache.exists():
        print(f"Loading cached embeddings from {CACHE_PATH}…")
        with open(cache, "rb") as f:
            data = pickle.load(f)
        if len(data["embeddings"]) == len(problems):
            print(f"Cache OK ({len(problems):,} problems).")
            return data["embeddings"]
        print("Cache size mismatch – rebuilding…")

    print(f"Building embeddings for {len(problems):,} problems (this takes ~5-10 min)…")
    texts = [p["search_text"] for p in problems]
    embeddings = embed_texts_anthropic(texts, api_key)
    with open(cache, "wb") as f:
        pickle.dump({"embeddings": embeddings}, f)
    print(f"Embeddings cached to {CACHE_PATH}")
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
#  HTML UI
# ─────────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="mk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CF Search</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Sora:wght@400;500;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0d0f14; --surface: #13151c; --border: #1e2130;
    --accent: #5b8af0; --accent2: #3dffa0;
    --text: #e8eaf0; --muted: #6b7280; --tag-bg: #1a2035;
    --tag-text: #7aa0f5; --card-hover: #181b28;
    --rating-low: #3dffa0; --rating-mid: #f0c040; --rating-hard: #f05a5a;
  }
  body { font-family:'Sora',sans-serif; background:var(--bg); color:var(--text);
         min-height:100vh; display:flex; flex-direction:column; }
  header { padding: 2.5rem 2rem 1.5rem; border-bottom:1px solid var(--border); }
  .logo { font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:var(--accent2);
          letter-spacing:0.08em; margin-bottom:0.4rem; }
  h1 { font-size:2rem; font-weight:600; color:var(--text); line-height:1.2; }
  h1 span { color:var(--accent); }

  .search-bar { display:flex; gap:0.75rem; margin-top:1.5rem; flex-wrap:wrap; }
  .search-bar input[type=text], .search-bar select, .search-bar input[type=number] {
    background:var(--surface); border:1px solid var(--border); color:var(--text);
    border-radius:8px; padding:0.65rem 1rem; font-family:'Sora',sans-serif;
    font-size:0.95rem; outline:none; transition:border 0.2s;
  }
  .search-bar input[type=text]:focus, .search-bar select:focus { border-color:var(--accent); }
  #query { flex:1; min-width:220px; }
  #tag-filter { width:160px; }
  .rating-group { display:flex; align-items:center; gap:0.4rem; }
  .rating-group input { width:80px; }
  .rating-group span { color:var(--muted); font-size:0.85rem; }
  button { background:var(--accent); border:none; color:#fff; border-radius:8px;
           padding:0.65rem 1.4rem; font-family:'Sora',sans-serif; font-size:0.95rem;
           font-weight:500; cursor:pointer; transition:opacity 0.15s; white-space:nowrap; }
  button:hover { opacity:0.85; }
  button:disabled { opacity:0.5; cursor:default; }
  .mode-toggle { display:flex; gap:0.5rem; margin-top:1rem; }
  .mode-btn { background:var(--surface); border:1px solid var(--border); color:var(--muted);
              border-radius:6px; padding:0.35rem 0.9rem; font-size:0.82rem; cursor:pointer; }
  .mode-btn.active { border-color:var(--accent); color:var(--accent); background:rgba(91,138,240,0.08); }

  main { flex:1; padding:1.5rem 2rem; max-width:1100px; width:100%; margin:0 auto; }
  #status { color:var(--muted); font-size:0.88rem; margin-bottom:1rem;
            font-family:'JetBrains Mono',monospace; }
  #results { display:flex; flex-direction:column; gap:0.75rem; }

  .card { background:var(--surface); border:1px solid var(--border); border-radius:12px;
          padding:1.1rem 1.4rem; transition:background 0.15s, border-color 0.15s;
          position:relative; }
  .card:hover { background:var(--card-hover); border-color:#2a3050; }
  .card-top { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }
  .card-title { font-weight:500; font-size:1rem; color:var(--text); text-decoration:none; }
  .card-title:hover { color:var(--accent); }
  .card-meta { display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap; margin-top:0.55rem; }
  .rating-badge { font-family:'JetBrains Mono',monospace; font-size:0.8rem; font-weight:500;
                  padding:2px 8px; border-radius:4px; }
  .r-low  { background:rgba(61,255,160,0.12); color:var(--rating-low); }
  .r-mid  { background:rgba(240,192,64,0.12); color:var(--rating-mid); }
  .r-hard { background:rgba(240,90,90,0.12);  color:var(--rating-hard); }
  .r-none { background:rgba(107,114,128,0.15); color:var(--muted); }
  .tags { display:flex; flex-wrap:wrap; gap:0.35rem; }
  .tag { background:var(--tag-bg); color:var(--tag-text); border-radius:4px;
         padding:2px 7px; font-size:0.75rem; }
  .score-bar { position:absolute; right:1.4rem; top:50%; transform:translateY(-50%);
               display:flex; flex-direction:column; align-items:center; gap:2px; }
  .score-val { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--muted); }
  .score-ring { width:36px; height:36px; }

  .empty { text-align:center; padding:3rem 1rem; color:var(--muted); }
  .empty b { display:block; font-size:1.1rem; color:var(--text); margin-bottom:0.5rem; }
  .spinner { display:inline-block; width:18px; height:18px; border:2px solid var(--border);
             border-top-color:var(--accent); border-radius:50%; animation:spin 0.7s linear infinite;
             vertical-align:middle; margin-right:6px; }
  @keyframes spin { to { transform:rotate(360deg); } }

  .setup-banner { background:rgba(91,138,240,0.08); border:1px solid rgba(91,138,240,0.25);
                  border-radius:10px; padding:1rem 1.3rem; margin-bottom:1.5rem; font-size:0.88rem; }
  .setup-banner code { font-family:'JetBrains Mono',monospace; background:rgba(255,255,255,0.07);
                       border-radius:4px; padding:1px 6px; }
</style>
</head>
<body>
<header>
  <div class="logo">&lt;/&gt; codeforces search</div>
  <h1>Find your next <span>problem</span></h1>
  <div class="search-bar">
    <input id="query" type="text" placeholder="Describe the problem idea… e.g. shortest path in grid with obstacles" autocomplete="off">
    <input id="tag-input" type="text" placeholder="tag filter" style="width:140px">
    <div class="rating-group">
      <input id="min-r" type="number" placeholder="min ★" value="" min="0" max="4000">
      <span>–</span>
      <input id="max-r" type="number" placeholder="max ★" value="" min="0" max="4000">
    </div>
    <input id="topk" type="number" value="10" min="1" max="50" style="width:72px" title="Results count">
    <button id="search-btn" onclick="doSearch()">Search</button>
  </div>
  <div class="mode-toggle">
    <button class="mode-btn active" id="btn-keyword" onclick="setMode('keyword')">Keyword</button>
    <button class="mode-btn" id="btn-semantic" onclick="setMode('semantic')">Semantic (AI)</button>
  </div>
</header>
<main>
  <div id="setup-note" class="setup-banner" style="display:none">
    <b>Semantic search needs an Anthropic API key.</b>
    Set it before starting the server:<br>
    <code>export ANTHROPIC_API_KEY="sk-ant-..."</code>
    then run <code>python search_engine.py --dataset CodeForces.csv</code>
  </div>
  <div id="status">Ready — type a query and press Search.</div>
  <div id="results"></div>
</main>

<script>
let mode = 'keyword';
const API_READY = __API_READY__;

function setMode(m) {
  mode = m;
  document.getElementById('btn-keyword').classList.toggle('active', m==='keyword');
  document.getElementById('btn-semantic').classList.toggle('active', m==='semantic');
  if (m==='semantic' && !API_READY) {
    document.getElementById('setup-note').style.display='block';
  } else {
    document.getElementById('setup-note').style.display='none';
  }
}

document.getElementById('query').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});

function ratingClass(r) {
  if (!r) return 'r-none';
  if (r <= 1400) return 'r-low';
  if (r <= 2000) return 'r-mid';
  return 'r-hard';
}

function scoreArc(s) {
  const pct = Math.max(0, Math.min(1, s));
  const r = 14, cx = 18, cy = 18;
  const angle = pct * 2 * Math.PI - Math.PI/2;
  const x2 = cx + r * Math.cos(angle);
  const y2 = cy + r * Math.sin(angle);
  const large = pct > 0.5 ? 1 : 0;
  const color = pct > 0.7 ? '#3dffa0' : pct > 0.4 ? '#f0c040' : '#6b7280';
  return `<svg class="score-ring" viewBox="0 0 36 36">
    <circle cx="18" cy="18" r="14" fill="none" stroke="#1e2130" stroke-width="3"/>
    ${pct > 0.01 ? `<path d="M18 4 A14 14 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}"
      fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round"/>` : ''}
  </svg>`;
}

function renderResults(data) {
  const el = document.getElementById('results');
  if (!data.results || data.results.length === 0) {
    el.innerHTML = '<div class="empty"><b>No results found</b>Try different keywords or remove filters.</div>';
    return;
  }
  document.getElementById('status').textContent =
    `${data.results.length} result${data.results.length!==1?'s':''} · ${data.mode} search · ${data.time_ms}ms`;
  el.innerHTML = data.results.map((p, i) => {
    const rc = ratingClass(p.rating);
    const rLabel = p.rating ? `★ ${p.rating}` : '★ N/A';
    const tags = (p.tags || []).slice(0, 6).map(t =>
      `<span class="tag">${t}</span>`).join('');
    const scoreDisplay = data.mode === 'semantic'
      ? `<div class="score-bar">${scoreArc(p.score)}<span class="score-val">${(p.score*100).toFixed(0)}%</span></div>`
      : '';
    return `<div class="card">
      <div class="card-top">
        <div>
          <a class="card-title" href="${p.url}" target="_blank" rel="noopener">${i+1}. ${p.title}</a>
          <div class="card-meta">
            <span class="rating-badge ${rc}">${rLabel}</span>
            <div class="tags">${tags}</div>
          </div>
        </div>
      </div>
      ${scoreDisplay}
    </div>`;
  }).join('');
}

async function doSearch() {
  const query = document.getElementById('query').value.trim();
  if (!query) return;
  const btn = document.getElementById('search-btn');
  btn.disabled = true;
  document.getElementById('status').innerHTML = '<span class="spinner"></span>Searching…';
  document.getElementById('results').innerHTML = '';

  const params = new URLSearchParams({
    q: query,
    mode: mode,
    min_r: document.getElementById('min-r').value || 0,
    max_r: document.getElementById('max-r').value || 9999,
    tag:   document.getElementById('tag-input').value.trim(),
    top_k: document.getElementById('topk').value || 10,
  });

  try {
    const resp = await fetch('/api/search?' + params);
    const data = await resp.json();
    if (data.error) {
      document.getElementById('status').textContent = 'Error: ' + data.error;
    } else {
      renderResults(data);
    }
  } catch(e) {
    document.getElementById('status').textContent = 'Server error: ' + e.message;
  }
  btn.disabled = false;
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
#  HTTP SERVER
# ─────────────────────────────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):
    problems   = []
    embeddings = []
    api_key    = ""
    has_embeds = False

    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            html = HTML.replace("__API_READY__", "true" if self.has_embeds else "false")
            self._send(200, "text/html", html.encode())
        elif parsed.path == "/api/search":
            self._handle_search(parsed.query)
        else:
            self._send(404, "text/plain", b"Not found")

    def _handle_search(self, qs: str):
        import time
        params   = urllib.parse.parse_qs(qs)
        query    = (params.get("q", [""])[0]).strip()
        mode     = params.get("mode", ["keyword"])[0]
        min_r    = int(params.get("min_r", [0])[0] or 0)
        max_r    = int(params.get("max_r", [9999])[0] or 9999)
        tag      = params.get("tag", [""])[0].strip()
        top_k    = min(int(params.get("top_k", [10])[0] or 10), 50)

        t0 = time.time()
        try:
            if mode == "semantic" and self.has_embeds:
                results = semantic_search(
                    query, self.problems, self.embeddings,
                    self.api_key, min_r, max_r, tag, top_k)
            else:
                mode = "keyword"
                results = keyword_search(query, self.problems, min_r, max_r, tag, top_k)
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            self._send(200, "application/json", body)
            return

        elapsed = int((time.time() - t0) * 1000)
        out = {
            "results":  results,
            "mode":     mode,
            "time_ms":  elapsed,
        }
        self._send(200, "application/json", json.dumps(out, ensure_ascii=False).encode())

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Codeforces Semantic Search")
    parser.add_argument("--dataset",  default="CodeForces.csv",
                        help="Path to the Kaggle Codeforces CSV")
    parser.add_argument("--port",     type=int, default=5000)
    parser.add_argument("--embed",    action="store_true",
                        help="Build / use semantic embeddings (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--rebuild",  action="store_true",
                        help="Force rebuild embedding cache")
    args = parser.parse_args()

    print("Loading dataset…")
    problems = load_dataset(args.dataset)
    print(f"Loaded {len(problems):,} problems from '{args.dataset}'")

    embeddings = []
    has_embeds = False
    api_key    = os.environ.get("ANTHROPIC_API_KEY", "")

    if args.embed and api_key:
        try:
            embeddings = load_or_build_cache(problems, api_key, force=args.rebuild)
            has_embeds = True
            print("Semantic search enabled.")
        except Exception as e:
            print(f"Warning: embeddings unavailable – {e}")
            print("Falling back to keyword search.")
    elif args.embed and not api_key:
        print("Warning: --embed requires ANTHROPIC_API_KEY. Using keyword search.")
    else:
        print("Keyword search mode. Add --embed to enable AI semantic search.")

    Handler.problems   = problems
    Handler.embeddings = embeddings
    Handler.api_key    = api_key
    Handler.has_embeds = has_embeds

    server = http.server.HTTPServer(("", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\nServer running at {url}")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
