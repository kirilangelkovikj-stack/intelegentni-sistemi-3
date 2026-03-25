"""
Пребарувач на програмски проблеми - Codeforces
Семантичко пребарување со OpenAI embeddings + FAISS
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from openai import OpenAI

# ── опционално: FAISS ──────────────────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS не е инсталиран – ќе се користи numpy fallback (побавно).")

# ─────────────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "text-embedding-3-small"
EMBED_DIM     = 1536
BATCH_SIZE    = 100          # броj описи по API повик
INDEX_PATH    = "cf_index.faiss"
META_PATH     = "cf_meta.pkl"
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ПОМОШНИ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌  Не е поставен OPENAI_API_KEY.\n"
            "    Постави го со:  export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Враќа (N, EMBED_DIM) float32 матрица."""
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        # замени празни редови со placeholder
        batch = [t.strip() or "no description" for t in batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs = [item.embedding for item in resp.data]
        all_vecs.extend(vecs)
        print(f"  Embeddings: {min(i+BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.array(all_vecs, dtype="float32")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ВЧИТУВАЊЕ НА ДАТАСЕТОТ
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    """
    Прифаќа CSV или JSON/JSONL.
    Очекува колони (или кој клуч ги има): name, statement/description, tags, rating/difficulty
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"❌  Датасетот не е пронајден: {path}")

    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".json", ".jsonl"):
        try:
            df = pd.read_json(path)
        except Exception:
            df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Неподдржан формат: {ext}")

    # ── нормализирај имиња на колони ──────────────────────────────────────
    df.columns = [c.lower().strip() for c in df.columns]

    # Пронајди ја колоната со опис на задачата
    desc_candidates = ["statement", "description", "problem_statement", "body", "text", "content"]
    desc_col = next((c for c in desc_candidates if c in df.columns), None)
    if desc_col is None:
        # земи ја третата текстуална колона ако нема позната
        text_cols = df.select_dtypes(include="object").columns.tolist()
        desc_col = text_cols[2] if len(text_cols) > 2 else text_cols[0]
        print(f"⚠️  Колоната за опис не е пронајдена – се користи: '{desc_col}'")
    df["_description"] = df[desc_col].fillna("").astype(str)

    # Пронајди ги останатите корисни колони
    name_col = next((c for c in ["name", "title", "problem_name"] if c in df.columns), None)
    df["_name"] = df[name_col].fillna("Unnamed").astype(str) if name_col else "Unnamed"

    rating_col = next((c for c in ["rating", "difficulty", "points"] if c in df.columns), None)
    df["_rating"] = df[rating_col].fillna(0).astype(int) if rating_col else 0

    tags_col = next((c for c in ["tags", "categories", "topics"] if c in df.columns), None)
    df["_tags"] = df[tags_col].fillna("").astype(str) if tags_col else ""

    url_col = next((c for c in ["url", "link", "problem_url"] if c in df.columns), None)
    df["_url"] = df[url_col].fillna("").astype(str) if url_col else ""

    print(f"✅  Вчитани {len(df):,} задачи од '{path}'")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ИЗГРАДБА НА ИНДЕКСОТ
# ══════════════════════════════════════════════════════════════════════════════

def build_index(dataset_path: str, force_rebuild: bool = False):
    """
    Изгради (или вчитај кеширан) FAISS индекс.
    Враќа (index_or_matrix, metadata_list).
    """
    if not force_rebuild and Path(INDEX_PATH).exists() and Path(META_PATH).exists():
        print("📂  Вчитувам кеширан индекс…")
        meta = pickle.load(open(META_PATH, "rb"))
        if FAISS_AVAILABLE:
            idx = faiss.read_index(INDEX_PATH)
        else:
            idx = np.load(INDEX_PATH + ".npy")
        print(f"✅  Индексот е подготвен ({len(meta):,} задачи).")
        return idx, meta

    print("🔨  Градам нов индекс…")
    client = get_client()
    df     = load_dataset(dataset_path)

    # Текст за embedding = наслов + опис (max 500 chars за брзина)
    texts = (df["_name"] + ". " + df["_description"].str[:500]).tolist()

    print(f"📡  Генерирам embeddings за {len(texts):,} задачи (може да трае неколку минути)…")
    vecs = embed_texts(client, texts)

    # Зачувај метаподатоци
    meta = df[["_name", "_description", "_rating", "_tags", "_url"]].to_dict("records")
    pickle.dump(meta, open(META_PATH, "wb"))

    # Зачувај индекс
    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(EMBED_DIM)   # Inner-Product ~ cosine на нормализирани вектори
        faiss.normalize_L2(vecs)
        index.add(vecs)
        faiss.write_index(index, INDEX_PATH)
    else:
        # Numpy fallback – зачувај матрица
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs  = vecs / np.where(norms == 0, 1, norms)
        np.save(INDEX_PATH + ".npy", vecs)
        index = vecs

    print(f"✅  Индексот е изграден и зачуван ({len(meta):,} задачи).")
    return index, meta


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ПРЕБАРУВАЊЕ
# ══════════════════════════════════════════════════════════════════════════════

def search(
    query: str,
    index,
    meta: list[dict],
    client: OpenAI,
    top_k: int = 10,
    min_rating: int = 0,
    max_rating: int = 9999,
    tag_filter: str = "",
) -> list[dict]:
    """
    Семантичко пребарување + опционален филтер по тежина/таг.
    """
    # Embed на барањето
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query.strip()])
    q_vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

    if FAISS_AVAILABLE:
        faiss.normalize_L2(q_vec)
        scores, indices = index.search(q_vec, min(top_k * 5, len(meta)))
        candidates = [(meta[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
    else:
        # Numpy dot-product (вектори се веќе нормализирани)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        sims = (index @ q_vec.T).flatten()
        top_idx = np.argsort(-sims)[: top_k * 5]
        candidates = [(meta[i], float(sims[i])) for i in top_idx]

    # Примени филтри
    results = []
    for m, score in candidates:
        r = int(m["_rating"]) if m["_rating"] else 0
        if not (min_rating <= r <= max_rating):
            continue
        if tag_filter and tag_filter.lower() not in m["_tags"].lower():
            continue
        results.append({**m, "score": score})
        if len(results) == top_k:
            break

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ПРИКАЖУВАЊЕ НА РЕЗУЛТАТИ
# ══════════════════════════════════════════════════════════════════════════════

def display_results(results: list[dict], query: str):
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  🔍  Резултати за: \"{query}\"  ({len(results)} најдени)")
    print(sep)

    if not results:
        print("  Нема резултати. Обиди се со поинаков опис.")
        return

    for i, r in enumerate(results, 1):
        rating_str = f"⭐ {r['_rating']}" if r["_rating"] else "⭐ N/A"
        tags_str   = r["_tags"][:60] + "…" if len(r["_tags"]) > 60 else r["_tags"]
        desc_str   = r["_description"][:200].replace("\n", " ") + "…"
        url_str    = f"\n     🔗  {r['_url']}" if r["_url"] else ""

        print(f"\n  [{i:02d}]  {r['_name']}")
        print(f"       {rating_str}   🏷  {tags_str or 'нема тагови'}")
        print(f"       📄  {desc_str}{url_str}")
        print(f"       📊  Сличност: {r['score']:.4f}")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ИНТЕРАКТИВНА ЈАМКА (CLI)
# ══════════════════════════════════════════════════════════════════════════════

def interactive_loop(index, meta: list[dict]):
    client = get_client()
    print("\n" + "═" * 70)
    print("   🎯  ПРЕБАРУВАЧ НА CODEFORCES ЗАДАЧИ")
    print("   Команди:  'q' за излез  |  'r 800-1600' за рејтинг филтер")
    print("             't graph' за таг филтер  |  'top 5' за број резултати")
    print("═" * 70)

    top_k      = 10
    min_rating = 0
    max_rating = 9999
    tag_filter = ""

    while True:
        try:
            raw = input("\n🔎  Внеси опис/идеја:  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Збогум!")
            break

        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit", "излез"):
            print("👋  Збогум!")
            break

        # Парсирај специјални команди
        if raw.lower().startswith("r "):
            parts = raw[2:].split("-")
            try:
                min_rating = int(parts[0])
                max_rating = int(parts[1]) if len(parts) > 1 else 9999
                print(f"✔️  Рејтинг филтер: {min_rating}–{max_rating}")
            except ValueError:
                print("⚠️  Формат: r 800-1600")
            continue

        if raw.lower().startswith("t "):
            tag_filter = raw[2:].strip()
            print(f"✔️  Таг филтер: '{tag_filter}'  (внеси 't' за да го исчистиш)")
            continue

        if raw.lower().startswith("top "):
            try:
                top_k = int(raw.split()[1])
                print(f"✔️  Прикажувај {top_k} резултати")
            except (ValueError, IndexError):
                print("⚠️  Формат: top 5")
            continue

        results = search(
            query=raw,
            index=index,
            meta=meta,
            client=client,
            top_k=top_k,
            min_rating=min_rating,
            max_rating=max_rating,
            tag_filter=tag_filter,
        )
        display_results(results, raw)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Codeforces семантички пребарувач")
    parser.add_argument("--dataset",  default="codeforces_problems.csv",
                        help="Патека до CSV/JSON датасетот")
    parser.add_argument("--rebuild",  action="store_true",
                        help="Принудно пресметај ги embeddings повторно")
    parser.add_argument("--query",    default="",
                        help="Директно пребарување (без интерактивен режим)")
    parser.add_argument("--top",      type=int, default=10,
                        help="Број на резултати (default: 10)")
    args = parser.parse_args()

    index, meta = build_index(args.dataset, force_rebuild=args.rebuild)

    if args.query:
        client  = get_client()
        results = search(args.query, index, meta, client, top_k=args.top)
        display_results(results, args.query)
    else:
        interactive_loop(index, meta)
