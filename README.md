# Пребарувач на програмски проблеми

**Учесници:** Филип Veljanovskи, Кирил Ангелковиќ  
**Тип на проект:** Пребарување (Search)

## Опис

Овој проект претставува пребарувач за Codeforces програмски задачи кој им овозможува на корисниците да најдат релевантни задачи според опис, клучни зборови или проблемска идеја. Системот анализира наслови и тагови на задачи и го наоѓа најсличниот резултат.

Поддржува два режими:
- **Keyword Search** — брзо пребарување по клучни зборови, работи без API клуч
- **Semantic (AI) Search** — паметно пребарување базирано на значење, користи Anthropic API

**Датасет:** Kaggle – Codeforces Problem Dataset (~10,900 задачи)

---

## Инсталација и користење

### Барања

- Python 3.10 или понов
- Интернет конекција (за семантичко пребарување)

### Чекор 1 — Клонирај го репото

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### Чекор 2 — Стартувај го (keyword режим, без API клуч)

```bash
python search_engine.py --dataset CodeForces.csv
```

Отвори го прелистувачот на: **http://localhost:5000**

### Чекор 3 — Семантичко пребарување (опционално)

Постави го Anthropic API клучот пред стартување:

**Mac / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python search_engine.py --dataset CodeForces.csv --embed
```

**Windows:**
```cmd
set ANTHROPIC_API_KEY=sk-ant-...
python search_engine.py --dataset CodeForces.csv --embed
```

Прв пат кога се стартува со `--embed`, системот генерира embeddings за сите задачи (~5-10 мин). После тоа се зачувуваат во `cf_embeddings.pkl` и следниот пат се вчитуваат веднаш.

---

## Карактеристики

- Пребарување по клучни зборови и по семантичко значење
- Филтрирање по тежина (рејтинг) и по таг
- Прикажување на директни линкови до Codeforces задачите
- Визуелен резултат за сличност (само во semantic режим)
- Работи без инсталирање на дополнителни Python пакети

---

## Структура на проектот

```
cf-search/
├── search_engine.py     # Главна апликација (веб сервер + пребарувач)
├── CodeForces.csv       # Датасет со ~10,900 Codeforces задачи
├── cf_embeddings.pkl    # Кеш на embeddings (се создава автоматски)
└── README.md            # Овој документ
```

---

## Пример пребарувања

| Барање | Режим |
|---|---|
| `shortest path grid obstacles` | keyword или semantic |
| `count connected components undirected graph` | semantic |
| `dp knapsack items weight` | keyword или semantic |
| `string pattern matching` | keyword |

---

## Технологии

- **Python** — стандардна библиотека (http.server, csv, json, pickle)
- **Anthropic API** — Voyage embeddings за семантичка сличност
- **HTML/CSS/JS** — веб интерфејс (без framework)
- **Датасет:** [Codeforces Problem Dataset на Kaggle](https://www.kaggle.com/datasets/muonneutrino/codeforces-problem-set)
