
import argparse
import json
import os
import re
import time
from typing import List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine

try:
    import openai
except Exception:
    openai = None


# --------------------------- Utilities -------------------------------------

def log(msg: str):
    print(f"[agent] {msg}")


def read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        log(f"Loaded CSV: {path} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        log(f"CSV not found: {path}")
        return None
    except Exception as e:
        log(f"CSV load error ({path}): {e}")
        return None


def read_sqlite(db_path: str, table: Optional[str] = None, query: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(db_path):
        log(f"DB not found: {db_path}")
        return None
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        if query:
            df = pd.read_sql_query(query, engine)
        elif table:
            df = pd.read_sql_table(table, engine)
        else:
            # attempt to list tables and pick the first
            with engine.connect() as conn:
                res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [r[0] for r in res]
            if not tables:
                log(f"No tables found in DB: {db_path}")
                return None
            log(f"No table specified. Using first table: {tables[0]}")
            df = pd.read_sql_table(tables[0], engine)
        log(f"Loaded DB: {db_path} ({len(df)} rows)")
        return df
    except Exception as e:
        log(f"DB load error ({db_path}): {e}")
        return None


def read_json(path: str) -> Optional[pd.DataFrame]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # raw could be list of dicts or dict of lists; try to normalize
        if isinstance(raw, dict):
            # if values are lists and same length -> DataFrame. else wrap dict
            try:
                df = pd.DataFrame(raw)
            except Exception:
                df = pd.DataFrame([raw])
        else:
            df = pd.DataFrame(raw)
        log(f"Loaded JSON: {path} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        log(f"JSON not found: {path}")
        return None
    except Exception as e:
        log(f"JSON load error ({path}): {e}")
        return None


def normalize_email(email: str) -> str:
    if pd.isna(email):
        return ""
    return str(email).strip().lower()


def merge_sources(dfs: List[pd.DataFrame], key: str = "email") -> pd.DataFrame:
    # Standardize key column name (lowercase)
    cleaned = []
    for i, df in enumerate(dfs):
        if df is None:
            continue
        df = df.copy()
        if key not in df.columns:
            # try common alternatives
            for alt in ["email_address", "Email", "e-mail", "E-mail"]:
                if alt in df.columns:
                    df.rename(columns={alt: key}, inplace=True)
                    break
        if key in df.columns:
            df[key] = df[key].map(normalize_email)
        else:
            # if no key, create one using index (not ideal)
            df[key] = [f"_row_{i}_{j}" for j in range(len(df))]
        cleaned.append(df)
    if not cleaned:
        return pd.DataFrame()
    # Merge iteratively with outer join on key
    merged = cleaned[0]
    for df in cleaned[1:]:
        merged = pd.merge(merged, df, on=key, how="outer", suffixes=(None, None))
    # coalesce columns with same semantic name (very simple rule)
    merged = coalesce_columns(merged)
    log(f"Merged dataframes -> {len(merged)} unique keys")
    return merged


def coalesce_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If there are columns like first_name_x, first_name_y -> coalesce to first_name
    new_df = df.copy()
    pattern = re.compile(r"(?P<base>.+)_(x|y|[0-9]+)$")
    cols = list(new_df.columns)
    grouped = {}
    for c in cols:
        m = pattern.match(c)
        if m:
            base = m.group("base")
            grouped.setdefault(base, []).append(c)
    for base, cs in grouped.items():
        targets = [base] + cs
        # create base if doesn't exist
        if base not in new_df.columns:
            new_df[base] = None
        for c in cs:
            new_df[base] = new_df[base].combine_first(new_df[c])
            new_df.drop(columns=[c], inplace=True)
    return new_df


def search_by_keywords(df: pd.DataFrame, keywords: List[str], fields: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    kw_regex = re.compile("|".join([re.escape(k.strip()) for k in keywords if k.strip()]), re.IGNORECASE)
    def row_matches(row):
        for f in fields:
            if f in row and pd.notna(row[f]):
                if kw_regex.search(str(row[f])):
                    return True
        return False
    mask = df.apply(row_matches, axis=1)
    matched = df[mask].copy()
    log(f"Keyword search found {len(matched)} rows matching {keywords}")
    return matched


# --------------------------- Email generation ------------------------------

LOCAL_TEMPLATE = ("Subject: {subject}\n\nDear {first_name} {last_name},\n\n"
                  "{body}\n\nBest regards,\nYour Company")


def build_prompt_for_contact(contact: dict, keywords: List[str], instruction: str) -> str:
    # Give the LLM a compact, structured prompt that includes contact info and the keywords.
    snippet = []
    for k, v in contact.items():
        snippet.append(f"{k}: {v}")
    snippet_str = "\n".join(snippet)
    prompt = (
        "You are a helpful but succinct assistant tasked with composing a short, personalized email to a client.\n"
        f"Keywords to mention (or reason to reach out): {', '.join(keywords)}\n"
        "Make the email professional, 3-6 sentences, and include a clear call to action. Use the contact info below.\n"
        f"Contact info:\n{snippet_str}\n\nInstruction: {instruction}\n\nReturn a JSON object with fields: subject, body."
    )
    return prompt


def call_openai_generate(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 300) -> Tuple[Optional[str], Optional[str]]:
    if openai is None:
        log("openai package not installed. Cannot call API.")
        return None, None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("OPENAI_API_KEY not set. Skipping OpenAI call.")
        return None, None
    openai.api_key = api_key
    try:
        # Chat completion style
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.6,
        )
        content = resp["choices"][0]["message"]["content"]
        # attempt to extract JSON object from content
        try:
            # find first { ... } block
            jtext = re.search(r"\{.*\}", content, re.DOTALL)
            if jtext:
                obj = json.loads(jtext.group(0))
                return obj.get("subject"), obj.get("body")
        except Exception:
            pass
        # fallback: split by first blank line -> subject/body heuristic
        lines = content.strip().splitlines()
        subject = lines[0] if lines else ""
        body = "\n".join(lines[1:])
        return subject, body
    except Exception as e:
        log(f"OpenAI call failed: {e}")
        return None, None


# --------------------------- Main agent flow -------------------------------


def agent_flow(
    csv_path: Optional[str],
    db_path: Optional[str],
    json_path: Optional[str],
    db_table: Optional[str],
    keywords: List[str],
    fields: List[str],
    key: str,
    dry_run: bool = True,
    instruction: str = "Write a polite, concise outreach email tailored to the contact.",
    batch_size: int = 20,
    model: str = "gpt-4o-mini",
):
    # 1) Load sources in preferred order, but allow multiple sources to be available
    dfs = []
    if csv_path:
        df_csv = read_csv(csv_path)
        if df_csv is not None:
            dfs.append(df_csv)
    if db_path:
        df_db = read_sqlite(db_path, table=db_table)
        if df_db is not None:
            dfs.append(df_db)
    if json_path:
        df_json = read_json(json_path)
        if df_json is not None:
            dfs.append(df_json)

    if not dfs:
        log("No data loaded from any source. Exiting.")
        return None

    merged = merge_sources(dfs, key=key)

    # 2) Search by keywords
    matched = search_by_keywords(merged, keywords, fields)
    if matched.empty:
        log("No contacts matched keywords. Exiting.")
        return None

    # 3) For each matched contact, prepare contact dict and generate email
    results = []
    total = len(matched)
    rows = matched.to_dict(orient="records")
    for i, row in enumerate(rows):
        contact = {k: v for k, v in row.items()}
        email_key = contact.get(key, "")
        # Build prompt
        prompt = build_prompt_for_contact(contact, keywords, instruction)
        subject = None
        body = None
        if dry_run:
            # simple local template: subject derived from keywords
            subject = f"Regarding {', '.join(keywords[:2])}"
            first_name = contact.get("first_name") or contact.get("fname") or ""
            last_name = contact.get("last_name") or contact.get("lname") or ""
            body = (
                f"Hi {first_name} {last_name},\n\nI wanted to reach out about {', '.join(keywords)}. "
                "Could we schedule a quick call to discuss?"
            )
        else:
            # call OpenAI
            sub, bod = call_openai_generate(prompt, model=model)
            if sub or bod:
                subject, body = sub, bod
            else:
                # fallback to local template
                subject = f"Follow up on {', '.join(keywords[:2])}"
                body = (
                    f"Hi {contact.get('first_name','')},\n\nI wanted to follow up about {', '.join(keywords)}. "
                    "Are you available for a 15-minute call next week?"
                )
        results.append({
            key: email_key,
            "first_name": contact.get("first_name", ""),
            "last_name": contact.get("last_name", ""),
            "subject": subject,
            "body": body,
        })
        # batch sleep to be polite if not dry-run
        if not dry_run and (i + 1) % batch_size == 0:
            log(f"Sleeping 1s after {i+1} API calls to avoid burst limits")
            time.sleep(1)

    out_df = pd.DataFrame(results)
    out_path = "output_emails.csv"
    out_df.to_csv(out_path, index=False)
    log(f"Wrote {len(out_df)} generated emails to {out_path}")
    return out_df


# --------------------------- CLI -------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Semi-agentic multi-source email generator")
    p.add_argument("--csv", help="Path to CSV file", dest="csv")
    p.add_argument("--db", help="Path to sqlite .db file", dest="db")
    p.add_argument("--table", help="Table name to read from sqlite", dest="table")
    p.add_argument("--json", help="Path to JSON file", dest="json")
    p.add_argument("--keywords", help="Comma-separated keywords to search for", required=True)
    p.add_argument("--fields", help="Comma-separated fields to search in (defaults to ['notes','company','first_name','last_name'])")
    p.add_argument("--key", help="Primary key to merge on (defaults to 'email')", default="email")
    p.add_argument("--dry-run", help="Don't call OpenAI API; use local template", action="store_true")
    p.add_argument("--instruction", help="Instruction for the LLM", default="Write a polite, concise outreach email tailored to the contact.")
    p.add_argument("--model", help="Model to use for OpenAI calls", default="gpt-4o-mini")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if args.fields:
        fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    else:
        fields = ["notes", "company", "first_name", "last_name"]
    df = agent_flow(
        csv_path=args.csv,
        db_path=args.db,
        json_path=args.json,
        db_table=args.table,
        keywords=keywords,
        fields=fields,
        key=args.key,
        dry_run=args.dry_run,
        instruction=args.instruction,
        model=args.model,
    )
    if df is not None:
        log("Done.")
    else:
        log("No output generated.")
