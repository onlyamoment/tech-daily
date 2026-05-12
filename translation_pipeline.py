#!/usr/bin/env python3
"""Translation pipeline runner for tech-intel.

Scans data items for source_status=fetched and translation_status=pending/queued,
marks them queued, and outputs a clear translation backlog.

This module manages queue state and can perform LLM translation via the
configured Anthropic-compatible API (e.g., MiniMax). Translation falls back
to extracting embedded Chinese sections when available.

Status semantics:
  pending      → eligible for translation, not yet in queue
  queued       → acknowledged by pipeline, awaiting LLM call
  in_progress  → LLM call in flight (set externally)
  completed    → zh_full_translation populated
  failed       → LLM call failed; reason recorded in translation_error
"""
import json
import os
import re
import argparse
import urllib.request
import sys
import io
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout on Windows to avoid UnicodeEncodeError in reports
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SOURCES_DIR = ROOT / "sources"

VALID_TRANSLATION_STATUSES = {"pending", "queued", "in_progress", "completed", "failed"}


def _item_source_filename(item: dict) -> str | None:
    """Guess the source filename on disk for an item."""
    item_id = str(item.get("id", "")).strip()
    title = str(item.get("title", "")).strip()
    if not item_id or not title:
        return None
    slug = re.sub(r"[^a-zA-Z0-9\s-]", " ", title)
    slug = re.sub(r"[-\s]+", "-", slug).strip("-").lower()
    slug = slug[:40].rstrip("-") or "article"
    ident = f"{item_id}-{slug}"
    itype = item.get("type", "article")
    url = str(item.get("url", "")).strip()
    is_github = itype == "github" or (url and "github.com" in url)
    # Check known extensions
    for ext in (".README.md", ".README.zh-CN.md", ".txt"):
        cand = SOURCES_DIR / f"{ident}{ext}"
        if cand.exists():
            return str(cand.relative_to(ROOT))
    # Also check without strict ident matching (fallback)
    for f in SOURCES_DIR.iterdir():
        if f.name.startswith(f"{item_id}-"):
            return str(f.relative_to(ROOT))
    return None


def sync_disk_sources(datas):
    """If a source file exists on disk but item source_status is not 'fetched', fix it."""
    fixed = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for data in datas:
        for item in data.get("items", []):
            ss = str(item.get("source_status", "")).strip()
            if ss == "fetched":
                continue
            existing_path = str(item.get("source_text_path", "")).strip()
            if existing_path:
                p = ROOT / existing_path
                if p.exists():
                    item["source_status"] = "fetched"
                    item["source_fetched_at"] = item.get("source_fetched_at") or now
                    item["source_fetch_error"] = ""
                    fixed += 1
                    continue
            guessed = _item_source_filename(item)
            if guessed:
                item["source_status"] = "fetched"
                item["source_text_path"] = guessed
                item["source_fetched_at"] = item.get("source_fetched_at") or now
                item["source_fetch_error"] = ""
                item["translation_source"] = item.get("translation_source") or "GitHub README.md"
                fixed += 1
    return fixed


def load_data_files():
    files = sorted(DATA_DIR.glob("*.json"))
    result = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        data["_file"] = f
        result.append(data)
    return result


def scan_backlog(datas):
    """Return (backlog, stuck, blocked) where:
    - backlog: items source_status=fetched, translation_status in (pending, queued, '')
    - stuck: items translation_status=queued but no zh_full_translation (likely stalled)
    - blocked: items source_status=failed or translation_status=failed and no zh_full_translation
    """
    backlog = []
    stuck = []
    blocked = []
    for data in datas:
        date = data.get("date", "unknown")
        for item in data.get("items", []):
            ss = str(item.get("source_status", "")).strip()
            ts = str(item.get("translation_status", "")).strip()
            has_trans = bool(item.get("zh_full_translation", "").strip())

            if ss == "fetched" and ts in ("pending", "queued", ""):
                backlog.append({
                    "date": date,
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "type": item.get("type", ""),
                    "url": item.get("url", ""),
                    "source_text_path": item.get("source_text_path", ""),
                    "translation_status": ts or "pending",
                })

            # Track queued items without a completed body separately.  This is
            # intentionally not an ``elif``: queued items are also part of the
            # backlog, but they need an explicit stale/stuck warning so the UI
            # does not imply that a background translator is actively running.
            if ts == "queued" and not has_trans:
                stuck.append({
                    "date": date,
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "source_text_path": item.get("source_text_path", ""),
                    "translation_error": item.get("translation_error", ""),
                })
            # Track failed items that will never auto-progress without intervention
            if (ss.startswith("failed") or ts == "failed") and not has_trans:
                blocked.append({
                    "date": date,
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "type": item.get("type", ""),
                    "url": item.get("url", ""),
                    "source_status": ss,
                    "translation_status": ts,
                    "source_fetch_error": item.get("source_fetch_error", ""),
                })
    return backlog, stuck, blocked


def mark_queued(datas, backlog_items):
    """Mark backlog items as queued and return count."""
    updated = 0
    backlog_keys = {(b["date"], b["id"]) for b in backlog_items}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for data in datas:
        date = data.get("date", "unknown")
        for item in data.get("items", []):
            if (date, item.get("id")) in backlog_keys:
                ts = str(item.get("translation_status", "")).strip()
                if ts in ("pending", ""):
                    item["translation_status"] = "queued"
                    item["last_translated_at"] = now
                    item["translation_model"] = "pipeline-queued"
                    updated += 1
    return updated


def save_datas(datas):
    for data in datas:
        f = data.pop("_file", None)
        if f:
            f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _try_extract_chinese_section(text: str) -> str | None:
    """If source text contains a Chinese section marker or high Chinese density, extract it."""
    marker = '<a name="chinese"></a>'
    idx = text.find(marker)
    if idx == -1:
        m = re.search(r'<a\s+name=["\']chinese["\'][^>]*>', text, re.I)
        if m:
            idx = m.start()
    if idx != -1:
        section = text[idx:]
        section = re.sub(r'^<a\s+name=["\']chinese["\'][^>]*>\s*', '', section, flags=re.I)
        return section.strip()
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if len(text) > 500 and cn_chars / len(text) > 0.25:
        return text.strip()
    return None


def _pick_translation_candidate(datas):
    """Pick the highest-priority item for translation."""
    candidates = []
    for data in datas:
        date = data.get("date", "unknown")
        for item in data.get("items", []):
            ss = str(item.get("source_status", "")).strip()
            ts = str(item.get("translation_status", "")).strip()
            has_trans = bool(item.get("zh_full_translation", "").strip())
            if ss != "fetched" or has_trans:
                continue
            score = 0
            if ts == "queued":
                score += 10
            elif ts == "pending":
                score += 5
            else:
                score += 1
            candidates.append((score, date, item))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][2]


def recover_stale_queue(datas):
    """Reset queued items stuck for >4h back to pending."""
    recovered = 0
    now = datetime.now(timezone.utc)
    for data in datas:
        for item in data.get("items", []):
            if str(item.get("translation_status", "")).strip() != "queued":
                continue
            lta = item.get("last_translated_at", "")
            if not lta:
                continue
            try:
                dt = datetime.fromisoformat(lta.replace("Z", "+00:00"))
                if (now - dt).total_seconds() > 4 * 3600:
                    item["translation_status"] = "pending"
                    item["translation_error"] = f"stale queue: reset after {(now-dt).total_seconds()/3600:.1f}h"
                    recovered += 1
            except Exception:
                pass
    return recovered


def retry_failed_sources(datas, retry_blocked=False):
    """Reset failed source_status back to pending for retry.

    Args:
        retry_blocked: if True, also reset items with translation_status=failed
                       (i.e. items whose LLM translation failed) back to pending
                       so they can be retried in a future pipeline cycle.
    """
    updated = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for data in datas:
        for item in data.get("items", []):
            ss = str(item.get("source_status", "")).strip()
            ts = str(item.get("translation_status", "")).strip()
            has_trans = bool(item.get("zh_full_translation", "").strip())
            if ss.startswith("failed") and not has_trans:
                item["source_status"] = "pending"
                item["source_fetch_error"] = f"reset for retry at {now}"
                item["translation_status"] = "pending"
                item["translation_error"] = ""
                updated += 1
            elif retry_blocked and ts == "failed" and not has_trans:
                # Also reset translation-failed items so they re-enter the queue
                item["translation_status"] = "pending"
                item["translation_error"] = f"translation retry reset at {now}"
                updated += 1
    return updated


def translate_one(datas, dry_run=False, force_llm=False):
    """Attempt to translate exactly one candidate item.
    
    Args:
        dry_run: if True, describe what would be done but don't write changes.
        force_llm: if True, skip Chinese-section extraction and call LLM directly.
    """
    item = _pick_translation_candidate(datas)
    if item is None:
        print("No translation candidate found.")
        return 0
    title = item.get("title", "")
    item_id = item.get("id", "")
    print(f"Candidate: [{item_id}] {title[:60]}")
    path = str(item.get("source_text_path", "")).strip()
    if not path:
        print("  -> No source_text_path. Cannot translate.")
        return 0
    p = ROOT / path
    if not p.exists():
        print(f"  -> Source file missing: {path}")
        return 0
    source_text = p.read_text(encoding="utf-8", errors="ignore")
    if len(source_text) < 100:
        print(f"  -> Source too short ({len(source_text)} chars).")
        return 0
    extracted = _try_extract_chinese_section(source_text)
    if extracted and len(extracted) > 500:
        print(f"  -> Extracted Chinese section ({len(extracted)} chars).")
        trans = extracted
        source_note = item.get("translation_source", "") or "GitHub README.md（内含官方中文 section）"
        model = "pipeline-extracted"
    elif force_llm:
        # force_llm means caller wants LLM translation even without embedded Chinese
        print(f"  -> No Chinese section found; force_llm=True — falling through to LLM call.")
        return 0  # Hand off to run_llm_translation instead (caller must invoke it separately)
    else:
        print(f"  -> No Chinese section found in source ({len(source_text)} chars).")
        print("  -> External LLM translation required. Source preview (first 600 chars):")
        preview = source_text[:600].replace("\n", " ")
        print(f"     {preview}...")
        print("  -> Integrate an LLM API here to auto-generate zh_full_translation.")
        return 0
    if dry_run:
        print("  -> Dry run: would write translation and mark completed.")
        return 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    item["zh_full_translation"] = trans
    item["translation_status"] = "completed"
    item["last_translated_at"] = now
    item["translation_model"] = model
    if source_note:
        item["translation_source"] = source_note
    item["translation_error"] = ""
    print(f"  -> Written zh_full_translation ({len(trans)} chars). Status=completed.")
    return 1


def _build_translation_prompt(title: str, source_text: str) -> str:
    return f"""你是一位技术编辑。请把以下技术文档/README翻译成自然、可读的中文技术文章。

翻译原则：
1. 先理解，再改写。不逐句硬翻。
2. 保留事实，不编造。不确定的内容不要补。
3. 中文自然，少用「革命性」「赋能」「生态」「范式」这类空词，少用「不仅……而且……」「这标志着……」等AI味句式。
4. 技术名词保留英文，必要时补一句中文解释。
5. 每篇至少回答：它是什么？解决什么问题？值得关注的点是什么？适合谁看/用？

推荐结构（可灵活调整）：
### 中文标题
开头 1-2 段：自然说明这是什么，别像新闻通稿。
#### 主要内容
用短段落或列表概括要点。
#### 为什么值得看
具体说明价值，不夸大。
#### 适合谁
说明读者/使用场景。

长度：500-900 中文字。

以下是需要翻译的技术文档（项目：{title}）：
```
{source_text}
```
"""


def _call_llm(prompt: str) -> tuple[str | None, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    if not api_key:
        print("  -> ANTHROPIC_API_KEY not set.")
        return None, ""
    body = json.dumps({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read().decode("utf-8"))
        content = data.get("content", [])
        text = ""
        for block in content:
            if block.get("type") == "text":
                text += block.get("text", "")
        model = data.get("model", "unknown")
        return text, model
    except Exception as e:
        print(f"  -> LLM API error: {type(e).__name__}: {str(e)[:200]}")
        return None, ""


def run_llm_translation(datas, dry_run=False):
    """Translate one candidate using external LLM API."""
    item = _pick_translation_candidate(datas)
    if item is None:
        print("No translation candidate found.")
        return 0
    title = item.get("title", "")
    item_id = item.get("id", "")
    print(f"LLM Candidate: [{item_id}] {title[:60]}")

    path = str(item.get("source_text_path", "")).strip()
    if not path:
        print("  -> No source_text_path. Cannot translate.")
        return 0
    p = ROOT / path
    if not p.exists():
        print(f"  -> Source file missing: {path}")
        return 0

    source_text = p.read_text(encoding="utf-8", errors="ignore")
    if len(source_text) < 100:
        print(f"  -> Source too short ({len(source_text)} chars).")
        return 0

    MAX_SOURCE_CHARS = 10000
    truncated = False
    if len(source_text) > MAX_SOURCE_CHARS:
        source_text = source_text[:MAX_SOURCE_CHARS]
        truncated = True
        print(f"  -> Source truncated to {MAX_SOURCE_CHARS} chars for LLM.")

    prompt = _build_translation_prompt(title, source_text)

    if dry_run:
        print("  -> Dry run: would call LLM API.")
        return 0

    trans, model_name = _call_llm(prompt)
    if not trans:
        print("  -> LLM translation failed.")
        item["translation_status"] = "failed"
        item["translation_error"] = "LLM API call returned no translation"
        return 0

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    item["zh_full_translation"] = trans
    item["translation_status"] = "completed"
    item["last_translated_at"] = now
    item["translation_model"] = model_name
    item["translation_source"] = f"LLM全文翻译（基于 README 原文{'前'+str(MAX_SOURCE_CHARS)+'字符' if truncated else ''}）"
    item["translation_error"] = ""
    print(f"  -> Written zh_full_translation ({len(trans)} chars). Status=completed. Model={model_name}")
    return 1


def main():
    p = argparse.ArgumentParser(description="Translation pipeline queue runner")
    p.add_argument("--mark-queued", action="store_true", help="Mark pending items as queued")
    p.add_argument("--sync-sources", action="store_true", help="Fix source_status for items whose source files already exist on disk")
    p.add_argument("--dry-run", action="store_true", help="Don't write changes")
    p.add_argument("--verbose", "-v", action="store_true", help="Show all items including non-translatable")
    p.add_argument("--report", action="store_true", help="Print pipeline status report (default action; compatibility alias)")
    p.add_argument("--translate-one", action="store_true", help="Translate exactly one candidate (extracts Chinese section if available)")
    p.add_argument("--run-llm", action="store_true", help="Translate exactly one candidate using LLM API")
    p.add_argument("--retry-failed", action="store_true", help="Reset failed source_status back to pending for retry")
    p.add_argument("--run-one-cycle", action="store_true", help="Run one full cycle: sync sources, mark queued, translate one item, save results")
    args = p.parse_args()

    datas = load_data_files()

    if args.run_one_cycle:
        # Full cycle: sync → recover stale → scan → mark queued → translate one → save
        sync_fixed = sync_disk_sources(datas)
        if sync_fixed:
            print(f"=== Source Sync === Fixed {sync_fixed} item(s) with existing source files.")
        recovered = recover_stale_queue(datas)
        if recovered:
            print(f"=== Stale Queue Recovery === Recovered {recovered} stale queued item(s) to pending.")
        backlog, stuck, blocked = scan_backlog(datas)
        if backlog:
            marked = mark_queued(datas, backlog)
            print(f"=== Queue Marking === Marked {marked} item(s) as queued.")
        else:
            print("=== Queue Marking === No pending items to queue.")
        # Try extract-based translation first, then LLM
        done = translate_one(datas, dry_run=args.dry_run)
        if not done:
            llm_done = run_llm_translation(datas, dry_run=args.dry_run)
            done = llm_done
        if not args.dry_run and (sync_fixed or recovered or done):
            save_datas(datas)
            print("=== Saved === Changes persisted to data files.")
        elif args.dry_run:
            print("=== Dry Run === No changes saved.")
        else:
            print("=== No Translation Done === No candidate ready (all fetched items already translated, or no source file available).")
        return 0

    if args.retry_failed:
        reset = retry_failed_sources(datas)
        print(f"=== Retry Failed Sources ===\nReset {reset} failed item(s) to pending.")
        if not args.dry_run and reset:
            save_datas(datas)
            print("Changes saved.")
        else:
            print("Dry run: changes not saved.")
        return 0

    if args.run_llm:
        recovered = recover_stale_queue(datas)
        if recovered:
            print(f"=== Stale Queue Recovery ===\nRecovered {recovered} stale queued item(s).")
        done = run_llm_translation(datas, dry_run=args.dry_run)
        if not args.dry_run and (done or recovered):
            save_datas(datas)
            print("Changes saved.")
        else:
            print("Dry run: changes not saved.")
        return 0

    if args.translate_one:
        recovered = recover_stale_queue(datas)
        if recovered:
            print(f"=== Stale Queue Recovery ===\nRecovered {recovered} stale queued item(s).")
        # Try extract-based translation first; if that returns 0, fall through to LLM
        done = translate_one(datas, dry_run=args.dry_run)
        if not done:
            llm_done = run_llm_translation(datas, dry_run=args.dry_run)
            done = llm_done
        if not args.dry_run and (done or recovered):
            save_datas(datas)
            print("Changes saved.")
        else:
            print("Dry run: changes not saved.")
        return 0

    if args.sync_sources:
        fixed = sync_disk_sources(datas)
        print(f"=== Source Sync ===")
        print(f"Fixed {fixed} item(s) with existing source files on disk.")
        if not args.dry_run and fixed:
            save_datas(datas)
            print("Changes saved.")
        else:
            print("Dry run: changes not saved.")
        if not args.mark_queued and not args.verbose:
            return 0
        print()

    backlog, stuck, blocked = scan_backlog(datas)

    print("=== Translation Pipeline Status ===")
    print(f"scanned_files={len(datas)}  backlog={len(backlog)}  stuck_queued={len(stuck)}  blocked={len(blocked)}")

    # Summarise items not yet ready (no source fetched)
    not_ready = []
    for data in datas:
        date = data.get("date", "unknown")
        for item in data.get("items", []):
            ss = str(item.get("source_status", "")).strip()
            ts = str(item.get("translation_status", "")).strip()
            has_trans = bool(item.get("zh_full_translation", "").strip())
            if ss != "fetched" and ts not in ("completed", "failed") and not has_trans:
                not_ready.append({
                    "date": date,
                    "id": item.get("id"),
                    "title": item.get("title", "")[:50],
                    "source_status": ss or "pending",
                })

    if not_ready:
        print(f"\nNot ready for translation ({len(not_ready)} items, no source text fetched):")
        for b in not_ready[:10]:
            print(f"  [{b['date']}#{b['id']}] ss={b['source_status']:12s} | {b['title']}")
        if len(not_ready) > 10:
            print(f"  ... and {len(not_ready) - 10} more")

    if stuck:
        print(f"\n⚠  Stale queue entries ({len(stuck)} items — queued but no translation body):")
        for b in stuck:
            path = b.get("source_text_path", "")
            size_info = ""
            if path:
                p_ = ROOT / path
                if p_.exists():
                    size_info = f" ({p_.stat().st_size} bytes)"
                else:
                    size_info = " (file MISSING)"
            err = b.get("translation_error", "")
            err_note = f" | error={err}" if err else ""
            print(f"  [{b['date']}#{b['id']}] {b['title'][:50]}{size_info}{err_note}")

    if blocked:
        print(f"\n[x] Blocked items ({len(blocked)} -- failed and will not auto-progress):")
        for b in blocked:
            err = b.get("source_fetch_error", "")
            err_note = f" | error={err[:100]}" if err else ""
            print(f"  [{b['date']}#{b['id']}] ss={b['source_status']:12s} ts={b['translation_status']:10s} | {b['title'][:50]}{err_note}")
        print("  Run with --retry-failed to reset these to pending for re-fetch.")

    if not backlog:
        # List completed translations so the report is self-documenting
        completed = []
        for data in datas:
            date = data.get("date", "unknown")
            for item in data.get("items", []):
                ts = str(item.get("translation_status", "")).strip()
                has_trans = bool(item.get("zh_full_translation", "").strip())
                if has_trans and ts == "completed":
                    completed.append({
                        "date": date,
                        "id": item.get("id"),
                        "title": item.get("title", "")[:50],
                        "source": item.get("translation_source", "")[:40],
                    })
        if completed:
            print(f"\n[ok] Completed translations ({len(completed)} items -- all fetched items fully translated):")
            for c in completed:
                src_note = f" ({c['source']})" if c["source"] else ""
                print(f"  [{c['date']}#{c['id']}] {c['title']}{src_note}")
        print("\nNo items ready for translation queue.")
        if not stuck and not not_ready and not blocked:
            print("  All fetched items already have zh_full_translation.")
        if stuck or not_ready or blocked:
            print("  Use --retry-failed to reset blocked source-fetch failures.")
        return 0

    print(f"\nReady for translation queue ({len(backlog)} items):")
    for b in backlog:
        path = b.get("source_text_path", "")
        size_info = ""
        if path:
            p_ = ROOT / path
            if p_.exists():
                size_info = f" ({p_.stat().st_size} bytes)"
            else:
                size_info = " (file MISSING)"
        print(f"  [{b['date']}#{b['id']}] {b['type']} | {b['title'][:50]}{size_info}")

    if args.mark_queued:
        updated = mark_queued(datas, backlog)
        print(f"\nMarked {updated} item(s) as queued.")
        if not args.dry_run:
            save_datas(datas)
            print("Changes saved.")
        else:
            print("Dry run: changes not saved.")
    else:
        print("\nRun with --mark-queued to update status fields.")

    print("\nNOTE: This runner manages queue state and can extract embedded Chinese sections.")
    print("For sources without Chinese content, integrate an LLM API or run --translate-one with a custom hook.")
    print("After running this runner, invoke the LLM translator to populate zh_full_translation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
