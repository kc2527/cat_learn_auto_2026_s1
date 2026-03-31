from datetime import datetime, timedelta
import os
import re
import sys
import pandas as pd


def load_file_summary(path, fallback_date_key):
    df = pd.read_csv(path)
    n_rows = int(df.shape[0])

    start_ts = None
    end_ts = None
    if n_rows > 0 and "ts_iso" in df.columns:
        ts = pd.to_datetime(df["ts_iso"], errors="coerce").dropna()
        if not ts.empty:
            start_ts = ts.iloc[0].to_pydatetime()
            end_ts = ts.iloc[-1].to_pydatetime()

    if start_ts is None:
        start_day = datetime.strptime(fallback_date_key, "%Y_%m_%d").date()
        start_ts = datetime.combine(start_day, datetime.min.time())
    if end_ts is None:
        end_ts = start_ts

    return {
        "n_rows": n_rows,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }


def resolve_session(dir_data,
                    subject,
                    n_total,
                    resume_window=timedelta(hours=12),
                    new_session_cooldown=timedelta(hours=8),
                    now=None,
                    task_tag=None):
    now = datetime.now() if now is None else now
    today = now.date()

    if task_tag is None:
        fn_re_part = re.compile(
            rf"^sub_{re.escape(str(subject))}_sess_(\d{{3}})_part_(\d{{3}})_date_(\d{{4}}_\d{{2}}_\d{{2}})_data\.csv$"
        )
    else:
        fn_re_part = re.compile(
            rf"^sub_{re.escape(str(subject))}_task_{re.escape(str(task_tag))}_sess_(\d{{3}})_part_(\d{{3}})_date_(\d{{4}}_\d{{2}}_\d{{2}})_data\.csv$"
        )

    def build_filename(session_num, part_num, date_key):
        if task_tag is None:
            return (
                f"sub_{subject}_sess_{int(session_num):03d}_part_{int(part_num):03d}"
                f"_date_{date_key}_data.csv"
            )
        return (
            f"sub_{subject}_task_{task_tag}_sess_{int(session_num):03d}_part_{int(part_num):03d}"
            f"_date_{date_key}_data.csv"
        )

    session_records = {}

    for fn in os.listdir(dir_data):
        full = os.path.join(dir_data, fn)
        if not os.path.isfile(full):
            continue

        m_part = fn_re_part.match(fn)
        if m_part:
            session_num_i = int(m_part.group(1))
            part_num_i = int(m_part.group(2))
            date_key_i = m_part.group(3)
            try:
                file_summary = load_file_summary(full, date_key_i)
            except Exception:
                continue
            session_records.setdefault(session_num_i, []).append({
                "part_num": part_num_i,
                "date_key": date_key_i,
                "start_ts": file_summary["start_ts"],
                "end_ts": file_summary["end_ts"],
                "fn": fn,
                "full": full,
                "n_rows": file_summary["n_rows"]
            })
            continue

    sessions = []
    for s_num, parts in session_records.items():
        parts_sorted = sorted(parts, key=lambda p: (p["part_num"], p["start_ts"]))
        first_part = min(parts_sorted, key=lambda p: (p["part_num"], p["start_ts"]))
        n_done_session = int(sum(p["n_rows"] for p in parts_sorted))
        latest_part = max(parts_sorted, key=lambda p: p["end_ts"])
        sessions.append({
            "session_num": int(s_num),
            "parts": parts_sorted,
            "n_done": n_done_session,
            "last_ts": latest_part["end_ts"],
            "max_part": int(max(p["part_num"] for p in parts_sorted)),
            "session_day": first_part["start_ts"].date(),
            "is_complete": bool(n_done_session >= n_total),
        })
    sessions.sort(key=lambda s: s["last_ts"], reverse=True)

    incomplete_sessions = [s for s in sessions if not s["is_complete"]]
    if len(incomplete_sessions) > 1:
        conflict_desc = ", ".join(
            f"sess_{s['session_num']:03d} (day {s['session_day']}, last save {s['last_ts']:%Y-%m-%d %H:%M})"
            for s in incomplete_sessions)
        print(
            "Multiple incomplete sessions were found for this participant.\n"
            f"Resolve them manually before continuing: {conflict_desc}"
        )
        sys.exit()

    recent_incomplete = None
    completed_sessions = [s for s in sessions if s["is_complete"]]
    last_completed = max(
        completed_sessions, key=lambda s: s["last_ts"], default=None)
    session_num = max((s["session_num"] for s in sessions), default=0) + 1
    part_num = 1
    today_key = now.strftime("%Y_%m_%d")
    f_name = build_filename(session_num, part_num, today_key)
    full_path = os.path.join(dir_data, f_name)
    n_done = 0

    for sess in incomplete_sessions:
        if (now - sess["last_ts"]) <= resume_window:
            recent_incomplete = sess
            break

    if recent_incomplete is not None:
        session_num = recent_incomplete["session_num"]
        part_num = recent_incomplete["max_part"] + 1
        n_done = recent_incomplete["n_done"]
        today_key = now.strftime("%Y_%m_%d")
        f_name = build_filename(session_num, part_num, today_key)
        full_path = os.path.join(dir_data, f_name)
        remaining = n_total - n_done
        print(
            f"Resuming your last incomplete session "
            f"(last saved {recent_incomplete['last_ts']:%Y-%m-%d %H:%M})."
        )
        print(
            f"You have {remaining} trials remaining in this session. "
            "Please try to finish today’s trials so you can stay on track."
        )
    else:
        completed_today = next(
            (s for s in completed_sessions if s["session_day"] == today), None)
        if completed_today is not None:
            next_ok = datetime.combine(
                today + timedelta(days=1), datetime.min.time())
            print(
                "You already completed a session assigned to today.\n"
                f"Please wait until {next_ok:%Y-%m-%d %H:%M} before starting another."
            )
            sys.exit()

        if last_completed is not None:
            age = now - last_completed["last_ts"]
            if age < new_session_cooldown:
                next_ok = last_completed["last_ts"] + new_session_cooldown
                print(
                    "It has been fewer than 8 hours since your last completed session.\n"
                    f"Please wait until {next_ok:%Y-%m-%d %H:%M} before trying again."
                )
                sys.exit()

        session_num = max((s["session_num"] for s in sessions), default=0) + 1
        part_num = 1
        today_key = now.strftime("%Y_%m_%d")
        f_name = build_filename(session_num, part_num, today_key)
        full_path = os.path.join(dir_data, f_name)
        n_done = 0

    if n_done >= n_total:
        print(f"Session is already complete ({n_done} trials). Aborting.")
        sys.exit()

    print(
        f"Subject: {subject} | Session: {session_num} | Part: {part_num} | Date: {today_key} | "
        f"Resuming at trial: {n_done}"
    )

    return {
        "session_num": session_num,
        "part_num": part_num,
        "today_key": today_key,
        "f_name": f_name,
        "full_path": full_path,
        "n_done": n_done,
    }
