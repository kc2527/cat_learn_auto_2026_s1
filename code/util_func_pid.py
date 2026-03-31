import re
import sys
from psychopy import core, event, visual


def _prompt_for_pid_text(win, pid_digits, is_valid_pid, invalid_id_message):
    pid_input = ""
    pid_error = ""
    pid_prompt = visual.TextStim(win,
                                 text="",
                                 color='white',
                                 height=28,
                                 wrapWidth=1500)

    while True:
        pid_prompt.text = (
            f"Enter {pid_digits}-digit Participant ID\n\n"
            f"ID: {pid_input or '___'}\n\n"
            "Press ENTER to continue, BACKSPACE to edit, ESC to quit.\n"
            f"{pid_error}"
        )
        pid_prompt.draw()
        win.flip()

        keys = event.getKeys()
        for k in keys:
            if k == "escape":
                win.close()
                core.quit()
                sys.exit()
            if k == "backspace":
                pid_input = pid_input[:-1]
                pid_error = ""
                continue
            if k in {"return", "num_enter"}:
                if len(pid_input) != pid_digits:
                    pid_error = f"\nInvalid ID format. Enter exactly {pid_digits} digits."
                    continue
                if not is_valid_pid(pid_input):
                    pid_error = invalid_id_message
                    continue
                return pid_input

            digit = None
            if re.fullmatch(r"[0-9]", k):
                digit = k
            else:
                m = re.fullmatch(r"num_([0-9])", k)
                if m:
                    digit = m.group(1)
            if digit is not None and len(pid_input) < pid_digits:
                pid_input += digit
                pid_error = ""


def prompt_for_pid(win, pid_digits, condition_by_subject):
    subject = _prompt_for_pid_text(
        win,
        pid_digits,
        lambda pid: pid in condition_by_subject,
        "\nThis Participant ID is not enrolled for this study.",
    )
    condition = condition_by_subject[subject]
    return subject, condition


def prompt_for_pid_in_set(win, pid_digits, allowed_subject_ids):
    subject = _prompt_for_pid_text(
        win,
        pid_digits,
        lambda pid: pid in allowed_subject_ids,
        "\nThis Participant ID is not enrolled for this study.",
    )
    return subject


def prompt_for_day(win):
    prompt = visual.TextStim(win,
                             text="",
                             color='white',
                             height=28,
                             wrapWidth=1500)

    while True:
        prompt.text = (
            "Select Day:\n\n"
            "1 = baseline\n"
            "2 = post1\n"
            "3 = post2\n\n"
            "Press 1/2/3 to continue, ESC to quit."
        )
        prompt.draw()
        win.flip()

        keys = event.getKeys()
        for k in keys:
            if k == "escape":
                win.close()
                core.quit()
                sys.exit()
            if k in {"1", "num_1"}:
                return "baseline"
            if k in {"2", "num_2"}:
                return "post1"
            if k in {"3", "num_3"}:
                return "post2"
