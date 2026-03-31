import re
import sys
from psychopy import core, event, visual


def prompt_for_pid(win, pid_digits, condition_by_subject):
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
                if pid_input not in condition_by_subject:
                    pid_error = "\nThis Participant ID is not enrolled for this study."
                    continue
                subject = pid_input
                condition = condition_by_subject[subject]
                return subject, condition

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
