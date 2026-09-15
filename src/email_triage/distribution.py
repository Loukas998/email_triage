from collections import Counter
from inspect import cleandoc

from ollama import chat

from email_triage.triage import MODEL, SYSTEM_PROMPT, EmailTriage



def triage_at(email: str, temperature: float) -> EmailTriage:
    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email}
        ],
        format=EmailTriage.model_json_schema(),
        options={"temperature": temperature}
    )

    return EmailTriage.model_validate_json(response.message.content)


def distribution(email: str, runs: int = 20, temperature: float = 0.8) -> Counter:
    return Counter(triage_at(email, temperature).category for _ in range(runs))


AMBIGUOUS = (
    "hi, i was charged twice for my subscription this month "
    "and the app also keeps crashing when i open settings. "
    "can someone sort this out"
)

CLEAR = (
    "Please update the credit card on file for my account. "
    "The current one expires next month."
)

if __name__ == "__main__":
    for name, email in [("AMBIGUOUS", AMBIGUOUS), ("CLEAR", CLEAR)]:
        counts = distribution(email)
        total = sum(counts.values())
        print(f"\n{name}")
        for category, n in counts.most_common():
            print(f"  {category:<10} {n:>2}/{total}  {'#' * n}")
