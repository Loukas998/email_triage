"""How much does the judge agree with the human? Ordinary code with a right answer — so it has tests.

"Positive" here means *fail*: the thing the judge exists to catch. So
    TPR = of the summaries the human failed, how many did the judge also fail  (fails caught)
    TNR = of the summaries the human passed, how many did the judge also pass  (passes kept)
Raw agreement alone misleads when failures are rare: a judge that always says pass scores 95%
agreement on a set with 5% failures and catches none of them (Hamel Husain, evals FAQ).
"""

from pydantic import BaseModel


class Agreement(BaseModel):
    n: int
    agreed: int
    fails: int # Human FAIL
    fails_caught: int
    passes: int # Human PASS
    passes_kept: int

    @property
    def agreement(self) -> float:
        return self.agreed / self.n

    @property
    def tpr(self) -> float | None:
        """None when the human failed nothing — there is nothing to catch, so the rate is undefined."""
        return None if self.fails == 0 else self.fails_caught / self.fails 

    @property
    def tnr(self) -> float | None:
        return None if self.passes == 0 else self.passes_kept / self.passes


def measure(human_passed: list[bool], judge_passed: list[bool]) -> Agreement:
    if len(human_passed) != len(judge_passed):
        raise ValueError(f"{len(human_passed)} human labels but {len(judge_passed)} judge verdicts")
    if not human_passed:
        raise ValueError("no labels to measure")

    pairs = list(zip(human_passed, judge_passed))
    return Agreement(
        n=len(pairs),
        agreed=sum(h == j for h, j in pairs),
        fails=sum(not h for h, _ in pairs),
        fails_caught=sum(not h and not j for h, j in pairs),
        passes=sum(h for h, _ in pairs),
        passes_kept=sum(h and j for h, j in pairs)
    )


def describe(a: Agreement) -> str:
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"
    return (
        f"{a.n} labels · agree {a.agreed}/{a.n} ({a.agreement:.0%}) · "
        f"TPR {pct(a.tpr)} ({a.fails_caught}/{a.fails} fails caught) · "
        f"TNR {pct(a.tnr)} ({a.passes_kept}/{a.passes} passes kept)"
    )