from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from ollama import chat
from email_triage.prompts import PromptConfig

class EmailTriage(BaseModel):
    """The contract between model and the rest of the system"""

    category: Literal["billing", "technical", "account", "general"]
    summary: str = Field(max_length=200)
    confidence: int =Field(ge=1, le=5)


class Usage(BaseModel):
    """What one call cost. Recorded whether or not the answer was usable."""

    duration_ms: int
    """Wall-clock time for the whole call, model load included."""

    prompt_tokens: int
    """Tokens the model read: system prompt + email. Grows with every rule you add to the prompt."""

    output_tokens: int
    """Tokens the model wrote, thinking included. This is what num_predict caps."""

    @classmethod
    def from_response(cls, response) -> "Usage":
        """Ollama reports durations in nanoseconds and token counts as *_count. Convert once, here."""
        return cls(
            duration_ms=round(response.total_duration / 1_000_000),
            prompt_tokens=response.prompt_eval_count,
            output_tokens=response.eval_count
        )


class Triaged(BaseModel):
    """The answer, plus what it cost. The answer alone is not the whole result."""

    output: EmailTriage
    usage: Usage


class TriageError(Exception):
    """The model answered, but not in the shape the contract requires — usually because the token
    cap cut it off mid-thought. The call still cost something, so the cost travels with the error."""

    def __init__(self, cause: ValidationError, response):
        super().__init__(str(cause))
        self.usage = Usage.from_response(response)
        self.done_reason = response.done_reason
        """"length" when num_predict stopped it; "stop" when the model finished on its own."""
    


def triage(email: str, config: PromptConfig) -> Triaged:
    response = chat(
        model=config.model,
        messages=[
            {"role": "system", "content": config.system},
            {"role": "user", "content": email}
        ],
        format=EmailTriage.model_json_schema(),
        options={"temperature": config.temperature, "num_predict": config.num_predict},
        think=config.think
    )
    try:
        output = EmailTriage.model_validate_json(response.message.content)
    except ValidationError as err:
        raise TriageError(err, response) from err
    return Triaged(output=output, usage=Usage.from_response(response))

if __name__ == "__main__":
    result = triage(
        "hi, i was charged twice for my subscription this month "
        "and the app also keeps crashing when i open settings. "
        "can someone sort this out"
    )

    print(result)
    print(result.category)