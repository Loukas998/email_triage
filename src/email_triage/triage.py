from pydantic import BaseModel, Field
from typing import Literal
from ollama import chat
from email_triage.prompts import PromptConfig

class EmailTriage(BaseModel):
    """The contract between model and the rest of the system"""

    category: Literal["billing", "technical", "account", "general"]
    summary: str = Field(max_length=200)
    confidence: int =Field(ge=1, le=5)


def triage(email: str, config: PromptConfig) -> EmailTriage:
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
    return EmailTriage.model_validate_json(response.message.content)

if __name__ == "__main__":
    result = triage(
        "hi, i was charged twice for my subscription this month "
        "and the app also keeps crashing when i open settings. "
        "can someone sort this out"
    )

    print(result)
    print(result.category)