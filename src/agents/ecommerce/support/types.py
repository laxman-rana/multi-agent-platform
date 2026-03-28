import operator
from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    order: Optional[dict]
    messages: Annotated[Sequence[BaseMessage], operator.add]
