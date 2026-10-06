# Copyright Sierra

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union


class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]


class Task(BaseModel):
    info: Dict[str, Any]


class EnvState(BaseModel):
    state: Dict[str, Any]


class EnvResponse(BaseModel):
    observation: List[str]
    reward: float
    done: bool


class EnvResetResponse(BaseModel):
    observation: List[str]


class SolveResult(BaseModel):
    reward: float
    messages: List[Dict[str, Any]]