"""
Agents Package

Multi-agent orchestration layer for mental health analysis.

Agents:
- ClassificationAgent : Detects emotion from user text
- CrisisAgent         : Assesses crisis risk and triggers alerts
- RAGAgent            : Retrieves wellness recommendations
- AgentOrchestrator   : Coordinates all agents in parallel pipeline

Execution flow:
    User Text
        │
        ▼
    ClassificationAgent          (Step 1 - blocking)
        │
        ├──────────────────┐
        ▼                  ▼
    CrisisAgent        RAGAgent  (Step 2 - parallel)
        │                  │
        └──────────────────┘
                │
                ▼
        AgentOrchestrator        (Step 3 - aggregation)
                │
                ▼
        Unified Response

Usage:
    from backend.agents.orchestrator import AgentOrchestrator
    orchestrator = AgentOrchestrator()
    result = await orchestrator.process(text="I feel anxious")
"""

from backend.agents.classification_agent import ClassificationAgent
from backend.agents.crisis_agent import CrisisAgent
from backend.agents.rag_agent import RAGAgent
from backend.agents.orchestrator import AgentOrchestrator

__all__ = [
    "ClassificationAgent",
    "CrisisAgent",
    "RAGAgent",
    "AgentOrchestrator",
]