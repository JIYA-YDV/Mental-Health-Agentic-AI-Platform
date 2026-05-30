# backend/agents/__init__.py
"""
Agents Package

Multi-agent orchestration layer for mental health analysis.

Agents:
- ClassificationAgent : Detects emotion from user text
- CrisisAgent         : Assesses crisis risk and triggers alerts
- RAGAgent            : Retrieves wellness recommendations
- WellnessAgent       : Generates personalized wellness tips
- AgentOrchestrator   : Coordinates all agents in parallel pipeline

Usage:
    from backend.agents.orchestrator import AgentOrchestrator
    orchestrator = AgentOrchestrator()
    result = await orchestrator.process(text="I feel anxious")
"""

__all__ = [
    "ClassificationAgent",
    "CrisisAgent",
    "RAGAgent",
    "WellnessAgent",
    "AgentOrchestrator",
]