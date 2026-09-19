# IBVAP
## Intelligent Border Video Analytics Platform

> An AI-powered video analytics and intelligent incident monitoring platform for real-time surveillance, behavioral analysis, contextual threat detection, cross-camera entity correlation, and incident investigation.

---

## 📌 Overview

**IBVAP (Intelligent Border Video Analytics Platform)** is a computer-vision-based surveillance and video analytics system designed to transform conventional camera feeds into an intelligent, event-driven security monitoring platform.

The system combines:

- Real-time object detection
- Multi-object tracking
- Track lifecycle intelligence
- Spatial intelligence
- Behavioral analysis
- Early-warning detection
- Multi-event threat correlation
- Context-aware risk assessment
- Incident intelligence
- Evidence capture
- Persistent event storage
- Investigation reporting
- Cross-camera entity correlation
- Global entity identification
- Camera health monitoring
- System performance monitoring

Rather than treating every detection as an isolated event, IBVAP maintains **context across time**, allowing multiple observations involving the same tracked entity to be combined into a single evolving incident.

---

# 🎯 Objectives

The primary objective of IBVAP is to move surveillance systems beyond basic object detection toward **context-aware intelligent monitoring**.

The platform is designed to answer questions such as:

- Who or what was detected?
- Where did the entity move?
- How long was the entity present?
- Was a restricted area entered?
- Was suspicious behavior observed?
- Did multiple suspicious events occur sequentially?
- Did the same entity appear on another camera?
- How did the incident's risk level evolve?
- What evidence was captured?
- What happened before and during the incident?
- What is the current state of the incident?

---

# 🧠 Core Architecture

```text
                    ┌─────────────────────┐
                    │   Camera / Video    │
                    │       Sources       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Video Ingestion &   │
                    │ Camera Management   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ YOLO Object         │
                    │ Detection           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Multi-Object        │
                    │ Tracking / ByteTrack│
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      Track Lifecycle    Spatial Analysis   Appearance
        Intelligence       Intelligence       Features
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Cross-Camera Entity │
                    │ Correlation         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Behavior Analysis   │
                    │ & Early Warning      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Multi-Event Threat  │
                    │ Correlation          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Incident            │
                    │ Intelligence        │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       Risk Scoring       Evidence Capture   Event Storage
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Investigation &     │
                    │ Reporting           │
                    └─────────────────────┘

        ─────────────────────────────────────────────

             System Health & Performance Layer
        Camera Health • CPU • RAM • GPU • VRAM
        FPS • Processing Time • Errors • Uptime
