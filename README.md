# Proactive Social Robot Test Harness

A comprehensive testing framework for a proactive social robot behavior system that uses behavior trees and reinforcement learning to interact naturally with people.

## Overview

This system simulates a robot that:
- Detects and recognizes people through face detection
- Responds to human actions (waving, eating, phone use, etc.)
- Adapts behavior based on social context (calm vs. lively environments)
- Learns optimal interaction strategies through Q-learning
- Manages greetings and maintains interaction history

## System Components

- **`embodied_behaviour.py`** - Main robot module with state machine and behavior trees
- **`behaviour_trees.py`** - Low Priority (LP) and High Priority (HP) behavior tree implementations
- **`learning.py`** - Q-learning system for adaptive branch selection
- **`test_harness.py`** - Comprehensive test suite that simulates YARP perception ports
- **`run_complete_test.sh`** - Automated test execution script

## How It Works

The robot operates with two behavior modes:

**Low Priority (LP) Tree**: Engages when mutual gaze is detected
- Checks for known faces
- Verifies if person was already greeted today
- Responds to detected actions

**High Priority (HP) Tree**: More proactive engagement
- Detects known faces without requiring mutual gaze
- Initiates greetings with new people
- Responds to actions like waving

The system uses Q-learning to decide which behavior tree to activate based on the social context and interaction outcomes (measured by valence and arousal).

## Quick Start

### Prerequisites
- Python 3
- YARP (Yet Another Robot Platform)
- py_trees library

### Running Tests

**Standard 2-minute test:**
```bash
./run_complete_test.sh
```

**Clean run with fresh databases:**
```bash
./run_complete_test.sh --clean 5
```

**Interactive mode:**
```bash
./run_complete_test.sh --interactive
```

**Run module in separate terminal for visibility:**
```bash
./run_complete_test.sh --separate 10
```

## Test Scenarios

The test harness validates:
1. New person greetings with mutual gaze
2. Action-based responses (waving, eating, etc.)
3. Context-aware behavior selection
4. State transitions (ACTIVE ↔ INACTIVE)
5. Q-learning updates and convergence
6. Multi-person target selection

## Data Collection

The system logs all interactions to SQLite databases:
- `q_table.json` - Learned action values
- `last_greeted.db` - Greeting history
- `data_collection.db` - Detailed interaction logs

## Architecture

The system uses YARP ports for perception data:
- `/embodiedBehaviour/context:i` - Social context (calm/lively)
- `/embodiedBehaviour/valence_arousal:i` - Emotional feedback
- `/embodiedBehaviour/face_id:i` - Face detection and recognition
- `/embodiedBehaviour/actions:i` - Detected human actions

Mock RPC services simulate robot outputs:
- `/interactionInterface` - Animation commands
- `/acapelaSpeak/speech:i` - Speech synthesis

## Configuration

Key parameters in `embodied_behaviour.py`:
- `INACTIVE_TIMEOUT = 60.0` - Seconds before entering inactive state
- `ALPHA_REWARD = 1.0` - Valence weight in reward function
- `BETA_REWARD = 0.3` - Arousal weight in reward function
- `ETA_LEARNING = 0.1` - Q-learning rate

---

**Note**: This is a test harness for a YARP-based social robot system. The actual robot hardware/simulation is not included in this repository.
