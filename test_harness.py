#!/usr/bin/env python3
"""
UNIFIED TEST HARNESS FOR EMBODIED BEHAVIOUR MODULE

Comprehensive test environment that simulates all YARP perception ports and
validates the complete behavior tree logic, state machine, and Q-learning.

SYSTEM UNDER TEST:
    - INACTIVE/ACTIVE state machine (60s timeout)
    - Tree Selector (context-based LP/HP selection with epsilon-greedy)
    - LP Tree: MutualGaze → KnownPerson → GreetedToday → ActionWait
    - HP Tree: KnownPerson → GreetedToday → ActionWait
    - Q-learning with reward = α*valence + β*arousal

PORT MAPPINGS (Test Harness → Module):
    /testHarness/context:o         → /embodiedBehaviour/context:i  
    /testHarness/valence_arousal:o → /embodiedBehaviour/valence_arousal:i
    /testHarness/face_id:o         → /embodiedBehaviour/face_id:i
    /testHarness/actions:o         → /embodiedBehaviour/actions:i
    
NOTE: Vision port removed - face detection via faceID only

RPC SERVICES (Mock Servers):
    /interactionInterface          - Receives "exe <ao_cmd>" 
    /acapelaSpeak/speech:i         - Receives speech text

SCENARIOS:
    1. LP Path: New person greeting (mutual gaze + not greeted today)
    2. LP Path: Action response (greeted today + action)
    3. HP Path: Unknown person wave (no known faces)
    4. HP Path: New person greeting (known face + not greeted today)
    5. State transitions (ACTIVE ↔ INACTIVE)
    6. Learning validation (Q-table updates, epsilon decay)
    7. Multi-person target selection (biggest box)
"""

import yarp
import time
import random
import sys
import signal
import threading
import argparse
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Test Result Tracking
# =============================================================================

class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    TIMEOUT = "TIMEOUT"


@dataclass
class ExpectedOutcome:
    """Define expected outcome for a test scenario."""
    speech_contains: List[str] = field(default_factory=list)  # Expected strings in speech
    actions: List[str] = field(default_factory=list)  # Expected action commands
    min_actions: int = 0  # Minimum number of actions expected
    max_wait_time: float = 20.0  # Maximum time to wait for results
    q_should_update: bool = False  # Whether Q-table should change
    description: str = ""


@dataclass
class TestResult:
    """Result of a test scenario."""
    scenario_name: str
    status: TestStatus
    expected: ExpectedOutcome
    actual_speech: List[str] = field(default_factory=list)
    actual_actions: List[str] = field(default_factory=list)
    q_changed: bool = False
    elapsed_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, msg: str):
        self.errors.append(msg)
        if self.status == TestStatus.PASS:
            self.status = TestStatus.FAIL
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        if self.status == TestStatus.PASS:
            self.status = TestStatus.PARTIAL
    
    def print_summary(self):
        """Print formatted test result."""
        status_symbols = {
            TestStatus.PASS: "[PASS]",
            TestStatus.FAIL: "[FAIL]",
            TestStatus.PARTIAL: "[WARN]",
            TestStatus.TIMEOUT: "[TIME]"
        }
        
        print(f"\n{'='*70}")
        print(f"{status_symbols[self.status]} {self.scenario_name}")
        print(f"{'='*70}")
        print(f"Duration: {self.elapsed_time:.1f}s")
        
        if self.expected.description:
            print(f"Test: {self.expected.description}")
        
        # Expected vs Actual
        print(f"\nExpected:")
        if self.expected.speech_contains:
            print(f"  Speech: {self.expected.speech_contains}")
        if self.expected.actions:
            print(f"  Actions: {self.expected.actions}")
        if self.expected.min_actions > 0:
            print(f"  Min Actions: {self.expected.min_actions}")
        
        print(f"\nActual:")
        if self.actual_speech:
            print(f"  Speech: {self.actual_speech}")
        else:
            print(f"  Speech: (none)")
        if self.actual_actions:
            print(f"  Actions: {self.actual_actions}")
        else:
            print(f"  Actions: (none)")
        
        if self.expected.q_should_update:
            print(f"  Q-table changed: {'YES' if self.q_changed else 'NO'}")
        
        # Errors and warnings
        if self.errors:
            print(f"\nErrors:")
            for err in self.errors:
                print(f"  - {err}")
        
        if self.warnings:
            print(f"\nWarnings:")
            for warn in self.warnings:
                print(f"  - {warn}")
        
        print(f"{'='*70}")


# =============================================================================
# Mock RPC Servers
# =============================================================================

class InteractionInterfaceServer(threading.Thread):
    """
    Mock RPC server for /interactionInterface.
    Receives: "exe <ao_command>" and responds "ack"
    Simulates reactive human behavior by adjusting valence/arousal.
    """
    def __init__(self, world_state_manager, port_name: str = "/interactionInterface"):
        super().__init__(daemon=True)
        self.port_name = port_name
        self.port = yarp.RpcServer()
        self._running = False
        self.received_commands = []
        self._lock = threading.Lock()
        self.world_state = world_state_manager
        self.action_count = 0
        self.last_action_time = 0
        
        # Reaction configs: (delay_sec, valence_delta, arousal_delta, description)
        self.reactions = {
            'ao_start': (0.0, 0.0, 0.0, "▶️  System activated"),
            'ao_stop': (0.0, 0.0, 0.0, "⏹️  System deactivated"),
            'ao_wave': (0.3, +0.10, +0.05, "👋 Wave response - engagement up"),
            'ao_greet': (0.3, +0.15, +0.08, "😊 Greeting response - positive"),
            'ao_yawn_phone': (0.3, -0.05, -0.02, "📱 Phone yawn - slight negative"),
            'ao_curious_lean_in': (0.3, +0.12, +0.06, "🤔 Curious lean"),
            'ao_look_around': (0.3, -0.02, +0.01, "👀 Looking around"),
            'ao_coffee_break': (0.3, +0.08, -0.02, "☕ Coffee break"),
        }
    
    def open(self) -> bool:
        if not self.port.open(self.port_name):
            print(f"[InteractionInterface] ❌ Failed to open {self.port_name}")
            return False
        print(f"[InteractionInterface] ✅ Listening on {self.port_name}")
        return True
    
    def run(self):
        """Listen for commands and simulate reactive behavior."""
        self._running = True
        
        while self._running:
            try:
                bottle_in = yarp.Bottle()
                bottle_out = yarp.Bottle()
                
                if self.port.read(bottle_in, True):
                    cmd_str = bottle_in.toString()
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    # Parse command
                    action = None
                    if bottle_in.size() >= 2 and bottle_in.get(0).asString() == "exe":
                        action = bottle_in.get(1).asString()
                    
                    # Log
                    with self._lock:
                        self.action_count += 1
                        self.last_action_time = time.time()
                        self.received_commands.append({
                            'time': timestamp,
                            'raw': cmd_str,
                            'action': action,
                            'count': self.action_count
                        })
                    
                    print(f"\n{'='*70}")
                    print(f"[ACTION #{self.action_count}] {action or cmd_str} @ {timestamp}")
                    print(f"{'='*70}")
                    
                    # Reply with ack
                    bottle_out.addString("ack")
                    self.port.reply(bottle_out)
                    
                    # Trigger reactive behavior
                    if action and action in self.reactions:
                        threading.Thread(
                            target=self._simulate_reaction,
                            args=(action, self.reactions[action]),
                            daemon=True
                        ).start()
                        
            except Exception as e:
                if self._running:
                    print(f"[InteractionInterface] Error: {e}")
        
        print("[InteractionInterface] Stopped")
    
    def _simulate_reaction(self, action: str, config: Tuple[float, float, float, str]):
        """Simulate human reaction to robot action."""
        delay, v_delta, a_delta, description = config
        
        if delay > 0:
            time.sleep(delay)
        
        if v_delta != 0 or a_delta != 0:
            state = self.world_state.get_state()
            new_v = max(-1.0, min(1.0, state['valence'] + v_delta))
            new_a = max(0.0, min(1.0, state['arousal'] + a_delta))
            
            self.world_state.update_state(valence=new_v, arousal=new_a)
            
            print(f"[Reaction] {description}")
            print(f"           Valence: {state['valence']:.2f} → {new_v:.2f} (Δ{v_delta:+.2f})")
            print(f"           Arousal: {state['arousal']:.2f} → {new_a:.2f} (Δ{a_delta:+.2f})")
    
    def stop(self):
        self._running = False
        self.port.interrupt()
        self.port.close()
    
    def get_commands(self) -> List[Dict]:
        with self._lock:
            return list(self.received_commands)
    
    def clear_commands(self):
        with self._lock:
            self.received_commands.clear()
    
    def get_action_count(self) -> int:
        with self._lock:
            return self.action_count
    
    def get_last_action(self) -> Optional[str]:
        with self._lock:
            if self.received_commands:
                return self.received_commands[-1].get('action')
            return None


class SpeechServer(threading.Thread):
    """
    Mock server for /acapelaSpeak/speech:i.
    Receives speech text from module.
    """
    def __init__(self, port_name: str = "/acapelaSpeak/speech:i"):
        super().__init__(daemon=True)
        self.port_name = port_name
        self.port = yarp.BufferedPortBottle()
        self._running = False
        self.received_speech = []
        self._lock = threading.Lock()
    
    def open(self) -> bool:
        if not self.port.open(self.port_name):
            print(f"[SpeechServer] ❌ Failed to open {self.port_name}")
            return False
        print(f"[SpeechServer] ✅ Listening on {self.port_name}")
        return True
    
    def run(self):
        self._running = True
        while self._running:
            try:
                bottle = self.port.read(False)
                if bottle is not None:
                    text = bottle.toString()
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    with self._lock:
                        self.received_speech.append({
                            'time': timestamp,
                            'text': text
                        })
                    
                    print(f"[Speech] 💬 [{timestamp}] {text}")
                else:
                    time.sleep(0.01)
            except Exception as e:
                if self._running:
                    print(f"[SpeechServer] Error: {e}")
        
        print("[SpeechServer] Stopped")
    
    def stop(self):
        self._running = False
        self.port.interrupt()
        self.port.close()
    
    def get_speech(self) -> List[Dict]:
        with self._lock:
            return list(self.received_speech)
    
    def clear_speech(self):
        with self._lock:
            self.received_speech.clear()
    
    def get_last_speech(self) -> Optional[str]:
        with self._lock:
            if self.received_speech:
                return self.received_speech[-1].get('text')
            return None


# =============================================================================
# World State Manager (Thread-Safe)
# =============================================================================

class WorldStateManager:
    """Thread-safe manager for simulated world state."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            # Face detection (via faceID only - no vision port)
            'faces_count': 0,  # Total faces (known + unknown)
            
            # Context (-1=uncertain, 0=calm, 1=lively)
            'context_label': -1,
            'episode_id': 1,
            'chunk_id': 1,
            
            # Known faces: [{name, confidence, box}]
            'known_faces': [],
            
            # Valence/Arousal (per face, but we use global for simplicity)
            'valence': 0.5,
            'arousal': 0.3,
            
            # Detected actions: {person_id: (action, prob)}
            'detected_actions': {},
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get copy of current state."""
        with self._lock:
            state = dict(self._state)
            state['known_faces'] = list(self._state['known_faces'])
            state['detected_actions'] = dict(self._state['detected_actions'])
            return state
    
    def update_state(self, **kwargs):
        """Update specific state fields."""
        with self._lock:
            for key, value in kwargs.items():
                if key in self._state:
                    self._state[key] = value
    
    def add_face(self, name: str, confidence: float, box: List[float]):
        """Add a known face."""
        with self._lock:
            self._state['known_faces'].append({
                'name': name,
                'confidence': confidence,
                'box': box  # [x1, y1, x2, y2]
            })
            self._state['faces_count'] = len(self._state['known_faces'])
    
    def set_faces_count(self, count: int):
        """Set faces count (for unknown faces)."""
        with self._lock:
            self._state['faces_count'] = count
    
    def clear_faces(self):
        """Remove all faces."""
        with self._lock:
            self._state['known_faces'] = []
            self._state['faces_count'] = 0
    
    def set_action(self, person_id: str, action: str, prob: float = 0.9):
        """Set detected action for a person."""
        with self._lock:
            self._state['detected_actions'][person_id] = (action, prob)
    
    def clear_actions(self):
        """Clear all detected actions."""
        with self._lock:
            self._state['detected_actions'] = {}


# =============================================================================
# Data Pump (20Hz Continuous Publishing)
# =============================================================================

class DataPump(threading.Thread):
    """
    Continuously publishes perception data at 20Hz.
    Maintains stable data stream to module input ports.
    """
    
    def __init__(self, world_state: WorldStateManager):
        super().__init__(daemon=True)
        self.world_state = world_state
        self._running = False
        
        # Output ports (NO vision port - face detection via faceID only)
        self.context_port = yarp.BufferedPortBottle()
        self.va_port = yarp.BufferedPortBottle()
        self.face_id_port = yarp.BufferedPortBottle()
        self.action_port = yarp.BufferedPortBottle()
        
        # Port names
        self.port_names = {
            'context': "/testHarness/context:o",
            'va': "/testHarness/valence_arousal:o",
            'face_id': "/testHarness/face_id:o",
            'actions': "/testHarness/actions:o",
        }
        
        # Target ports
        self.targets = {
            'context': "/embodiedBehaviour/context:i",
            'va': "/embodiedBehaviour/valence_arousal:i",
            'face_id': "/embodiedBehaviour/face_id:i",
            'actions': "/embodiedBehaviour/actions:i",
        }
        
        self.publish_count = 0
    
    def open_ports(self) -> bool:
        """Open all output ports."""
        ports = [
            (self.context_port, self.port_names['context']),
            (self.va_port, self.port_names['va']),
            (self.face_id_port, self.port_names['face_id']),
            (self.action_port, self.port_names['actions']),
        ]
        
        for port, name in ports:
            if not port.open(name):
                print(f"[DataPump] ❌ Failed to open {name}")
                return False
        
        print(f"[DataPump] ✅ All ports opened")
        return True
    
    def connect_ports(self, timeout: float = 15.0) -> bool:
        """Connect to module input ports."""
        connections = [
            (self.port_names['context'], self.targets['context']),
            (self.port_names['va'], self.targets['va']),
            (self.port_names['face_id'], self.targets['face_id']),
            (self.port_names['actions'], self.targets['actions']),
        ]
        
        print("\n[DataPump] Waiting for module ports...")
        start_time = time.time()
        
        for src, dst in connections:
            while time.time() - start_time < timeout:
                if yarp.Network.exists(dst):
                    break
                time.sleep(0.5)
            else:
                print(f"[DataPump] ⏱️  Timeout waiting for {dst}")
                return False
            
            if not yarp.Network.connect(src, dst):
                print(f"[DataPump] ❌ Failed: {src} → {dst}")
                return False
            print(f"[DataPump] ✅ {src} → {dst}")
        
        return True
    
    def close_ports(self):
        """Close all ports."""
        for port in [self.context_port, self.va_port,
                     self.face_id_port, self.action_port]:
            port.interrupt()
            port.close()
    
    def run(self):
        """20Hz publishing loop."""
        print("[DataPump] ▶️  Starting 20Hz loop...")
        self._running = True
        
        while self._running:
            try:
                self.publish_count += 1
                state = self.world_state.get_state()
                
                # Add small noise to VA for realism
                v_noise = random.uniform(-0.02, 0.02)
                a_noise = random.uniform(-0.02, 0.02)
                v = max(-1.0, min(1.0, state['valence'] + v_noise))
                a = max(0.0, min(1.0, state['arousal'] + a_noise))
                
                # Publish all (NO vision - face detection via faceID only)
                if self.publish_count % 10 == 0:  # Context at 2Hz
                    self._publish_context(state)
                self._publish_face_id(state)
                self._publish_va(state, v, a)
                self._publish_actions(state)
                
                # Status log every 10s
                if self.publish_count % 200 == 0:
                    print(f"[DataPump] 📊 {self.publish_count} msgs | "
                          f"Faces={state['faces_count']} "
                          f"Ctx={state['context_label']} V={state['valence']:.2f} A={state['arousal']:.2f}")
                
                time.sleep(0.05)  # 20Hz
                
            except Exception as e:
                if self._running:
                    print(f"[DataPump] Error: {e}")
        
        print("[DataPump] ⏹️  Stopped")
    
    def _publish_context(self, state: Dict):
        """
        Publish context.
        Format: <episode_id> <chunk_id> <label>
        """
        bottle = self.context_port.prepare()
        bottle.clear()
        
        bottle.addInt32(state['episode_id'])
        bottle.addInt32(state['chunk_id'])
        bottle.addInt32(state['context_label'])
        
        self.context_port.write()
    
    def _publish_face_id(self, state: Dict):
        """
        Publish face IDs.
        Format: ((class <name>) (score <conf>) (box (<x1> <y1> <x2> <y2>)))
        """
        bottle = self.face_id_port.prepare()
        bottle.clear()
        
        for face in state['known_faces']:
            face_list = bottle.addList()
            
            # class
            c = face_list.addList()
            c.addString("class")
            c.addString(face['name'])
            
            # score
            s = face_list.addList()
            s.addString("score")
            s.addFloat64(face['confidence'])
            
            # box
            b = face_list.addList()
            b.addString("box")
            box_list = b.addList()
            for coord in face['box']:
                box_list.addFloat64(coord)
        
        self.face_id_port.write()
    
    def _publish_va(self, state: Dict, valence: float, arousal: float):
        """
        Publish valence/arousal.
        Format: ((id <idx>)(class <name>)(score <conf>)(valence <v>)(arousal <a>)(box <x> <y> <w> <h>)(status ok))
        """
        bottle = self.va_port.prepare()
        bottle.clear()
        
        for idx, face in enumerate(state['known_faces']):
            va_list = bottle.addList()
            
            # id
            i = va_list.addList()
            i.addString("id")
            i.addInt32(idx)
            
            # class
            c = va_list.addList()
            c.addString("class")
            c.addString(face['name'])
            
            # score
            s = va_list.addList()
            s.addString("score")
            s.addFloat64(face['confidence'])
            
            # valence
            v = va_list.addList()
            v.addString("valence")
            v.addFloat64(valence)
            
            # arousal
            a = va_list.addList()
            a.addString("arousal")
            a.addFloat64(arousal)
            
            # box (x, y, w, h format)
            b = va_list.addList()
            b.addString("box")
            box = face['box']
            b.addFloat64(box[0])  # x
            b.addFloat64(box[1])  # y
            b.addFloat64(box[2] - box[0])  # w
            b.addFloat64(box[3] - box[1])  # h
            
            # status
            st = va_list.addList()
            st.addString("status")
            st.addString("ok")
        
        self.va_port.write()
    
    def _publish_actions(self, state: Dict):
        """
        Publish detected actions.
        Format: (stamp <ts>) (people (((class <id>) (action <label>) (prob <conf>)) ...))
        """
        bottle = self.action_port.prepare()
        bottle.clear()
        
        # stamp
        stamp = bottle.addList()
        stamp.addString("stamp")
        stamp.addFloat64(time.time())
        
        # people
        people = bottle.addList()
        people.addString("people")
        people_list = people.addList()
        
        for person_id, (action, prob) in state['detected_actions'].items():
            person = people_list.addList()
            
            # class (person_id)
            c = person.addList()
            c.addString("class")
            c.addString(str(person_id))
            
            # action
            a = person.addList()
            a.addString("action")
            a.addString(action)
            
            # prob
            p = person.addList()
            p.addString("prob")
            p.addFloat64(prob)
        
        self.action_port.write()
    
    def stop(self):
        self._running = False


# =============================================================================
# Test Harness Main Controller
# =============================================================================

class UnifiedTestHarness:
    """
    Main test harness controller.
    Coordinates world state, data pump, RPC servers, and test scenarios.
    """
    
    # Allowed actions for testing
    ALLOWED_ACTIONS = [
        "answer phone",
        "carry/hold (an object)",
        "drink",
        "eat",
        "text on/look at a cellphone",
        "hand wave",
    ]
    
    # Expected responses for actions
    ACTION_RESPONSES = {
        "answer phone": ("speech", "Salutalo da parte mia"),
        "carry/hold (an object)": ("speech", "Tieni forte"),
        "drink": ("speech", "Salute"),
        "eat": ("speech", "Buon appetito"),
        "text on/look at a cellphone": ("ao", "ao_yawn_phone"),
        "hand wave": ("ao", "ao_wave"),
    }
    
    def __init__(self):
        """Initialize harness components."""
        yarp.Network.init()
        if not yarp.Network.checkNetwork():
            print("YARP network not available!")
            sys.exit(1)
        
        print("\n" + "="*80)
        print("  UNIFIED TEST HARNESS FOR EMBODIED BEHAVIOUR")
        print("="*80)
        print("  Tests: State machine, BT paths, Q-learning, action responses")
        print("  Press Ctrl+C anytime to stop")
        print("="*80 + "\n")
        
        # Core components
        self.world_state = WorldStateManager()
        self.interaction_server = InteractionInterfaceServer(self.world_state)
        self.speech_server = SpeechServer()
        self.data_pump = DataPump(self.world_state)
        
        # Control
        self._running = False
        
        # Q-table tracking
        self.q_file = "./q_table.json"
        self.initial_q = None
        
        # Test results tracking
        self.test_results: List[TestResult] = []
        
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print("\n\n[Harness] 🛑 Interrupted")
        self._running = False
    
    def setup(self, auto_connect: bool = True, wait_for_module: bool = True) -> bool:
        """Set up all components."""
        print("[Setup] 🔧 Initializing...")
        
        # Open ports
        if not self.data_pump.open_ports():
            return False
        if not self.interaction_server.open():
            return False
        if not self.speech_server.open():
            return False
        
        # Start servers
        print("\n[Setup] 🚀 Starting RPC servers...")
        self.interaction_server.start()
        self.speech_server.start()
        time.sleep(0.5)
        
        # Connect
        if auto_connect:
            if wait_for_module:
                print("\n[Setup] ⏳ Waiting for embodied_behaviour module...")
                print("         (Start it in another terminal if not running)")
            
            if not self.data_pump.connect_ports(timeout=30.0 if wait_for_module else 5.0):
                if wait_for_module:
                    print("\n❌ Failed to connect to module!")
                    return False
        
        # Start data pump
        print("\n[Setup] 🚀 Starting 20Hz data pump...")
        self.data_pump.start()
        time.sleep(0.5)
        
        # Snapshot initial Q-table
        self._snapshot_q_table()
        
        self._running = True
        
        print("\n" + "="*80)
        print("  ✅ Setup Complete")
        print("="*80 + "\n")
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        print("\n[Cleanup] Shutting down...")
        self._running = False
        
        self.data_pump.stop()
        self.interaction_server.stop()
        self.speech_server.stop()
        
        time.sleep(0.5)
        self.data_pump.close_ports()
        
        print("[Cleanup] Done")
    
    def _wait_for_action(self, timeout: float = 20.0, min_actions: int = 1, ignore_state_commands: bool = True) -> bool:
        """Wait for robot to perform action(s), return True if received.
        
        Args:
            ignore_state_commands: If True, don't count ao_start/ao_stop as actions to wait for
        """
        start_time = time.time()
        
        # Get initial action count, excluding state commands if requested
        if ignore_state_commands:
            initial_commands = [c for c in self.interaction_server.get_commands() if c['action'] not in ['ao_start', 'ao_stop']]
            initial_count = len(initial_commands)
        else:
            initial_count = self.interaction_server.get_action_count()
        
        print(f"[Wait] Expecting {min_actions} action(s), max wait {timeout:.0f}s...")
        
        while time.time() - start_time < timeout:
            if ignore_state_commands:
                # Count only non-state-machine commands
                commands = [c for c in self.interaction_server.get_commands() if c['action'] not in ['ao_start', 'ao_stop']]
                current_count = len(commands)
            else:
                current_count = self.interaction_server.get_action_count()
            
            if current_count >= initial_count + min_actions:
                elapsed = time.time() - start_time
                print(f"[Wait] Action(s) received after {elapsed:.1f}s")
                return True
            time.sleep(0.5)
        
        print(f"[Wait] Timeout waiting for actions")
        return False
    
    def _wait_for_speech(self, timeout: float = 20.0, contains: str = None) -> bool:
        """Wait for speech output, optionally checking content."""
        start_time = time.time()
        initial_count = len(self.speech_server.get_speech())
        
        print(f"[Wait] Expecting speech{f' containing "{contains}"' if contains else ''}, max wait {timeout:.0f}s...")
        
        while time.time() - start_time < timeout:
            speeches = self.speech_server.get_speech()
            if len(speeches) > initial_count:
                if contains:
                    for speech in speeches[initial_count:]:
                        if contains.lower() in speech['text'].lower():
                            elapsed = time.time() - start_time
                            print(f"[Wait] Speech received after {elapsed:.1f}s")
                            return True
                else:
                    elapsed = time.time() - start_time
                    print(f"[Wait] Speech received after {elapsed:.1f}s")
                    return True
            time.sleep(0.5)
        
        print(f"[Wait] Timeout waiting for speech")
        return False
    
    def _verify_scenario(self, scenario_name: str, expected: ExpectedOutcome) -> TestResult:
        """Verify test scenario results against expectations."""
        result = TestResult(
            scenario_name=scenario_name,
            status=TestStatus.PASS,
            expected=expected,
            elapsed_time=expected.max_wait_time
        )
        
        # Collect actual results
        result.actual_actions = [cmd['action'] for cmd in self.interaction_server.get_commands()]
        result.actual_speech = [sp['text'] for sp in self.speech_server.get_speech()]
        
        # Verify actions
        if expected.actions:
            for exp_action in expected.actions:
                if exp_action not in result.actual_actions:
                    result.add_error(f"Expected action '{exp_action}' not received")
        
        if expected.min_actions > 0:
            action_count = len([a for a in result.actual_actions if a and a.startswith('ao_')])
            if action_count < expected.min_actions:
                result.add_error(f"Expected >= {expected.min_actions} actions, got {action_count}")
        
        # Verify speech
        if expected.speech_contains:
            all_speech = ' '.join(result.actual_speech).lower()
            for exp_text in expected.speech_contains:
                if exp_text.lower() not in all_speech:
                    result.add_error(f"Expected speech containing '{exp_text}' not found")
        
        # Verify Q-table updates
        if expected.q_should_update:
            current_q = self._get_q_table()
            if self.initial_q and current_q:
                for ctx in ['calm', 'lively']:
                    for br in ['LP', 'HP']:
                        old_val = self.initial_q.get(ctx, {}).get(br, 0)
                        new_val = current_q.get(ctx, {}).get(br, 0)
                        if abs(new_val - old_val) > 0.0001:
                            result.q_changed = True
                            break
                
                if not result.q_changed:
                    result.add_warning("Q-table did not update (action may not have completed)")
            else:
                result.add_warning("Could not verify Q-table changes")
        
        return result
    
    def _snapshot_q_table(self):
        """Snapshot Q-table for comparison."""
        if os.path.exists(self.q_file):
            try:
                with open(self.q_file) as f:
                    self.initial_q = json.load(f)
            except:
                self.initial_q = None
    
    def _get_q_table(self) -> Optional[Dict]:
        """Get current Q-table."""
        if os.path.exists(self.q_file):
            try:
                with open(self.q_file) as f:
                    return json.load(f)
            except:
                return None
        return None
    
    # =========================================================================
    # Interactive Mode
    # =========================================================================
    
    def run_interactive(self):
        """Run interactive CLI."""
        print("="*80)
        print("  INTERACTIVE MODE")
        print("="*80)
        print("Type 'help' for commands\n")
        
        self._print_status()
        
        while self._running:
            try:
                cmd = input("\n> ").strip()
                if cmd:
                    self._handle_command(cmd)
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nExiting interactive mode...")
    
    def _handle_command(self, cmd: str):
        """Handle interactive command."""
        parts = cmd.split()
        if not parts:
            return
        
        action = parts[0].lower()
        args = parts[1:]
        
        # Face management
        if action == "person" and len(args) >= 2:
            name = args[0]
            conf = float(args[1])
            x1, y1 = random.randint(100, 200), random.randint(100, 200)
            size = random.randint(150, 300)
            self.world_state.add_face(name, conf, [x1, y1, x1+size, y1+size])
            print(f"✅ Added {name} (conf={conf:.2f}, box_size={size})")
        
        elif action == "faces" and len(args) >= 1:
            count = int(args[0])
            self.world_state.set_faces_count(count)
            print(f"✅ Set faces count to {count} (unknown faces)")
        
        elif action == "clear":
            self.world_state.clear_faces()
            self.world_state.clear_actions()
            print("✅ Cleared all faces and actions")
        
        # Context
        elif action == "context" and len(args) >= 1:
            label = int(args[0])
            if label in [-1, 0, 1]:
                names = {-1: "uncertain", 0: "calm", 1: "lively"}
                self.world_state.update_state(context_label=label)
                print(f"✅ Context = {label} ({names[label]})")
        
        # VA
        elif action == "va" and len(args) >= 2:
            v, a = float(args[0]), float(args[1])
            self.world_state.update_state(valence=v, arousal=a)
            print(f"✅ V={v:.2f}, A={a:.2f}")
        
        # Actions
        elif action == "action" and len(args) >= 2:
            pid = args[0]
            act = " ".join(args[1:])
            self.world_state.set_action(pid, act, 0.9)
            print(f"✅ Person {pid} action: {act}")
        
        elif action == "clearactions":
            self.world_state.clear_actions()
            print("✅ Cleared actions")
        
        # Scenarios
        elif action.startswith("scenario") or action.startswith("test"):
            scenario_map = {
                "scenario1": self._scenario_lp_new_person,
                "test1": self._scenario_lp_new_person,
                "scenario2": self._scenario_lp_action_response,
                "test2": self._scenario_lp_action_response,
                "scenario3": self._scenario_hp_no_known,
                "test3": self._scenario_hp_no_known,
                "scenario4": self._scenario_hp_new_person,
                "test4": self._scenario_hp_new_person,
                "scenario5": self._scenario_state_transitions,
                "test5": self._scenario_state_transitions,
                "scenario6": self._scenario_learning_test,
                "test6": self._scenario_learning_test,
                "scenario7": self._scenario_multi_person,
                "test7": self._scenario_multi_person,
            }
            if action in scenario_map:
                scenario_map[action]()
            else:
                print(f"Unknown scenario. Available: scenario1-7 or test1-7")
        
        # Monitoring
        elif action == "status":
            self._print_status()
        
        elif action == "log":
            self._print_logs()
        
        elif action == "q":
            self._print_q_table()
        
        elif action == "clearlog":
            self.interaction_server.clear_commands()
            self.speech_server.clear_speech()
            print("✅ Cleared logs")
        
        # Help
        elif action == "help":
            self._print_help()
        
        elif action in ["quit", "exit"]:
            self._running = False
        
        else:
            print(f"Unknown: {action}. Type 'help'")
    
    def _print_help(self):
        print("""
══════════════════════════════════════════════════════════════════════
  COMMANDS
══════════════════════════════════════════════════════════════════════

👤 FACES:
  person <name> <conf>     Add known person (e.g., person Mario 0.95)
  faces <n>                Set face count (unknown faces)
  clear                    Remove all faces

🌍 ENVIRONMENT:
  context <-1|0|1>         Set context (uncertain/calm/lively)
  va <v> <a>               Set valence/arousal

🎬 ACTIONS:
  action <id> <action>     Trigger action (e.g., action 0 hand wave)
  clearactions             Clear detected actions

🧪 SCENARIOS:
  scenario1 / test1        LP: New person greeting
  scenario2 / test2        LP: Action response  
  scenario3 / test3        HP: No known person (wave only)
  scenario4 / test4        HP: New person greeting
  scenario5 / test5        State transitions test
  scenario6 / test6        Learning validation
  scenario7 / test7        Multi-person target selection

📊 MONITORING:
  status                   Show world state
  log                      Show received commands/speech
  q                        Show Q-table
  clearlog                 Clear logs

❓ OTHER:
  help                     This help
  quit                     Exit
══════════════════════════════════════════════════════════════════════
""")
    
    def _print_status(self):
        state = self.world_state.get_state()
        ctx_names = {-1: "uncertain", 0: "calm", 1: "lively"}
        
        print(f"""
──────────────────────────────────────────────────────────────────────
  CURRENT STATE
──────────────────────────────────────────────────────────────────────
  👥 Faces:        {state['faces_count']}
  🌍 Context:      {state['context_label']} ({ctx_names.get(state['context_label'], '?')})
  😊 Valence:      {state['valence']:.2f}
  ⚡ Arousal:      {state['arousal']:.2f}
  🧑 Known:        {[f['name'] for f in state['known_faces']] or '(none)'}
  🎬 Actions:      {state['detected_actions'] or '(none)'}
  📈 Robot Acts:   {self.interaction_server.get_action_count()}
──────────────────────────────────────────────────────────────────────""")
    
    def _print_logs(self):
        print("\n" + "─"*70)
        print("  INTERACTION COMMANDS")
        print("─"*70)
        for cmd in self.interaction_server.get_commands()[-10:]:
            print(f"  [{cmd['time']}] {cmd['action'] or cmd['raw']}")
        if not self.interaction_server.get_commands():
            print("  (none)")
        
        print("\n" + "─"*70)
        print("  SPEECH OUTPUT")
        print("─"*70)
        for sp in self.speech_server.get_speech()[-10:]:
            print(f"  [{sp['time']}] {sp['text']}")
        if not self.speech_server.get_speech():
            print("  (none)")
        print("─"*70)
    
    def _print_q_table(self):
        q = self._get_q_table()
        if q:
            print(f"""
──────────────────────────────────────────────────────────────────────
  Q-TABLE
──────────────────────────────────────────────────────────────────────
  calm:   LP={q.get('calm',{}).get('LP',0):.4f}  HP={q.get('calm',{}).get('HP',0):.4f}
  lively: LP={q.get('lively',{}).get('LP',0):.4f}  HP={q.get('lively',{}).get('HP',0):.4f}
  epsilon: {q.get('epsilon', 0.8):.2f}
──────────────────────────────────────────────────────────────────────""")
            
            if self.initial_q:
                print("  Changes from start:")
                for ctx in ['calm', 'lively']:
                    for br in ['LP', 'HP']:
                        old = self.initial_q.get(ctx, {}).get(br, 0)
                        new = q.get(ctx, {}).get(br, 0)
                        if abs(new - old) > 0.0001:
                            print(f"    {ctx}/{br}: {old:.4f} → {new:.4f} (Δ{new-old:+.4f})")
                eps_old = self.initial_q.get('epsilon', 0.8)
                eps_new = q.get('epsilon', 0.8)
                if abs(eps_new - eps_old) > 0.0001:
                    print(f"    epsilon: {eps_old:.2f} → {eps_new:.2f}")
        else:
            print("  (Q-table not found)")
    
    # =========================================================================
    # Test Scenarios
    # =========================================================================
    
    def _scenario_lp_new_person(self):
        """
        Scenario 1: LP Path - New Person Greeting
        Requires: close face (large box), known person, NOT greeted today
        Expected: Greet + wave, learning update
        """
        print("""
══════════════════════════════════════════════════════════════════════
  SCENARIO 1: LP - New Person Greeting
══════════════════════════════════════════════════════════════════════
  Path: CloseFace OK → KnownPerson OK → GreetedToday NO → Greet+Wave
  Expected: "Ciao <name>" speech + ao_wave command
  Note: LP requires large face box (close proximity), not mutual gaze
══════════════════════════════════════════════════════════════════════
""")
        # Setup
        self.world_state.clear_faces()
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        # Define expected outcome
        expected = ExpectedOutcome(
            speech_contains=["ciao"],
            actions=["ao_wave"],
            min_actions=1,
            max_wait_time=18.0,
            q_should_update=True,
            description="LP greeting path with close face (large box)"
        )
        
        start_time = time.time()
        
        # Calm context (tends to select LP)
        print("[Setup] Setting calm context...")
        self.world_state.update_state(context_label=0)
        time.sleep(2)  # Allow module to process context
        
        # Add person with LARGE box (close proximity for LP)
        name = f"TestPerson_{random.randint(100,999)}"
        print(f"[Setup] Adding '{name}' with LARGE bounding box (close proximity)...")
        # Large box = 300x400 pixels (area = 120,000 > threshold of 15,000)
        self.world_state.add_face(name, 0.95, [100, 100, 400, 500])
        self.world_state.update_state(valence=0.6, arousal=0.4)
        
        # Wait for response with verification
        if not self._wait_for_action(timeout=expected.max_wait_time, min_actions=1):
            print("[Result] No action received within timeout")
        
        # Give extra time for learning update
        time.sleep(2)
        
        # Verify results
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("LP New Person Greeting", expected)
        result.print_summary()
        self.test_results.append(result)
    
    def _scenario_lp_action_response(self):
        """
        Scenario 2: LP Path - Action Response
        Requires: close face (large box), known person, greeted today, action detected
        Expected: Appropriate response to action
        """
        print("""
══════════════════════════════════════════════════════════════════════
  SCENARIO 2: LP - Action Response
══════════════════════════════════════════════════════════════════════
  Path: CloseFace OK → KnownPerson OK → GreetedToday OK → ActionDetected OK
  Expected: Response based on action type
  Note: LP requires large face box (close proximity)
══════════════════════════════════════════════════════════════════════
""")
        print("  NOTE: Run scenario1 first to mark person as greeted today\n")
        
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        # Select random action
        action = random.choice(["hand wave", "drink", "eat"])
        expected_response = self.ACTION_RESPONSES.get(action)
        
        expected = ExpectedOutcome(
            min_actions=1,
            max_wait_time=15.0,
            description=f"LP action response to '{action}'"
        )
        
        if expected_response:
            resp_type, resp_value = expected_response
            if resp_type == "speech":
                expected.speech_contains = [resp_value]
            elif resp_type == "ao":
                expected.actions = [resp_value]
        
        start_time = time.time()
        
        # Ensure calm context
        self.world_state.update_state(context_label=0)
        
        # Ensure face with LARGE box (close proximity)
        state = self.world_state.get_state()
        if not state['known_faces']:
            print("[Setup] Adding known person with LARGE box...")
            # Large box = 300x400 pixels (area = 120,000 > threshold of 15,000)
            self.world_state.add_face("Mario", 0.95, [100, 100, 400, 500])
        
        print("[Setup] Waiting 3s for module to see state...")
        time.sleep(3)
        
        # Trigger action
        print(f"[Setup] Triggering action: '{action}'")
        self.world_state.set_action("0", action, 0.92)
        
        # Wait for response
        if not self._wait_for_action(timeout=expected.max_wait_time, min_actions=1):
            print("[Result] No response received within timeout")
        
        time.sleep(1)  # Allow final processing
        
        # Verify results
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("LP Action Response", expected)
        if expected_response:
            print(f"\nExpected response for '{action}': {expected_response}")
        result.print_summary()
        self.test_results.append(result)
    
    def _scenario_hp_no_known(self):
        """
        Scenario 3: HP Path - No Known Person
        Requires: no known faces (but faces > 0)
        Expected: Wave only (no greeting)
        """
        print("""
══════════════════════════════════════════════════════════════════════
  SCENARIO 3: HP - No Known Person (Wave Only)
══════════════════════════════════════════════════════════════════════
  Path: KnownPerson NO → Wave Only
  Expected: ao_wave command (no speech)
══════════════════════════════════════════════════════════════════════
""")
        self.world_state.clear_faces()
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        expected = ExpectedOutcome(
            actions=["ao_wave"],
            min_actions=1,
            max_wait_time=18.0,
            q_should_update=True,
            description="HP wave-only path (no known person)"
        )
        
        start_time = time.time()
        
        # Lively context (tends to select HP)
        print("[Setup] Setting lively context...")
        self.world_state.update_state(context_label=1)
        time.sleep(2)
        
        # Set faces but no known (just count)
        print("[Setup] Setting 1 unknown face...")
        self.world_state.set_faces_count(1)
        self.world_state.update_state(valence=0.5, arousal=0.5)
        
        # Wait for response
        if not self._wait_for_action(timeout=expected.max_wait_time, min_actions=1):
            print("[Result] No action received within timeout")
        
        time.sleep(2)
        
        # Verify results
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("HP Wave Only", expected)
        
        # Additional check: should NOT have greeting speech
        if result.actual_speech:
            result.add_warning(f"Unexpected speech received: {result.actual_speech}")
        
        result.print_summary()
        self.test_results.append(result)
    
    def _scenario_hp_new_person(self):
        """
        Scenario 4: HP Path - New Person Greeting
        Requires: known person, NOT greeted today (no mutual gaze needed)
        Expected: Greet + wave
        """
        print("""
══════════════════════════════════════════════════════════════════════
  SCENARIO 4: HP - New Person Greeting
══════════════════════════════════════════════════════════════════════
  Path: KnownPerson OK → GreetedToday NO → Greet+Wave
  Expected: "Ciao <name>" + ao_wave (HP has no close-face requirement)
══════════════════════════════════════════════════════════════════════
""")
        self.world_state.clear_faces()
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        expected = ExpectedOutcome(
            speech_contains=["ciao"],
            actions=["ao_wave"],
            min_actions=1,
            max_wait_time=18.0,
            q_should_update=True,
            description="HP greeting path"
        )
        
        start_time = time.time()
        
        # Lively context
        print("[Setup] Setting lively context...")
        self.world_state.update_state(context_label=1)
        time.sleep(2)
        
        # Add known person (HP doesn't need close face)
        name = f"Luigi_{random.randint(100,999)}"
        print(f"[Setup] Adding '{name}'...")
        self.world_state.add_face(name, 0.92, [150, 100, 400, 420])
        self.world_state.update_state(valence=0.55, arousal=0.45)
        
        # Wait for response
        if not self._wait_for_action(timeout=expected.max_wait_time, min_actions=1):
            print("[Result] No action received within timeout")
        
        time.sleep(2)
        
        # Verify results
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("HP New Person Greeting", expected)
        result.print_summary()
        self.test_results.append(result)
    
    def _scenario_state_transitions(self):
        """
        Scenario 5: Test ACTIVE/INACTIVE transitions
        Comprehensive test of state machine and ao_start/ao_stop
        """
        print("""
══════════════════════════════════════════════════════════════════════
  SCENARIO 5: State Transitions (Complete)
══════════════════════════════════════════════════════════════════════
  Tests: Full state machine with ao_start, ao_stop, and tree selection
  Duration: ~70 seconds (includes 60s INACTIVE timeout)
══════════════════════════════════════════════════════════════════════
""")
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        expected = ExpectedOutcome(
            actions=["ao_start", "ao_stop"],
            min_actions=2,
            max_wait_time=75.0,
            description="Complete state machine test"
        )
        
        start_time = time.time()
        
        # Phase 1: Ensure INACTIVE state
        print("\n[Phase 1] Ensuring INACTIVE state...")
        self.world_state.clear_faces()
        self.world_state.clear_actions()
        time.sleep(3)
        
        # Phase 2: Test INACTIVE → ACTIVE (ao_start)
        print("\n[Phase 2] Testing INACTIVE → ACTIVE transition...")
        print("[Action] Adding face → expect ao_start...")
        self.world_state.add_face("StateTest", 0.9, [100, 100, 300, 300])
        self.world_state.update_state(context_label=0)
        
        if self._wait_for_action(timeout=8.0, min_actions=1):
            cmds = [c['action'] for c in self.interaction_server.get_commands()]
            if 'ao_start' in cmds:
                print("[Result] PASS: ao_start received")
            else:
                print(f"[Result] FAIL: Expected ao_start, got {cmds}")
        else:
            print("[Result] FAIL: No ao_start within 8s")
        
        # Phase 3: Verify tree selection happens in ACTIVE state
        print("\n[Phase 3] Verifying tree selection in ACTIVE state...")
        print("[Action] Waiting 10s for tree selection and greeting...")
        time.sleep(10)
        
        speech_count = len(self.speech_server.get_speech())
        action_count = self.interaction_server.get_action_count()
        
        if speech_count > 0 or action_count > 1:  # More than just ao_start
            print(f"[Result] PASS: Tree selection active (speech={speech_count}, actions={action_count})")
        else:
            print(f"[Result] WARN: No tree selection detected (speech={speech_count}, actions={action_count})")
        
        # Phase 4: Test ACTIVE → INACTIVE (ao_stop after 60s)
        print("\n[Phase 4] Testing ACTIVE → INACTIVE transition...")
        print("[Action] Removing all faces → wait 60s for ao_stop...")
        self.world_state.clear_faces()
        
        # Wait 65 seconds for the 60s timeout + processing
        print("[Wait] This will take ~65 seconds...")
        for remaining in range(65, 0, -5):
            print(f"       {remaining}s remaining...")
            time.sleep(5)
        
        # Check for ao_stop
        print("\n[Check] Looking for ao_stop...")
        cmds = [c['action'] for c in self.interaction_server.get_commands()]
        if 'ao_stop' in cmds:
            print("[Result] PASS: ao_stop received after 60s timeout")
        else:
            print(f"[Result] FAIL: ao_stop not found. All commands: {cmds}")
        
        # Phase 5: Verify tree selection stops in INACTIVE state
        print("\n[Phase 5] Verifying no tree selection in INACTIVE state...")
        print("[Action] Adding face again (should NOT trigger greeting in INACTIVE)...")
        initial_action_count = self.interaction_server.get_action_count()
        self.world_state.add_face("InactiveTest", 0.9, [100, 100, 300, 300])
        self.world_state.update_state(context_label=0)
        time.sleep(8)
        
        new_action_count = self.interaction_server.get_action_count()
        if new_action_count == initial_action_count:
            print("[Result] PASS: No tree execution in INACTIVE state")
        else:
            print(f"[Result] FAIL: Tree executed in INACTIVE state (actions: {new_action_count - initial_action_count})")
        
        # Verify and summarize
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("State Machine Complete Test", expected)
        
        # Additional checks
        if 'ao_start' not in cmds:
            result.add_error("ao_start was not received")
        if 'ao_stop' not in cmds:
            result.add_error("ao_stop was not received after 60s")
        
        result.print_summary()
        self.test_results.append(result)
        
        print("\n[Summary] State machine test complete")
        print(f"          Total duration: {expected.elapsed_time:.1f}s")
        print(f"          Commands received: {len(cmds)}")
    
    def _scenario_learning_test(self):
        """
        Scenario 6: Validate Q-learning updates
        """
        print("""
══════════════════════════════════════════════════════════════════════
  SCENARIO 6: Learning Validation
══════════════════════════════════════════════════════════════════════
  Tests: Q-table updates and epsilon decay after actions
══════════════════════════════════════════════════════════════════════
""")
        # Snapshot Q before
        q_before = self._get_q_table()
        if q_before:
            eps_before = q_before.get('epsilon', 0.8)
            print(f"[Before] Epsilon: {eps_before:.2f}")
            print(f"         Q[calm]:   LP={q_before['calm']['LP']:.4f} HP={q_before['calm']['HP']:.4f}")
            print(f"         Q[lively]: LP={q_before['lively']['LP']:.4f} HP={q_before['lively']['HP']:.4f}")
        
        expected = ExpectedOutcome(
            speech_contains=["ciao"],
            actions=["ao_wave"],
            min_actions=1,
            max_wait_time=18.0,
            q_should_update=True,
            description="Learning update validation"
        )
        
        start_time = time.time()
        
        # Run a quick scenario to trigger learning
        print("\n[Test] Running greeting scenario to trigger learning...")
        self.world_state.clear_faces()
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        self.world_state.update_state(context_label=0)  # calm
        time.sleep(2)
        
        name = f"LearnTest_{random.randint(100,999)}"
        self.world_state.add_face(name, 0.95, [100, 100, 400, 500])
        self.world_state.update_state(valence=0.7, arousal=0.5)
        
        if not self._wait_for_action(timeout=expected.max_wait_time, min_actions=1):
            print("[Result] No action received within timeout")
        
        time.sleep(2)  # Allow Q-table write
        
        # Check Q after
        q_after = self._get_q_table()
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("Learning Validation", expected)
        
        if q_after and q_before:
            eps_after = q_after.get('epsilon', 0.8)
            print(f"\n[After] Epsilon: {eps_after:.2f} (Delta {eps_after-eps_before:+.2f})")
            
            for ctx in ['calm', 'lively']:
                for br in ['LP', 'HP']:
                    old = q_before[ctx][br]
                    new = q_after[ctx][br]
                    if abs(new - old) > 0.0001:
                        print(f"        Q[{ctx}][{br}]: {old:.4f} → {new:.4f} (Delta {new-old:+.4f})")
        
        result.print_summary()
        self.test_results.append(result)
    
    def _scenario_multi_person(self):
        """
        Scenario 7: Multi-person target selection (biggest box)
        """
        print("""
══════════════════════════════════════════════════════════════════════
  SCENARIO 7: Multi-Person Target Selection
══════════════════════════════════════════════════════════════════════
  Tests: Module selects person with biggest bounding box
══════════════════════════════════════════════════════════════════════
""")
        self.world_state.clear_faces()
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        expected = ExpectedOutcome(
            speech_contains=["BigPerson"],
            actions=["ao_wave"],
            min_actions=1,
            max_wait_time=18.0,
            description="Multi-person target selection (biggest box)"
        )
        
        start_time = time.time()
        
        self.world_state.update_state(context_label=0)
        time.sleep(2)
        
        # Add 3 people with different box sizes
        print("[Setup] Adding 3 people with different box sizes:")
        self.world_state.add_face("SmallPerson", 0.9, [100, 100, 200, 200])  # 100x100
        print("        SmallPerson: 100x100 box")
        self.world_state.add_face("BigPerson", 0.95, [250, 100, 550, 500])   # 300x400
        print("        BigPerson: 300x400 box (LARGEST)")
        self.world_state.add_face("MediumPerson", 0.88, [600, 100, 800, 350])  # 200x250
        print("        MediumPerson: 200x250 box")
        
        self.world_state.update_state(valence=0.6, arousal=0.4)
        
        print("\n[Expected] Module should greet 'BigPerson' (largest box)")
        
        if not self._wait_for_speech(timeout=expected.max_wait_time, contains="BigPerson"):
            print("[Result] BigPerson not mentioned in speech within timeout")
        
        time.sleep(2)
        
        # Verify results
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("Multi-Person Selection", expected)
        
        # Check if BigPerson was greeted
        if any("BigPerson" in s for s in result.actual_speech):
            print("\n[Verification] BigPerson was correctly selected!")
        else:
            result.add_error("BigPerson was not greeted (should select largest box)")
        
        result.print_summary()
        self.test_results.append(result)
    
    # =========================================================================
    # Auto-Run Mode
    # =========================================================================
    
    def run_auto_scenarios(self, duration_minutes: float):
        """Run automated scenarios for specified duration."""
        print(f"""
══════════════════════════════════════════════════════════════════════
  AUTO-RUN MODE
══════════════════════════════════════════════════════════════════════
  Duration: {duration_minutes:.1f} minutes
  Will cycle through comprehensive test scenarios
══════════════════════════════════════════════════════════════════════
""")
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        scenario_count = 0
        
        # Snapshot initial Q
        self._snapshot_q_table()
        
        scenarios = [
            ("LP: New Person Greeting", self._auto_lp_greet),
            ("LP: Action Response", self._auto_lp_action),
            ("HP: Wave Only", self._auto_hp_wave),
            ("HP: New Person Greeting", self._auto_hp_greet),
            ("State Machine Test", self._auto_state_machine),
            ("Context Switch", self._auto_context_switch),
            ("High Valence Interaction", self._auto_high_valence),
            ("Low Valence Interaction", self._auto_low_valence),
        ]
        
        scenario_idx = 0
        
        while time.time() < end_time and self._running:
            scenario_count += 1
            name, func = scenarios[scenario_idx % len(scenarios)]
            
            elapsed = (time.time() - start_time) / 60
            remaining = (end_time - time.time()) / 60
            
            print(f"\n{'═'*70}")
            print(f"  AUTO #{scenario_count}: {name}")
            print(f"  Elapsed: {elapsed:.1f}m | Remaining: {remaining:.1f}m")
            print(f"{'═'*70}")
            
            try:
                func()
            except Exception as e:
                print(f"[Auto] Error: {e}")
            
            scenario_idx += 1
            
            if time.time() < end_time and self._running:
                print(f"\n[Auto] Cooldown 5s...")
                time.sleep(5)
        
        # Final summary
        total_time = (time.time() - start_time) / 60
        action_count = self.interaction_server.get_action_count()
        
        # Count test results
        passed = sum(1 for r in self.test_results if r.status == TestStatus.PASS)
        failed = sum(1 for r in self.test_results if r.status == TestStatus.FAIL)
        partial = sum(1 for r in self.test_results if r.status == TestStatus.PARTIAL)
        
        print(f"""
══════════════════════════════════════════════════════════════════════
  AUTO-RUN COMPLETE
══════════════════════════════════════════════════════════════════════
  Duration:        {total_time:.1f} minutes
  Scenarios:       {scenario_count}
  Robot Actions:   {action_count}
  Action Rate:     {action_count/max(0.1,total_time):.1f} per minute
  
  Test Results:    {passed} PASS, {failed} FAIL, {partial} WARN
══════════════════════════════════════════════════════════════════════
""")
        
        # Show Q-table changes
        self._print_q_table()
        
        # Show summary of failures
        if failed > 0 or partial > 0:
            print("\n" + "="*70)
            print("  ISSUES DETECTED")
            print("="*70)
            for result in self.test_results:
                if result.status in [TestStatus.FAIL, TestStatus.PARTIAL]:
                    print(f"\n[{result.status.value}] {result.scenario_name}")
                    for err in result.errors:
                        print(f"  Error: {err}")
                    for warn in result.warnings:
                        print(f"  Warning: {warn}")
        
        print("\n[Summary] Test harness stopped")
    
    def _auto_lp_greet(self):
        """Auto: LP greeting scenario with verification."""
        self.world_state.clear_faces()
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        expected = ExpectedOutcome(
            speech_contains=["ciao"],
            actions=["ao_wave"],
            min_actions=1,
            max_wait_time=25.0,  # Increased for ao_start + greeting time
            description="Auto LP greeting"
        )
        
        start_time = time.time()
        self.world_state.update_state(context_label=0)  # calm → LP
        time.sleep(1.5)
        
        name = f"Auto_{random.randint(1000,9999)}"
        print(f"[Auto] Adding '{name}' with large face box (LP path)")
        self.world_state.add_face(name, random.uniform(0.88, 0.98), [100, 100, 400, 500])
        self.world_state.update_state(
            valence=random.uniform(0.5, 0.8),
            arousal=random.uniform(0.3, 0.6)
        )
        
        self._wait_for_action(timeout=expected.max_wait_time, min_actions=1)
        time.sleep(1)
        
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("Auto LP Greeting", expected)
        if result.status == TestStatus.FAIL:
            result.print_summary()
        self.test_results.append(result)
    
    def _auto_lp_action(self):
        """Auto: LP action response with verification.
        
        Action response requires:
        - Person already greeted today (so tree waits for action)
        - Action detected for that person
        - LP tree selected with mutual gaze
        
        Note: This test works best AFTER a greeting test (person already greeted).
        """
        # Don't clear faces - we need the existing person who was greeted
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        # Pick random action from allowed list
        action = random.choice(list(self.ACTION_RESPONSES.keys()))
        expected_response = self.ACTION_RESPONSES.get(action)
        
        expected = ExpectedOutcome(
            min_actions=1,
            max_wait_time=20.0,
            description=f"Auto LP action response to '{action}'"
        )
        
        if expected_response:
            resp_type, resp_value = expected_response
            if resp_type == "speech":
                expected.speech_contains = [resp_value]
            elif resp_type == "ao":
                expected.actions = [resp_value]
        
        start_time = time.time()
        
        # Ensure conditions for LP tree action detection
        self.world_state.update_state(
            context_label=0,  # calm context for LP
            valence=0.6,
            arousal=0.4
        )
        
        # Ensure there's a face with large box for LP (don't clear faces from previous test)
        state = self.world_state.get_state()
        if not state['known_faces']:
            print("[Auto] Adding test face with large box (no face from previous test)")
            self.world_state.add_face("ActionTest", 0.9, [100, 100, 400, 500])
        
        print(f"[Auto] Triggering action: '{action}'")
        if expected_response:
            resp_type, resp_value = expected_response
            print(f"       Expected response: {resp_type}='{resp_value}'")
        
        # Set action for person index 0 (first face in list)
        # The module matches person_id to face index
        self.world_state.set_action("0", action, 0.9)
        
        self._wait_for_action(timeout=expected.max_wait_time, min_actions=1)
        time.sleep(1)
        
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("Auto LP Action", expected)
        if result.status == TestStatus.FAIL:
            result.print_summary()
        self.test_results.append(result)
    
    def _auto_hp_wave(self):
        """Auto: HP wave only (no known person) with verification.
        
        HP wave-only requires:
        - Face detected (faces_count > 0)
        - NO known faces (no face_id data)
        - HP tree selected (lively context or Q-learning chooses HP)
        """
        self.world_state.clear_faces()  # Clear known faces
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        expected = ExpectedOutcome(
            actions=["ao_wave"],
            min_actions=1,
            max_wait_time=25.0,
            description="Auto HP wave-only"
        )
        
        start_time = time.time()
        
        # Set lively context (higher chance of HP selection)
        self.world_state.update_state(context_label=1)  # lively
        time.sleep(1.5)
        
        # Set faces_count > 0 but NO known faces (don't call add_face)
        # This triggers HP wave-only path: face detected but identity unknown
        print("[Auto] Unknown face detected (HP wave-only path)")
        print("       - faces_count=1 but no known_faces")
        print("       - HP tree should wave without greeting")
        self.world_state.set_faces_count(1)  # Face detected  
        self.world_state.update_state(
            valence=0.5,
            arousal=0.5
        )
        
        self._wait_for_action(timeout=expected.max_wait_time, min_actions=1)
        time.sleep(1)
        
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("Auto HP Wave", expected)
        if result.status == TestStatus.FAIL:
            result.print_summary()
        self.test_results.append(result)
    
    def _auto_hp_greet(self):
        """Auto: HP greeting (known face, no gaze required) with verification.
        
        HP greeting requires:
        - Known face (face_id with name)
        - Person NOT greeted today
        - HP tree selected (lively context or Q-learning)
        - NO mutual gaze needed (unlike LP)
        """
        self.world_state.clear_faces()
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        expected = ExpectedOutcome(
            speech_contains=["ciao"],
            actions=["ao_wave"],
            min_actions=1,
            max_wait_time=20.0,
            description="Auto HP greeting"
        )
        
        start_time = time.time()
        
        # Set lively context (higher chance of HP selection)
        self.world_state.update_state(context_label=1)  # lively
        time.sleep(1.5)
        
        # Add known face for HP greeting path
        name = f"HP_{random.randint(1000,9999)}"
        print(f"[Auto] Adding known face '{name}' (HP path)")
        self.world_state.add_face(name, random.uniform(0.88, 0.98), [100, 100, 350, 400])
        self.world_state.update_state(
            valence=random.uniform(0.5, 0.7),
            arousal=random.uniform(0.5, 0.6)
        )
        
        self._wait_for_action(timeout=expected.max_wait_time, min_actions=1)
        time.sleep(1)
        
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("Auto HP Greeting", expected)
        if result.status == TestStatus.FAIL:
            result.print_summary()
        self.test_results.append(result)
    
    def _auto_state_machine(self):
        """Auto: Quick state machine test - verifies trees execute in ACTIVE state.
        
        Note: Testing ao_start/ao_stop properly requires 60s INACTIVE timeout.
        This auto test verifies that behavior trees execute when conditions are met,
        which confirms the module is in ACTIVE state and processing correctly.
        """
        self.world_state.clear_faces()
        self.world_state.clear_actions()
        self.interaction_server.clear_commands()
        self.speech_server.clear_speech()
        
        # Test that trees actually execute (proof of ACTIVE state)
        expected = ExpectedOutcome(
            speech_contains=["ciao"],
            actions=["ao_wave"],
            min_actions=1,
            max_wait_time=20.0,
            description="Auto state machine (tree execution)"
        )
        
        start_time = time.time()
        print("[Auto] Testing state machine - tree execution in ACTIVE state")
        
        # Add face with large box → should trigger greeting
        name = f"StateTest_{random.randint(1000,9999)}"
        print(f"[Auto] Adding '{name}' with large face box...")
        self.world_state.add_face(name, 0.9, [100, 100, 400, 500])
        self.world_state.update_state(context_label=0, valence=0.7, arousal=0.5)
        
        # Wait for greeting action
        self._wait_for_action(timeout=expected.max_wait_time, min_actions=1)
        time.sleep(1)
        
        # Check what happened
        cmds = [c['action'] for c in self.interaction_server.get_commands()]
        speech = self.speech_server.get_speech()
        
        if 'ao_wave' in cmds and speech:
            print("[Auto] PASS: Tree executed, greeting occurred")
        elif cmds:
            print(f"[Auto] Actions received: {cmds}")
        else:
            print("[Auto] WARNING: No actions received")
        
        expected.elapsed_time = time.time() - start_time
        result = self._verify_scenario("Auto State Machine", expected)
        if result.status == TestStatus.FAIL:
            result.print_summary()
        self.test_results.append(result)
    
    def _auto_context_switch(self):
        """Auto: Context switching test."""
        print("[Auto] Testing context switch response")
        self.world_state.clear_faces()
        self.world_state.add_face("CtxTest", 0.9, [100, 100, 400, 500])
        
        print("[Auto] Starting with calm context")
        self.world_state.update_state(context_label=0)
        time.sleep(4)
        
        print("[Auto] Switching to lively context")
        self.world_state.update_state(context_label=1)
        time.sleep(4)
    
    def _auto_high_valence(self):
        """Auto: High valence interaction."""
        self.world_state.clear_faces()
        self.world_state.update_state(context_label=0)
        time.sleep(1.5)
        
        name = f"Happy_{random.randint(1000,9999)}"
        print(f"[Auto] High valence interaction with '{name}'")
        self.world_state.add_face(name, 0.92, [100, 100, 400, 500])
        self.world_state.update_state(
            valence=random.uniform(0.7, 0.9),
            arousal=random.uniform(0.5, 0.7)
        )
        time.sleep(10)
    
    def _auto_low_valence(self):
        """Auto: Low valence interaction."""
        self.world_state.clear_faces()
        self.world_state.update_state(context_label=1)
        time.sleep(1.5)
        
        name = f"Neutral_{random.randint(1000,9999)}"
        print(f"[Auto] Low valence interaction with '{name}'")
        self.world_state.add_face(name, 0.88, [100, 100, 350, 400])
        self.world_state.update_state(
            valence=random.uniform(0.2, 0.4),
            arousal=random.uniform(0.2, 0.4)
        )
        time.sleep(10)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Test Harness for Embodied Behaviour",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python unified_test_harness.py
  
  # Auto-run for 10 minutes  
  python unified_test_harness.py --auto-run 10
  
  # Don't wait for module
  python unified_test_harness.py --no-wait
"""
    )
    
    parser.add_argument("--auto-run", type=float, metavar="MINUTES",
                       help="Run automated tests for N minutes")
    parser.add_argument("--no-connect", action="store_true",
                       help="Don't auto-connect to module")
    parser.add_argument("--no-wait", action="store_true",
                       help="Don't wait for module to start")
    
    args = parser.parse_args()
    
    harness = UnifiedTestHarness()
    
    try:
        if not harness.setup(
            auto_connect=not args.no_connect,
            wait_for_module=not args.no_wait
        ):
            return 1
        
        if args.auto_run:
            harness.run_auto_scenarios(args.auto_run)
        else:
            harness.run_interactive()
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        harness.cleanup()
        yarp.Network.fini()


if __name__ == "__main__":
    sys.exit(main())
