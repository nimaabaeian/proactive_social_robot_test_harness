"""Behavior Tree builders for LP/HP proactivity branches."""

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status
import subprocess
import time
from typing import Optional, Tuple

ACTION_RESPONSES = {
    "answer phone": ("speak", "Salutalo da parte mia"),
    "drink": ("speak", "Salute"),
    "eat": ("speak", "Buon appetito"),
    "text on/look at a cellphone": ("ao", "ao_yawn_phone"),
    "hand wave": ("ao", "ao_wave"),
}


class ModuleAPI:
    """Interface for BT nodes to interact with the module."""
    def get_all_faces(self) -> list:
        raise NotImplementedError
    
    def get_known_faces(self) -> list:
        raise NotImplementedError
    
    def has_close_face(self, min_area: float = 15000.0) -> bool:
        raise NotImplementedError
    
    def get_max_face_area(self) -> float:
        raise NotImplementedError
    
    def select_target_by_biggest_box(self) -> Optional[Tuple[str, list]]:
        raise NotImplementedError
    
    def was_greeted_today(self, name: str) -> bool:
        raise NotImplementedError
    
    def mark_greeted_today(self, name: str):
        raise NotImplementedError
    
    def get_target_action(self, target_name: str, max_age: float = 5.0) -> Optional[str]:
        raise NotImplementedError
    
    def clear_detected_actions(self):
        raise NotImplementedError
    
    def start_va_capture(self):
        raise NotImplementedError
    
    def stop_va_capture_get_peak(self) -> tuple:
        raise NotImplementedError
    
    def update_learning(self, reward: float):
        raise NotImplementedError
    
    def get_context(self) -> int:
        raise NotImplementedError
    
    def get_branch(self) -> str:
        raise NotImplementedError
    
    def start_action_log(self, action_type: str, response_type: str, response_value: str,
                        trigger_reason: str, target_name: str, last_seen_today: bool) -> int:
        raise NotImplementedError
    
    def end_action_log(self, action_id: int, success: bool, note: str = ""):
        raise NotImplementedError
    
    def log_affect_summary(self, action_id: int, ts_start: float, samples: list,
                          reward: float, used_for_learning: bool):
        raise NotImplementedError


class AsyncActionBase(Behaviour):
    """Base class for non-blocking subprocess actions."""
    def __init__(self, name: str, api: ModuleAPI):
        super().__init__(name)
        self.api = api
        self._procs = []
        self._cmds = []
        self._started = False
        self._va_capturing = False
    
    def initialise(self):
        self._procs = []
        self._started = False
        self._va_capturing = False
        self._cmds = []
    
    def _start_execution(self):
        if self._started:
            return
        self._started = True
        self._ts_start = time.time()
        
        print(f"[BT:{self.name}] Initiating action execution")
        
        self._action_id = self.api.start_action_log(
            self._action_type, self._response_type, self._response_value,
            self._trigger_reason, self._target_name, self._last_seen_today
        )
        print(f"[BT:{self.name}] Action logged with ID={self._action_id}")
        
        self.api.start_va_capture()
        self._va_capturing = True
        print(f"[BT:{self.name}] VA capture started")
        
        for i, cmd in enumerate(self._cmds):
            try:
                proc = subprocess.Popen(cmd, shell=True,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                self._procs.append(proc)
                print(f"[BT:{self.name}] Command {i+1}/{len(self._cmds)} started (PID={proc.pid})")
            except Exception as e:
                print(f"[BT:{self.name}] ERROR: Command {i+1} failed - {e}")
    
    def update(self) -> Status:
        if not self._started:
            self._start_execution()
            return Status.RUNNING
        
        if not all(p.poll() is not None for p in self._procs):
            return Status.RUNNING
        
        print(f"[BT:{self.name}] All commands completed, processing results")
        success = True
        used_for_learning = False
        reward = 0.0
        
        if self._va_capturing:
            v, a = self.api.stop_va_capture_get_peak()
            self._va_capturing = False
            
            if v is not None and a is not None:
                from learning import compute_reward, clamp_reward
                reward = compute_reward(v, a)
                reward = clamp_reward(reward)
                print(f"[BT:{self.name}] VA captured: v={v:.2f}, a={a:.2f}, reward={reward:.3f}")
                self.api.update_learning(reward)
                used_for_learning = True
                print(f"[BT:{self.name}] Learning updated with reward={reward:.3f}")
                self.api.log_affect_summary(self._action_id, self._ts_start,
                                           [(v, a)], reward, used_for_learning)
            else:
                print(f"[BT:{self.name}] WARNING: No valid VA captured, skipping learning")
                success = False
        
        self.api.end_action_log(self._action_id, success)
        print(f"[BT:{self.name}] Action completed (success={success})")
        return Status.SUCCESS
    
    def terminate(self, new_status: Status):
        print(f"[BT:{self.name}] Terminating (status={new_status})")
        for p in self._procs:
            if p.poll() is None:
                try:
                    p.terminate()
                    print(f"[BT:{self.name}] Killed process PID={p.pid}")
                except:
                    pass
        self._procs = []
        
        if self._va_capturing:
            self.api.stop_va_capture_get_peak()
            self._va_capturing = False
            print(f"[BT:{self.name}] VA capture stopped")


class GreetWaveAction(AsyncActionBase):
    """Greet target with speech + wave. Marks person as greeted on success."""
    def __init__(self, api: ModuleAPI, target_name: str):
        super().__init__(f"GreetWave({target_name})", api)
        self.target_name = target_name
        self._action_type = "greet_wave"
        self._response_type = "speak+ao"
        self._response_value = f"Ciao {target_name} + ao_wave"
        self._target_name = target_name
        self._is_greeting = True
    
    def initialise(self):
        super().initialise()
        self._trigger_reason = "not_seen_today"
        self._last_seen_today = False
        self._cmds = [
            f'echo "Ciao {self.target_name}" | yarp write ... /acapelaSpeak/speech:i',
            'echo "exe ao_wave" | yarp rpc /interactionInterface'
        ]
    
    def update(self) -> Status:
        status = super().update()
        if status == Status.SUCCESS and self._target_name:
            self.api.mark_greeted_today(self._target_name)
            print(f"[BT:{self.name}] DATABASE: Marked '{self._target_name}' as greeted")
        return status


class WaveOnlyAction(AsyncActionBase):
    """Just wave (no known person)."""
    def __init__(self, api: ModuleAPI):
        super().__init__("WaveOnly", api)
        self._action_type = "wave_only"
        self._response_type = "ao"
        self._response_value = "ao_wave"
        self._trigger_reason = "no_known_person"
        self._target_name = ""
        self._last_seen_today = False
    
    def initialise(self):
        super().initialise()
        self._cmds = ['echo "exe ao_wave" | yarp rpc /interactionInterface']


class SpeakAction(AsyncActionBase):
    """Speak a phrase."""
    def __init__(self, api: ModuleAPI, phrase: str, name: str = "Speak"):
        super().__init__(name, api)
        self.phrase = phrase
        self._action_type = "speak"
        self._response_type = "speak"
        self._response_value = phrase
        self._trigger_reason = "direct"
        self._target_name = ""
        self._last_seen_today = False
    
    def initialise(self):
        super().initialise()
        self._cmds = [f'echo "{self.phrase}" | yarp write ... /acapelaSpeak/speech:i']


class AOAction(AsyncActionBase):
    """Execute an AO command."""
    def __init__(self, api: ModuleAPI, ao_cmd: str, name: str = "AO"):
        super().__init__(name, api)
        self.ao_cmd = ao_cmd
        self._action_type = "ao"
        self._response_type = "ao"
        self._response_value = ao_cmd
        self._trigger_reason = "direct"
        self._target_name = ""
        self._last_seen_today = False
    
    def initialise(self):
        super().initialise()
        self._cmds = [f'echo "exe {self.ao_cmd}" | yarp rpc /interactionInterface']


class ActionResponseAction(AsyncActionBase):
    """Execute response based on detected action."""
    def __init__(self, api: ModuleAPI, action_label: str):
        super().__init__(f"Response({action_label})", api)
        self.action_label = action_label
        self._action_type = "response"
        self._trigger_reason = "action_detected"
        self._target_name = ""
        self._last_seen_today = False
    
    def initialise(self):
        super().initialise()
        resp = ACTION_RESPONSES.get(self.action_label)
        if resp is None:
            self._cmds = []
            return
        
        resp_type, resp_val = resp
        self._response_type = resp_type
        self._response_value = resp_val
        
        if resp_type == "speak":
            self._cmds = [f'echo "{resp_val}" | yarp write ... /acapelaSpeak/speech:i']
        elif resp_type == "ao":
            self._cmds = [f'echo "exe {resp_val}" | yarp rpc /interactionInterface']


class CheckCloseFace(Behaviour):
    """Check if any face has area >= min_area."""
    def __init__(self, api: ModuleAPI, min_area: float = 15000.0):
        super().__init__("CheckCloseFace")
        self.api = api
        self.min_area = min_area
    
    def update(self) -> Status:
        return Status.SUCCESS if self.api.has_close_face(self.min_area) else Status.FAILURE


class CheckKnownPerson(Behaviour):
    """Check if any known person is visible."""
    def __init__(self, api: ModuleAPI):
        super().__init__("CheckKnownPerson")
        self.api = api
    
    def update(self) -> Status:
        return Status.SUCCESS if self.api.get_known_faces() else Status.FAILURE


class SelectTarget(Behaviour):
    """Select target person by biggest box, store in context."""
    def __init__(self, api: ModuleAPI, context: dict):
        super().__init__("SelectTarget")
        self.api = api
        self.context = context
    
    def update(self) -> Status:
        result = self.api.select_target_by_biggest_box()
        if result:
            name, box = result
            self.context["target_name"] = name
            self.context["target_box"] = box
            return Status.SUCCESS
        return Status.FAILURE


class WaitForAction(Behaviour):
    """Wait up to timeout for allowed action from target."""
    def __init__(self, api: ModuleAPI, context: dict, timeout: float = 5.0):
        super().__init__("WaitForAction")
        self.api = api
        self.context = context
        self.timeout = timeout
        self._start_time = 0.0
        self._started = False
    
    def initialise(self):
        self._start_time = time.time()
        self._started = True
        self.api.clear_detected_actions()
    
    def update(self) -> Status:
        if not self._started:
            self.initialise()
        
        name = self.context.get("target_name")
        if not name:
            return Status.FAILURE
        
        action = self.api.get_target_action(name, max_age=self.timeout)
        if action:
            self.context["detected_action"] = action
            return Status.SUCCESS
        
        if time.time() - self._start_time > self.timeout:
            return Status.FAILURE
        return Status.RUNNING
    
    def terminate(self, new_status: Status):
        self._started = False


class ExecuteActionResponse(AsyncActionBase):
    """Execute response for detected action from context."""
    def __init__(self, api: ModuleAPI, context: dict):
        super().__init__("ExecuteActionResponse", api)
        self.context = context
        self._action_type = "response"
        self._trigger_reason = "action_detected"
        self._target_name = context.get("target_name", "")
        self._last_seen_today = True
    
    def initialise(self):
        super().initialise()
        action = self.context.get("detected_action", "")
        resp = ACTION_RESPONSES.get(action)
        
        if resp is None:
            self._cmds = []
            return
        
        resp_type, resp_val = resp
        self._response_type = resp_type
        self._response_value = resp_val
        
        if resp_type == "speak":
            self._cmds = [f'echo "{resp_val}" | yarp write ... /acapelaSpeak/speech:i']
        elif resp_type == "ao":
            self._cmds = [f'echo "exe {resp_val}" | yarp rpc /interactionInterface']


class ExecuteGreetWave(AsyncActionBase):
    """Greet+wave using target from context. Marks person as greeted on success."""
    def __init__(self, api: ModuleAPI, context: dict):
        super().__init__("GreetWave", api)
        self.context = context
        self._action_type = "greet_wave"
        self._response_type = "speak+ao"
        self._trigger_reason = "not_seen_today"
        self._target_name = context.get("target_name", "")
        self._last_seen_today = False
    
    def initialise(self):
        super().initialise()
        name = self.context.get("target_name", "")
        self._target_name = name
        self._response_value = f"Ciao {name} + ao_wave"
        self._cmds = [
            f'echo "Ciao {name}" | yarp write ... /acapelaSpeak/speech:i',
            'echo "exe ao_wave" | yarp rpc /interactionInterface'
        ]
    
    def update(self) -> Status:
        status = super().update()
        if status == Status.SUCCESS and self._target_name:
            self.api.mark_greeted_today(self._target_name)
            print(f"[BT:{self.name}] DATABASE: Marked '{self._target_name}' as greeted")
        return status


class CheckNoKnownPerson(Behaviour):
    """Check if NO known person is visible."""
    def __init__(self, api: ModuleAPI):
        super().__init__("CheckNoKnownPerson")
        self.api = api
    
    def update(self) -> Status:
        return Status.SUCCESS if not self.api.get_known_faces() else Status.FAILURE


class IdleStartLP(Behaviour):
    """LP idle-start loop with internal cooldown."""
    def __init__(self, api: ModuleAPI):
        super().__init__("LP_IdleStart")
        self.api = api
        self.context = {}
        self._state = "idle"
        self._action_node = None
        self._action_wait_start = 0.0
        self._action_wait_timeout = 5.0
        self._cooldown_until = 0.0
    
    def initialise(self):
        self._state = "idle"
        self._action_node = None
        self._action_wait_start = 0.0
        self._cooldown_until = 0.0
        self.context = {}
    
    def update(self) -> Status:
        now = time.time()
        
        if self._state == "executing" and self._action_node:
            status = self._action_node.update()
            if status in (Status.SUCCESS, Status.FAILURE):
                result = "SUCCESS" if status == Status.SUCCESS else "FAILURE"
                print(f"[BT:LP_IdleStart] Execution finished with {result}, entering cooldown")
                self._enter_cooldown()
                return Status.RUNNING
            return Status.RUNNING
        
        if self._state == "cooldown":
            if now < self._cooldown_until:
                return Status.RUNNING
            print(f"[BT:LP_IdleStart] Cooldown expired, returning to idle")
            self._state = "idle"
            self.context = {}
            return Status.RUNNING
        
        if self._state == "waiting_action":
            name = self.context.get("target_name")
            action = self.api.get_target_action(name, max_age=self._action_wait_timeout)
            
            if action:
                print(f"[BT:LP_IdleStart] Action detected: '{action}' from '{name}'")
                self.context["detected_action"] = action
                self._action_node = ExecuteActionResponse(self.api, self.context)
                self._action_node.setup()
                self._action_node.initialise()
                self._state = "executing"
                return Status.RUNNING
            
            if now - self._action_wait_start >= self._action_wait_timeout:
                print(f"[BT:LP_IdleStart] Action wait timeout ({self._action_wait_timeout}s), entering cooldown")
                self._enter_cooldown()
            return Status.RUNNING
        
        # LP requires close face (large box area)
        if not self.api.has_close_face():
            return Status.RUNNING
        
        known_faces = self.api.get_known_faces()
        if not known_faces:
            return Status.RUNNING
        
        result = self.api.select_target_by_biggest_box()
        if not result:
            return Status.RUNNING
        
        name, box = result
        self.context["target_name"] = name
        self.context["target_box"] = box
        
        print(f"[BT:LP_IdleStart] Start conditions met: close_face=True, target='{name}'")
        
        if not self.api.was_greeted_today(name):
            print(f"[BT:LP_IdleStart] '{name}' not greeted today, executing greeting")
            self._action_node = ExecuteGreetWave(self.api, self.context)
            self._action_node.setup()
            self._action_node.initialise()
            self._state = "executing"
            return Status.RUNNING
        
        print(f"[BT:LP_IdleStart] '{name}' already greeted, waiting for action")
        self.api.clear_detected_actions()
        self._action_wait_start = now
        self._state = "waiting_action"
        return Status.RUNNING
    
    def _enter_cooldown(self):
        ctx = self.api.get_context()
        duration = 3.0 if ctx == 0 else 1.5
        self._cooldown_until = time.time() + duration
        self._state = "cooldown"
        self.context = {}
    
    def terminate(self, new_status: Status):
        if self._action_node and self._state == "executing":
            self._action_node.terminate(new_status)
        self._action_node = None


class IdleStartHP(Behaviour):
    """HP idle-start loop with internal cooldown."""
    def __init__(self, api: ModuleAPI):
        super().__init__("HP_IdleStart")
        self.api = api
        self.context = {}
        self._state = "idle"
        self._action_node = None
        self._action_wait_start = 0.0
        self._action_wait_timeout = 5.0
        self._cooldown_until = 0.0
    
    def initialise(self):
        self._state = "idle"
        self._action_node = None
        self._action_wait_start = 0.0
        self._cooldown_until = 0.0
        self.context = {}
    
    def update(self) -> Status:
        now = time.time()
        
        if self._state == "executing" and self._action_node:
            status = self._action_node.update()
            if status in (Status.SUCCESS, Status.FAILURE):
                result = "SUCCESS" if status == Status.SUCCESS else "FAILURE"
                print(f"[BT:HP_IdleStart] Execution finished with {result}, entering cooldown")
                self._enter_cooldown()
                return Status.RUNNING
            return Status.RUNNING
        
        if self._state == "cooldown":
            if now < self._cooldown_until:
                return Status.RUNNING
            self._state = "idle"
            self.context = {}
            return Status.RUNNING
        
        if self._state == "waiting_action":
            name = self.context.get("target_name")
            action = self.api.get_target_action(name, max_age=self._action_wait_timeout)
            
            if action:
                self.context["detected_action"] = action
                self._action_node = ExecuteActionResponse(self.api, self.context)
                self._action_node.setup()
                self._action_node.initialise()
                self._state = "executing"
                return Status.RUNNING
            
            if now - self._action_wait_start > self._action_wait_timeout:
                self._enter_cooldown()
            return Status.RUNNING
        
        known_faces = self.api.get_known_faces()
        
        if not known_faces:
            self._action_node = WaveOnlyAction(self.api)
            self._action_node.setup()
            self._action_node.initialise()
            self._state = "executing"
            return Status.RUNNING
        
        result = self.api.select_target_by_biggest_box()
        if not result:
            return Status.RUNNING
        
        name, box = result
        self.context["target_name"] = name
        self.context["target_box"] = box
        
        if not self.api.was_greeted_today(name):
            self._action_node = ExecuteGreetWave(self.api, self.context)
            self._action_node.setup()
            self._action_node.initialise()
            self._state = "executing"
            return Status.RUNNING
        
        self.api.clear_detected_actions()
        self._action_wait_start = now
        self._state = "waiting_action"
        return Status.RUNNING
    
    def _enter_cooldown(self):
        ctx = self.api.get_context()
        duration = 3.0 if ctx == 0 else 1.5
        self._cooldown_until = time.time() + duration
        self._state = "cooldown"
        self.context = {}
    
    def terminate(self, new_status: Status):
        if self._action_node and self._state == "executing":
            self._action_node.terminate(new_status)
        self._action_node = None


def build_lp_tree(api: ModuleAPI) -> Behaviour:
    """Build LP tree with idle-start pattern."""
    return IdleStartLP(api)


def build_hp_tree(api: ModuleAPI) -> Behaviour:
    """Build HP tree with idle-start pattern."""
    return IdleStartHP(api)
