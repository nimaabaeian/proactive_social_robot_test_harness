#!/usr/bin/env python3
"""
Mock Services for Testing faceSelector and interactionManager

Provides mock implementations of:
1. Context Publisher (/alwayson/stm/context:o)
2. Acapela Speak TTS (/acapelaSpeak/*)
3. interactionInterface RPC (/interactionInterface)
"""

import yarp
import time
import random
import argparse
import threading
import re


CTX_UNCERTAIN = -1
CTX_CALM = 0
CTX_LIVELY = 1

PUBLISH_INTERVAL = 5.0  # Publish every 5 seconds
CONTEXT_CHANGE_INTERVAL = 180.0  # Change context every 3 minutes


# ==================== Context Publisher Mock ====================

class ContextPublisherMock:
    """Simulates /alwayson/stm/context:o port."""
    
    def __init__(self, port_name="/alwayson/stm/context:o", seed=None):
        self.port_name = port_name
        self.seed = seed
        self.port = None
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the context publisher in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"[MockContext] Started on {self.port_name}")
    
    def stop(self):
        """Stop the context publisher."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.port:
            self.port.close()
        print(f"[MockContext] Stopped")
    
    def _run(self):
        """Main loop for context publishing."""
        # Set random seed if provided
        if self.seed is not None:
            random.seed(self.seed)
            print(f"[MockContext] Random seed set to: {self.seed}")
        
        # Create output port
        self.port = yarp.BufferedPortBottle()
        if not self.port.open(self.port_name):
            print(f"[MockContext] ERROR: Failed to open port {self.port_name}")
            self.running = False
            return
        
        print(f"[MockContext] Port opened: {self.port_name}")
        print(f"[MockContext] Publishing every {PUBLISH_INTERVAL}s")
        print(f"[MockContext] Changing context every {CONTEXT_CHANGE_INTERVAL}s")
        print(f"[MockContext] Context values: UNCERTAIN={CTX_UNCERTAIN}, CALM={CTX_CALM}, LIVELY={CTX_LIVELY}")
        
        # Initialize timing
        last_publish = 0.0
        last_context_change = time.time()
        current_context = random.choice([CTX_UNCERTAIN, CTX_CALM, CTX_LIVELY])
        
        # Episode and chunk tracking
        episode_id = 0
        chunk_id = -1  # -1 for full episodes
        
        context_names = {
            CTX_UNCERTAIN: "UNCERTAIN",
            CTX_CALM: "CALM",
            CTX_LIVELY: "LIVELY"
        }
        
        print(f"[MockContext] Initial context: {context_names[current_context]} ({current_context})")
        print(f"[MockContext] Episode ID: {episode_id}, Chunk ID: {chunk_id}")
        
        try:
            while self.running:
                now = time.time()
                
                # Change context every 3 minutes
                if now - last_context_change >= CONTEXT_CHANGE_INTERVAL:
                    current_context = random.choice([CTX_UNCERTAIN, CTX_CALM, CTX_LIVELY])
                    last_context_change = now
                    episode_id += 1  # Increment episode on context change
                    print(f"[MockContext] Context changed to: {context_names[current_context]} ({current_context})")
                    print(f"[MockContext] Episode ID incremented to: {episode_id}")
                
                # Publish every 5 seconds
                if now - last_publish >= PUBLISH_INTERVAL:
                    bottle = self.port.prepare()
                    bottle.clear()
                    
                    # Structure: [Int32: episode_id] [Int32: chunk_id] [Int8: label]
                    bottle.addInt32(episode_id)
                    bottle.addInt32(chunk_id)
                    bottle.addInt8(current_context)
                    
                    self.port.write()
                    last_publish = now
                    print(f"[MockContext] Published: episode={episode_id}, chunk={chunk_id}, label={context_names[current_context]} ({current_context})")
                
                # Sleep to avoid busy waiting
                time.sleep(0.1)
        
        except Exception as e:
            print(f"[MockContext] ERROR: {e}")
        finally:
            self.port.close()


# ==================== Acapela Speak Mock ====================

class AcapelaSpeakMock(yarp.RFModule):
    """
    Simulates Acapela Speak TTS module.
    
    Ports:
    - /acapelaSpeak - RPC handler
    - /acapelaSpeak/speech:i - input for text to speak
    - /acapelaSpeak/bookmark:o - outputs timing bookmarks (0=start, 1=end)
    - /acapelaSpeak/emotion:o - outputs emotion commands (optional)
    """
    
    def __init__(self):
        super().__init__()
        self.speech_port = None
        self.bookmark_port = None
        self.emotion_port = None
        self.running = False
        
        # TTS simulation timing
        self.chars_per_second = 15.0  # Simulate speech speed
        self.current_emotion = "neutral"
        self.current_voice = "fabiana"
    
    def configure(self, rf):
        """Configure and open all ports."""
        self.running = True
        
        # Speech input port
        self.speech_port = yarp.Port()
        if not self.speech_port.open("/acapelaSpeak/speech:i"):
            print("[MockAcapela] ERROR: Failed to open /acapelaSpeak/speech:i")
            return False
        
        # Bookmark output port
        self.bookmark_port = yarp.Port()
        if not self.bookmark_port.open("/acapelaSpeak/bookmark:o"):
            print("[MockAcapela] ERROR: Failed to open /acapelaSpeak/bookmark:o")
            return False
        
        # Emotion output port
        self.emotion_port = yarp.Port()
        if not self.emotion_port.open("/acapelaSpeak/emotion:o"):
            print("[MockAcapela] ERROR: Failed to open /acapelaSpeak/emotion:o")
            return False
        
        print("[MockAcapela] All ports opened successfully")
        print("[MockAcapela] Ready to receive speech commands on /acapelaSpeak/speech:i")
        
        return True
    
    def close(self):
        """Close all ports."""
        self.running = False
        for port in [self.speech_port, self.bookmark_port, self.emotion_port]:
            if port:
                port.close()
        print("[MockAcapela] Closed all ports")
        return True
    
    def interruptModule(self):
        """Interrupt module."""
        self.running = False
        return True
    
    def getPeriod(self):
        return 0.01  # 100 Hz for responsive port reading
    
    def updateModule(self):
        """Main loop: read speech input and simulate TTS."""
        if not self.running:
            return False
        
        # Check for incoming speech commands
        bottle = yarp.Bottle()
        if self.speech_port.read(bottle, False):  # Non-blocking read
            text = bottle.toString()
            if text:
                threading.Thread(target=self._process_speech, args=(text,), daemon=True).start()
        
        return True
    
    def respond(self, command, reply):
        """Handle RPC commands."""
        reply.clear()
        
        cmd = command.get(0).asString()
        
        if cmd == "help":
            reply.addString("Available commands: help, list, set <emotion>")
            return True
        
        elif cmd == "list":
            voices = ["fabiana", "vittorio", "aurora", "alessio", "will", "karen"]
            reply.addString("Available voices: " + ", ".join(voices))
            return True
        
        elif cmd == "set":
            if command.size() > 1:
                self.current_emotion = command.get(1).asString()
                reply.addString(f"Emotion set to: {self.current_emotion}")
            else:
                reply.addString("Usage: set <emotion>")
            return True
        
        else:
            reply.addString(f"Unknown command: {cmd}")
            return True
    
    def _process_speech(self, text):
        """Simulate speech synthesis with timing."""
        # Clean text from YARP bottle formatting
        text = text.strip('"')
        
        # Extract bookmarks from text (supports both \mkr=N\ and \mrk=N\ formats)
        bookmark_pattern = r'\\m[kr]r?=(\d+)\\'
        bookmarks = {}
        
        # Find all bookmarks and their positions
        for match in re.finditer(bookmark_pattern, text):
            bookmark_id = int(match.group(1))
            position = match.start()
            bookmarks[position] = bookmark_id
        
        # Remove bookmark tags from text for length calculation
        clean_text = re.sub(bookmark_pattern, '', text)
        
        # Remove other tags (speed, volume, pause, etc.)
        clean_text = re.sub(r'\\[a-z]+\s*=\s*[^\\]*\\', '', clean_text)
        clean_text = re.sub(r'\\rst\\', '', clean_text)
        
        print(f"[MockAcapela] Speaking: {clean_text[:80]}{'...' if len(clean_text) > 80 else ''}")
        
        # Send start bookmark (0)
        self._send_bookmark(0)
        
        # Send any custom bookmarks found in text
        if bookmarks:
            print(f"[MockAcapela] Found {len(bookmarks)} custom bookmark(s)")
        
        # Simulate speech duration
        duration = max(0.5, len(clean_text) / self.chars_per_second)
        
        # Send intermediate bookmarks at appropriate times
        if bookmarks:
            sorted_bookmarks = sorted(bookmarks.items())
            for i, (pos, bm_id) in enumerate(sorted_bookmarks):
                if bm_id not in [0, 1]:  # Don't send 0 or 1 manually
                    # Calculate when to send this bookmark
                    delay = (pos / len(text)) * duration
                    time.sleep(delay)
                    self._send_bookmark(bm_id)
        else:
            # Just wait for speech duration
            time.sleep(duration)
        
        # Send end bookmark (1)
        self._send_bookmark(1)
        
        print(f"[MockAcapela] Finished speaking ({duration:.2f}s)")
    
    def _send_bookmark(self, bookmark_id):
        """Send bookmark to output port."""
        if not self.bookmark_port:
            print(f"[MockAcapela] WARNING: Cannot send bookmark {bookmark_id} - port not initialized")
            return
        
        bottle = yarp.Bottle()
        bottle.clear()
        bottle.addInt32(bookmark_id)
        self.bookmark_port.write(bottle)
        
        if self.bookmark_port.getOutputCount() > 0:
            print(f"[MockAcapela] Sent bookmark: {bookmark_id}")
        else:
            print(f"[MockAcapela] WARNING: Sent bookmark {bookmark_id} but no connections (getOutputCount=0)")


# ==================== interactionInterface Mock ====================

class InteractionInterfaceMock(yarp.RFModule):
    """
    Simulates interactionInterface RPC module.
    
    Responds to commands like:
    - exe <behavior>
    - help
    """
    
    # Common behaviors from icub_demos.sh
    BEHAVIORS = [
        "saluta", "annuisci", "nega", "presentati", "ciao", 
        "guarda_destra", "guarda_sinistra", "guarda_centro",
        "batti_mani", "alza_spalle", "pensa", "triste", "felice"
    ]
    
    def __init__(self):
        super().__init__()
        self.running = False
    
    def configure(self, rf):
        """Configure module (RPC port is auto-created by RFModule)."""
        self.running = True
        print("[MockInterface] RPC port /interactionInterface opened")
        print(f"[MockInterface] Available behaviors: {', '.join(self.BEHAVIORS[:5])}...")
        return True
    
    def close(self):
        """Close module."""
        self.running = False
        print("[MockInterface] Closed")
        return True
    
    def interruptModule(self):
        """Interrupt module."""
        self.running = False
        return True
    
    def getPeriod(self):
        return 1.0
    
    def updateModule(self):
        """Main loop."""
        return self.running
    
    def respond(self, command, reply):
        """Handle RPC commands."""
        reply.clear()
        
        if command.size() == 0:
            reply.addString("ERROR: No command provided")
            return True
        
        cmd = command.get(0).asString()
        
        if cmd == "help":
            help_text = """Available commands:
  exe <behavior>  - Execute a behavior
  list           - List available behaviors
  help           - Show this help"""
            reply.addString(help_text)
            return True
        
        elif cmd == "list":
            reply.addString("Available behaviors:\n  " + "\n  ".join(self.BEHAVIORS))
            return True
        
        elif cmd == "exe":
            if command.size() < 2:
                reply.addString("ERROR: Usage: exe <behavior>")
                return True
            
            behavior = command.get(1).asString()
            
            # Simulate behavior execution
            duration = random.uniform(1.0, 3.0)
            print(f"[MockInterface] Executing behavior: {behavior} ({duration:.1f}s)")
            time.sleep(duration)
            print(f"[MockInterface] Finished: {behavior}")
            
            reply.addString(f"OK: Executed {behavior}")
            return True
        
        else:
            reply.addString(f"ERROR: Unknown command: {cmd}")
            return True


# ==================== Main Entry Point ====================

def main():
    parser = argparse.ArgumentParser(
        description="Mock Services for Testing faceSelector and interactionManager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all mock services
  python mock_context_publisher.py --all
  
  # Run only Acapela and interactionInterface
  python mock_context_publisher.py --acapela --interface
  
  # Run only context publisher with custom seed
  python mock_context_publisher.py --context --seed 42
        """
    )
    
    # Service selection
    parser.add_argument("--all", action="store_true", help="Enable all mock services")
    parser.add_argument("--context", action="store_true", help="Enable context publisher")
    parser.add_argument("--acapela", action="store_true", help="Enable Acapela Speak mock")
    parser.add_argument("--interface", action="store_true", help="Enable interactionInterface mock")
    
    # Context publisher options
    parser.add_argument("--context-port", default="/alwayson/stm/context:o", 
                        help="Context output port name")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for context publisher")
    
    args = parser.parse_args()
    
    # Default to context-only if nothing specified (backward compatibility)
    if not (args.all or args.context or args.acapela or args.interface):
        args.context = True
    
    # If --all is specified, enable everything
    if args.all:
        args.context = True
        args.acapela = True
        args.interface = True
    
    # Initialize YARP
    yarp.Network.init()
    if not yarp.Network.checkNetwork():
        print("[MockServices] ERROR: YARP network not available")
        return 1
    
    print("=" * 60)
    print("Mock Services for Robot Testing")
    print("=" * 60)
    
    # Track active services
    services = []
    modules = []
    threads = []
    
    try:
        # Start Context Publisher
        if args.context:
            print(f"\n[1/3] Starting Context Publisher...")
            context_mock = ContextPublisherMock(args.context_port, args.seed)
            context_mock.start()
            services.append(context_mock)
            time.sleep(0.5)  # Give it time to open port
        
        # Start Acapela Speak Mock
        if args.acapela:
            print(f"\n[2/3] Starting Acapela Speak Mock...")
            acapela_mock = AcapelaSpeakMock()
            rf_acapela = yarp.ResourceFinder()
            rf_acapela.setVerbose(False)
            
            if acapela_mock.configure(rf_acapela):
                modules.append(acapela_mock)
                # Run in separate thread
                acapela_thread = threading.Thread(
                    target=lambda: acapela_mock.runModule(rf_acapela), 
                    daemon=True
                )
                acapela_thread.start()
                threads.append(acapela_thread)
                time.sleep(0.5)
            else:
                print("[MockServices] WARNING: Failed to start Acapela mock")
        
        # Start interactionInterface Mock
        if args.interface:
            print(f"\n[3/3] Starting interactionInterface Mock...")
            interface_mock = InteractionInterfaceMock()
            rf_interface = yarp.ResourceFinder()
            rf_interface.setVerbose(False)
            rf_interface.setDefault("name", "/interactionInterface")
            
            if interface_mock.configure(rf_interface):
                modules.append(interface_mock)
                # Run in separate thread
                interface_thread = threading.Thread(
                    target=lambda: interface_mock.runModule(rf_interface),
                    daemon=True
                )
                interface_thread.start()
                threads.append(interface_thread)
                time.sleep(0.5)
            else:
                print("[MockServices] WARNING: Failed to start interactionInterface mock")
        
        print("\n" + "=" * 60)
        print("All requested services are running!")
        print("=" * 60)
        
        if args.context:
            print(f"✓ Context Publisher: {args.context_port}")
        if args.acapela:
            print("✓ Acapela Speak:")
            print("    - /acapelaSpeak (RPC)")
            print("    - /acapelaSpeak/speech:i (input)")
            print("    - /acapelaSpeak/bookmark:o (output)")
            print("    - /acapelaSpeak/emotion:o (output)")
        if args.interface:
            print("✓ interactionInterface:")
            print("    - /interactionInterface (RPC)")
        
        print("\nPress Ctrl+C to stop all services")
        print("=" * 60)
        print()
        
        # Keep main thread alive
        while True:
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\n[MockServices] Interrupted by user")
    
    finally:
        print("\n[MockServices] Shutting down all services...")
        
        # Stop context publisher
        for service in services:
            service.stop()
        
        # Stop RFModule-based services
        for module in modules:
            module.interruptModule()
            module.close()
        
        yarp.Network.fini()
        print("[MockServices] Shutdown complete")
    
    return 0


if __name__ == "__main__":
    exit(main())
