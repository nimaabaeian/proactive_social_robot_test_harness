#!/usr/bin/env python3
"""Mock Context Publisher - Simulates /alwayson/stm/context:o port."""

import yarp
import time
import random
import argparse


CTX_UNCERTAIN = -1
CTX_CALM = 0
CTX_LIVELY = 1

PUBLISH_INTERVAL = 5.0  # Publish every 5 seconds
CONTEXT_CHANGE_INTERVAL = 180.0  # Change context every 3 minutes


def main():
    parser = argparse.ArgumentParser(description="Mock Context Publisher for YARP")
    parser.add_argument("--port", default="/alwayson/stm/context:o", help="Output port name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Initialize YARP
    yarp.Network.init()
    if not yarp.Network.checkNetwork():
        print("[MockContext] ERROR: YARP network not available")
        return 1
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"[MockContext] Random seed set to: {args.seed}")
    
    # Create output port
    port = yarp.BufferedPortBottle()
    if not port.open(args.port):
        print(f"[MockContext] ERROR: Failed to open port {args.port}")
        return 1
    
    print(f"[MockContext] Port opened: {args.port}")
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
        while True:
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
                bottle = port.prepare()
                bottle.clear()
                
                # Structure: [Int32: episode_id] [Int32: chunk_id] [Int8: label]
                # episode_id: The ID of the current episode
                # chunk_id: The chunk ID within the episode, or -1 for full episodes
                # label: The cluster label (context) assigned by HDBSCAN (0, 1, -1)
                bottle.addInt32(episode_id)  # Index 0: episode_id
                bottle.addInt32(chunk_id)  # Index 1: chunk_id (-1 for full episodes)
                bottle.addInt8(current_context)  # Index 2: context label
                
                port.write()
                last_publish = now
                print(f"[MockContext] Published: episode={episode_id}, chunk={chunk_id}, label={context_names[current_context]} ({current_context})")
            
            # Sleep to avoid busy waiting
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n[MockContext] Interrupted by user")
    finally:
        port.close()
        yarp.Network.fini()
        print("[MockContext] Shutdown complete")
    
    return 0


if __name__ == "__main__":
    exit(main())
