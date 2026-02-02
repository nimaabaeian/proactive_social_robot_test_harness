#!/bin/bash
# Complete test run script with database consolidation
# Supports running module in separate terminal for visibility

# Parse arguments
CLEAN_DB=false
SEPARATE_TERMINAL=false
INTERACTIVE=false
DURATION=2

show_help() {
    echo "Usage: $0 [OPTIONS] [DURATION_MINUTES]"
    echo ""
    echo "Options:"
    echo "  --clean       Remove old databases and reset Q-table"
    echo "  --separate    Run module in a separate visible terminal"
    echo "  --interactive Run harness in interactive mode (no auto-run)"
    echo "  --help        Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --clean 5              # Clean run, 5 minute auto-test"
    echo "  $0 --separate --clean 10  # Module in separate terminal, 10 min"
    echo "  $0 --interactive          # Interactive mode"
    echo "  $0                         # Default: 2 minute auto-test"
}

# Parse all arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_DB=true
            shift
            ;;
        --separate)
            SEPARATE_TERMINAL=true
            shift
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            # Assume it's the duration
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                DURATION=$1
            fi
            shift
            ;;
    esac
done

echo "=============================================="
echo "  Embodied Behaviour - Complete Test Run"
echo "=============================================="
echo ""
echo "  Options:"
echo "    Clean DB:       $CLEAN_DB"
echo "    Separate Term:  $SEPARATE_TERMINAL"
echo "    Interactive:    $INTERACTIVE"
echo "    Duration:       $DURATION minutes"
echo ""

# Change to correct directory
cd "/home/nima/Desktop/TEST HARNESS"

# Check if YARP server is running
if ! yarp detect &> /dev/null; then
    echo "❌ YARP server not running!"
    echo "Please start: yarpserver --write"
    exit 1
fi
echo "✅ YARP server detected"

# Clean up old processes
echo "🧹 Cleaning up old processes..."
pkill -f "test_harness.py" 2>/dev/null
pkill -f "unified_test_harness.py" 2>/dev/null
pkill -f "embodied_behaviour.py" 2>/dev/null
sleep 2

# Clean up old database if requested
if [ "$CLEAN_DB" = true ]; then
    echo "🗑️  Removing old databases..."
    rm -f data_collection.db last_greeted.db q_table.json
    echo '{"calm": {"LP": 0.0, "HP": 0.0}, "lively": {"LP": 0.0, "HP": 0.0}, "epsilon": 0.8}' > q_table.json
fi

# Start embodied behaviour module
echo ""

if [ "$SEPARATE_TERMINAL" = true ]; then
    echo "🚀 Launching embodied behaviour in SEPARATE TERMINAL..."
    echo "   (Module output will be visible in the new terminal)"
    echo ""
    
    # Try different terminal emulators
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal --title="Embodied Behaviour Module" -- bash -c "cd '/home/nima/Desktop/TEST HARNESS' && python3 embodied_behaviour.py; echo ''; echo 'Press Enter to close...'; read"
    elif command -v xterm &> /dev/null; then
        xterm -title "Embodied Behaviour Module" -e "cd '/home/nima/Desktop/TEST HARNESS' && python3 embodied_behaviour.py; echo ''; echo 'Press Enter to close...'; read" &
    elif command -v konsole &> /dev/null; then
        konsole --new-tab -e bash -c "cd '/home/nima/Desktop/TEST HARNESS' && python3 embodied_behaviour.py; echo ''; echo 'Press Enter to close...'; read" &
    else
        echo "⚠️  No supported terminal found! Using background mode..."
        SEPARATE_TERMINAL=false
    fi
    
    if [ "$SEPARATE_TERMINAL" = true ]; then
        echo "⏳ Waiting for module to initialize (8 seconds)..."
        sleep 8
        EMBODIED_PID=$(pgrep -f "python3 embodied_behaviour.py" | head -1)
        if [ -z "$EMBODIED_PID" ]; then
            echo "❌ Module failed to start in separate terminal!"
            exit 1
        fi
        echo "✅ Module running in separate terminal (PID: $EMBODIED_PID)"
    fi
fi

if [ "$SEPARATE_TERMINAL" = false ]; then
    echo "🚀 Starting embodied behaviour module (background)..."
    python3 embodied_behaviour.py > /tmp/embodied_output.log 2>&1 &
    EMBODIED_PID=$!
    sleep 3

    # Check if module started
    if ! ps -p $EMBODIED_PID > /dev/null; then
        echo "❌ Module failed to start!"
        echo "Check log: cat /tmp/embodied_output.log"
        exit 1
    fi

    if grep -q "Configuration failed" /tmp/embodied_output.log; then
        echo "❌ Module configuration failed!"
        cat /tmp/embodied_output.log
        kill $EMBODIED_PID 2>/dev/null
        exit 1
    fi

    echo "✅ Module started (PID: $EMBODIED_PID)"
    tail -5 /tmp/embodied_output.log
fi

# Start test harness
echo ""

if [ "$INTERACTIVE" = true ]; then
    echo "🎯 Starting unified test harness in INTERACTIVE mode..."
    echo "   Type 'help' for commands, 'quit' to exit"
    echo ""
    python3 unified_test_harness.py
else
    echo "🎯 Starting unified test harness for $DURATION minutes..."
    echo "   (Press Ctrl+C to stop early)"
    echo ""
    python3 unified_test_harness.py --auto-run $DURATION
fi

# Capture exit code
TEST_EXIT=$?

# Cleanup
echo ""
echo "🛑 Stopping module..."
if [ -n "$EMBODIED_PID" ]; then
    kill $EMBODIED_PID 2>/dev/null
fi
pkill -f "embodied_behaviour.py" 2>/dev/null
sleep 1

# Show results
echo ""
echo "=============================================="
echo "  Test Results Summary"
echo "=============================================="

if [ -f "data_collection.db" ]; then
    echo ""
    echo "📊 Database Statistics:"
    sqlite3 data_collection.db <<EOF
.mode column
.headers on
SELECT 
    'Transitions' as Type, 
    COUNT(*) as Count 
FROM events 
WHERE event_type='transition'
UNION ALL
SELECT 
    'Selections' as Type, 
    COUNT(*) as Count 
FROM events 
WHERE event_type='selection'
UNION ALL
SELECT 
    'Actions' as Type,
    COUNT(*) as Count
FROM actions
UNION ALL
SELECT 
    'Affect Records' as Type,
    COUNT(*) as Count
FROM affect;
EOF
fi

if [ -f "q_table.json" ]; then
    echo ""
    echo "📈 Q-Table State:"
    cat q_table.json | python3 -m json.tool
fi

echo ""
echo "=============================================="
if [ $TEST_EXIT -eq 0 ]; then
    echo "✅ Test completed successfully!"
else
    echo "⚠️  Test interrupted or failed (exit code: $TEST_EXIT)"
fi
echo "=============================================="
echo ""
echo "📁 Files created:"
echo "   - data_collection.db (all logging)"
echo "   - last_greeted.db (tracks when each person was greeted)"
echo "   - q_table.json (Q-learning state)"
echo ""
echo "💡 To analyze data:"
echo "   sqlite3 data_collection.db"
echo "   > SELECT * FROM actions ORDER BY ts_start DESC LIMIT 10;"
echo ""
