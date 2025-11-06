#!/bin/bash
# Run analysis with live tracking view

cd /home/essashah10/phase1-SMO-
source venv/bin/activate

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Football Analysis - WITH LIVE TRACKING VIEW             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Clean old frames
if [ -d "output_frames" ]; then
    echo "🧹 Cleaning old frames..."
    rm -rf output_frames/*
fi

# Start analysis in background
echo "🚀 Starting analysis in background..."
python3 main.py > analysis.log 2>&1 &
ANALYSIS_PID=$!

echo "✅ Analysis started (PID: $ANALYSIS_PID)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 LIVE TRACKING MONITOR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Watching for updates... (Press Ctrl+C to stop watching)"
echo ""

# Monitor progress
frame_count=0
last_frame_count=0

while kill -0 $ANALYSIS_PID 2>/dev/null; do
    sleep 2
    
    # Count frames
    if [ -d "output_frames" ]; then
        frame_count=$(ls output_frames/frame_*.jpg 2>/dev/null | wc -l)
    fi
    
    # Show progress
    if [ $frame_count -gt $last_frame_count ]; then
        echo "📈 Progress: $frame_count frames processed"
        last_frame_count=$frame_count
    fi
    
    # Show latest preview info
    if [ -f "output_frames/LIVE_PREVIEW.jpg" ]; then
        mod_time=$(stat -c %Y output_frames/LIVE_PREVIEW.jpg 2>/dev/null || echo 0)
        current_time=$(date +%s)
        age=$((current_time - mod_time))
        
        if [ $age -lt 5 ]; then
            echo "🖼️  NEW PREVIEW: output_frames/LIVE_PREVIEW.jpg (updated ${age}s ago)"
            echo "   💡 Download this file to see current tracking!"
        fi
    fi
    
    # Show latest log lines
    if [ -f "analysis.log" ]; then
        tail -1 analysis.log 2>/dev/null | grep -E "(Frame|Ball|Players|Passes|Saved)" || true
    fi
done

# Wait for analysis to finish
wait $ANALYSIS_PID
EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Analysis Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show final stats
if [ -d "output_frames" ]; then
    final_count=$(ls output_frames/frame_*.jpg 2>/dev/null | wc -l)
    echo "📁 Total frames saved: $final_count"
fi

if [ -f "tracking_output.mp4" ]; then
    echo "📺 Tracking video: tracking_output.mp4"
    ls -lh tracking_output.mp4
fi

if [ -f "detailed_football_results.csv" ]; then
    pass_count=$(tail -n +2 detailed_football_results.csv 2>/dev/null | wc -l)
    echo "📊 Passes detected: $pass_count"
fi

echo ""
echo "📋 Full log: analysis.log"
echo ""

exit $EXIT_CODE

