#!/bin/bash
# Full match analysis with complete tracking video

cd /home/essashah10/phase1-SMO-
source venv/bin/activate

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║       FULL MATCH ANALYSIS - Complete Tracking Video       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "This will:"
echo "  ✅ Analyze the ENTIRE match video"
echo "  ✅ Track all players and ball in EVERY frame"
echo "  ✅ Detect ALL passes (short/long)"
echo "  ✅ Create COMPLETE tracking video with overlays"
echo "  ✅ Generate detailed pass statistics"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Clean old output
if [ -d "output_frames" ]; then
    echo "🧹 Cleaning old frames..."
    rm -rf output_frames/*
fi

if [ -f "tracking_output.mp4" ]; then
    rm -f tracking_output.mp4
    echo "🧹 Removed old tracking video"
fi

echo ""
echo "🚀 Starting full match analysis..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run analysis (will take time for full match)
python3 main.py

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Analysis Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show results
if [ -f "tracking_output.mp4" ]; then
    echo "🎬 TRACKING VIDEO READY!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ls -lh tracking_output.mp4
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 tracking_output.mp4 2>/dev/null)
    if [ ! -z "$duration" ]; then
        minutes=$(echo "$duration / 60" | bc)
        echo "Duration: ${minutes} minutes"
    fi
    echo ""
    echo "📺 TO WATCH THE TRACKING VIDEO:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Option 1: Download to your computer"
    echo "  scp $(whoami)@$(hostname):$(pwd)/tracking_output.mp4 ./"
    echo ""
    echo "Option 2: Stream via HTTP (run in another terminal)"
    echo "  python3 serve_video.py"
    echo "  Then open: http://$(hostname -I | awk '{print $1}'):8000"
    echo ""
    echo "Option 3: View on server (if X11 available)"
    echo "  vlc tracking_output.mp4"
    echo "  # or"
    echo "  mpv tracking_output.mp4"
    echo ""
else
    echo "⚠️  Tracking video not created"
    echo "Check output_frames/ for saved frames"
fi

if [ -f "detailed_football_results.csv" ]; then
    pass_count=$(tail -n +2 detailed_football_results.csv 2>/dev/null | wc -l)
    echo "📊 PASS STATISTICS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Total passes detected: $pass_count"
    echo ""
    echo "First 5 passes:"
    head -6 detailed_football_results.csv | tail -5
    echo ""
fi

if [ -f "detailed_accuracy_metrics.json" ]; then
    echo "📈 ACCURACY METRICS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cat detailed_accuracy_metrics.json
    echo ""
fi

echo ""
echo "📁 All output files:"
echo "  - tracking_output.mp4 (full match with tracking)"
echo "  - detailed_football_results.csv (all passes)"
echo "  - detailed_accuracy_metrics.json (statistics)"
echo "  - output_frames/ (individual frames)"
echo ""

exit $EXIT_CODE

