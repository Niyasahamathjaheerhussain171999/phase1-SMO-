#!/usr/bin/env python3
"""
Simple HTTP server to stream the tracking video
Run this in a separate terminal, then access via browser
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000

class VideoHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for video streaming"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            # Serve index page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Football Analysis - Tracking Video</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background: #1a1a1a;
                        color: #fff;
                        margin: 0;
                        padding: 20px;
                    }
                    h1 {
                        color: #4CAF50;
                        text-align: center;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                    }
                    video {
                        width: 100%;
                        max-width: 1200px;
                        display: block;
                        margin: 20px auto;
                        border: 2px solid #4CAF50;
                        border-radius: 8px;
                    }
                    .info {
                        background: #2a2a2a;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 20px 0;
                    }
                    .legend {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 10px;
                        margin-top: 10px;
                    }
                    .legend-item {
                        background: #333;
                        padding: 10px;
                        border-radius: 4px;
                        border-left: 4px solid;
                    }
                    .team-a { border-left-color: #ff0000; }
                    .team-b { border-left-color: #0000ff; }
                    .ball { border-left-color: #00ff00; }
                    .pass { border-left-color: #ffff00; }
                    code {
                        background: #444;
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-family: monospace;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎥 Football Analysis - Full Match Tracking</h1>
                    
                    <div class="info">
                        <h2>📺 Watch Complete Tracking Video</h2>
                        <p>This video shows the full match with:</p>
                        <div class="legend">
                            <div class="legend-item team-a">
                                <strong>Red Boxes</strong> = Team A Players
                            </div>
                            <div class="legend-item team-b">
                                <strong>Blue Boxes</strong> = Team B Players
                            </div>
                            <div class="legend-item ball">
                                <strong>Green Box</strong> = Ball Tracking
                            </div>
                            <div class="legend-item pass">
                                <strong>Yellow Lines</strong> = Pass Connections
                            </div>
                        </div>
                    </div>
                    
                    <video controls autoplay>
                        <source src="/tracking_output.mp4" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    
                    <div class="info">
                        <h3>🎮 Video Controls:</h3>
                        <ul>
                            <li><strong>Space</strong> - Play/Pause</li>
                            <li><strong>← →</strong> - Skip backward/forward 5 seconds</li>
                            <li><strong>↑ ↓</strong> - Volume up/down</li>
                            <li><strong>F</strong> - Fullscreen</li>
                        </ul>
                        
                        <h3>🔍 What to Look For:</h3>
                        <ul>
                            <li>Are player boxes stable and accurate?</li>
                            <li>Is the ball being tracked correctly?</li>
                            <li>Do pass detections make sense?</li>
                            <li>Are short/long passes classified correctly?</li>
                        </ul>
                        
                        <h3>📥 Download:</h3>
                        <p>
                            <a href="/tracking_output.mp4" download style="color: #4CAF50; text-decoration: none;">
                                💾 Download tracking_output.mp4
                            </a>
                        </p>
                        
                        <h3>📊 Results Files:</h3>
                        <ul>
                            <li><a href="/detailed_football_results.csv" style="color: #4CAF50;">detailed_football_results.csv</a> - All passes</li>
                            <li><a href="/detailed_accuracy_metrics.json" style="color: #4CAF50;">detailed_accuracy_metrics.json</a> - Statistics</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            # Serve files normally
            super().do_GET()

if __name__ == '__main__':
    os.chdir('/home/essashah10/phase1-SMO-')
    
    # Check if tracking video exists
    if not Path('tracking_output.mp4').exists():
        print("❌ tracking_output.mp4 not found!")
        print("⏳ Run the analysis first:")
        print("   ./run_full_match_analysis.sh")
        exit(1)
    
    with socketserver.TCPServer(("", PORT), VideoHandler) as httpd:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║        VIDEO STREAMING SERVER - Ready to Watch!           ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print("")
        print(f"📺 Access the tracking video at:")
        print(f"   http://{local_ip}:{PORT}")
        print(f"   http://localhost:{PORT}")
        print("")
        print("🌐 From another computer on same network:")
        print(f"   http://{local_ip}:{PORT}")
        print("")
        print("Press Ctrl+C to stop the server")
        print("")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")

