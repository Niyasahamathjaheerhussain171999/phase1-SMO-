#!/usr/bin/env python3
"""
Football Video Analysis - Main Execution Script

This script integrates all the Google Colab cells into a single runnable Python script.
It orchestrates the entire analysis pipeline:

1. Setup (GPU check) - Cell 1
2. Dependencies are installed via requirements.txt
3. Video input (local file or YouTube download) - Cells 3-4
4. Analysis class initialization - Cell 5
5. Video analysis execution - Cell 6
6. Validation and results - Cell 7

Usage:
    python main.py --video path/to/video.mp4
    python main.py --youtube "https://youtube.com/watch?v=..."
    python main.py --video video.mp4 --class aplus --no-validation
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path

# Default YouTube video URL for analysis
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/shorts/Ziw3d8sQSq8"

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Workaround for OpenCV typing module issue (cv2.dnn.DictValue AttributeError)
# This is a known bug in OpenCV typing stubs - cv2.dnn.DictValue doesn't exist in some versions
# We need to ensure DictValue exists before cv2.typing tries to access it during bootstrap
import types
import importlib

# Pre-create cv2.dnn module with DictValue in sys.modules BEFORE cv2 imports
# This ensures it's available when cv2.typing tries to access it
if 'cv2.dnn' not in sys.modules:
    cv2_dnn_mock = types.ModuleType('cv2.dnn')
    
    class DictValue:
        """Mock DictValue class for OpenCV typing stubs compatibility"""
        pass
    
    cv2_dnn_mock.DictValue = DictValue
    sys.modules['cv2.dnn'] = cv2_dnn_mock

# Monkey-patch importlib.import_module to ensure DictValue exists when cv2.typing loads
_original_import_module = importlib.import_module

def _patched_import_module(name, package=None):
    """Patched import_module to handle cv2.typing DictValue issue"""
    # Before importing cv2.typing, ensure cv2.dnn.DictValue exists
    if name == 'cv2.typing' or (package and 'cv2' in str(package)):
        # Ensure cv2.dnn exists with DictValue
        if 'cv2' in sys.modules:
            cv2_module = sys.modules['cv2']
            if hasattr(cv2_module, 'dnn'):
                if not hasattr(cv2_module.dnn, 'DictValue'):
                    class DictValue:
                        pass
                    cv2_module.dnn.DictValue = DictValue
            else:
                # If cv2 doesn't have dnn yet, create it
                if 'cv2.dnn' in sys.modules:
                    cv2_module.dnn = sys.modules['cv2.dnn']
    
    result = _original_import_module(name, package)
    
    # After import, ensure DictValue exists
    if name == 'cv2' and hasattr(result, 'dnn'):
        if not hasattr(result.dnn, 'DictValue'):
            class DictValue:
                pass
            result.dnn.DictValue = DictValue
    
    return result

# Replace importlib.import_module with our patched version
importlib.import_module = _patched_import_module

# Also patch __import__ to catch cv2 imports
_original_import = __import__

def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Patched __import__ to handle cv2 bootstrap"""
    module = _original_import(name, globals, locals, fromlist, level)
    
    # If cv2 is being loaded, ensure dnn.DictValue exists
    if name == 'cv2':
        if hasattr(module, 'dnn'):
            if not hasattr(module.dnn, 'DictValue'):
                class DictValue:
                    pass
                module.dnn.DictValue = DictValue
        elif 'cv2.dnn' in sys.modules:
            # If dnn was pre-created, attach it
            module.dnn = sys.modules['cv2.dnn']
    
    return module

# Temporarily replace __import__
import builtins
builtins.__import__ = _patched_import

try:
    import cv2
    # Ensure DictValue exists after full import
    if hasattr(cv2, 'dnn'):
        if not hasattr(cv2.dnn, 'DictValue'):
            class DictValue:
                pass
            cv2.dnn.DictValue = DictValue
    else:
        # If dnn doesn't exist, use our pre-created one
        if 'cv2.dnn' in sys.modules:
            cv2.dnn = sys.modules['cv2.dnn']
finally:
    # Restore original functions
    builtins.__import__ = _original_import
    importlib.import_module = _original_import_module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import analysis classes
try:
    # Try importing from package (if __init__.py exports them)
    from classes import FixedFootballAnalysis, APlusFootballAnalysis, BalancedFootballAnalysis
except ImportError:
    # Fallback to direct imports
    from classes.colab_cell_5_class import FixedFootballAnalysis  # type: ignore
    from classes.colab_cell_5_class_A_PLUS import APlusFootballAnalysis  # type: ignore
    from classes.colab_cell_5_class_BALANCED import BalancedFootballAnalysis  # type: ignore

# Try to import validation functions and patch Colab dependencies
try:
    # Mock google.colab.files for local execution
    from unittest.mock import MagicMock
    
    # Create a mock module for google.colab
    mock_colab = MagicMock()
    mock_files = MagicMock()
    mock_files.download = lambda x: print(f"📁 File saved: {x} (local execution - no download needed)")
    mock_colab.files = mock_files
    
    # Inject the mock before importing validation
    sys.modules['google'] = MagicMock()
    sys.modules['google.colab'] = mock_colab
    sys.modules['google.colab.files'] = mock_files
    
    try:
        # Try importing from package (if __init__.py exports it)
        from validation import create_image_validation_system
    except ImportError:
        # Fallback to direct import
        from validation.colab_cell_7_image_validation import create_image_validation_system  # type: ignore
    validation_available = True
except Exception as e:
    print(f"⚠️ Warning: Could not import validation module: {e}")
    create_image_validation_system = None
    validation_available = False


def check_gpu():
    """Check if GPU is available"""
    print("🔍 Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        return 'cuda'
    else:
        print("⚠️ No GPU detected. Using CPU (will be slower)")
        return 'cpu'


def download_youtube_video(url, output_dir='downloads'):
    """Download video from YouTube with flexible format selection"""
    try:
        import yt_dlp
    except ImportError:
        print("❌ yt-dlp not installed. Install with: pip install yt-dlp")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📥 Downloading video from YouTube...")
    print(f"⏱️ This may take 5-10 minutes for long videos...")
    
    # Try multiple format options in order of preference
    format_options = [
        'best[height<=1080]',  # Best quality up to 1080p
        'best[height<=720]',   # Best quality up to 720p
        'best',                # Best available quality
        'worst',               # Worst quality (should always work)
    ]
    
    for format_choice in format_options:
        ydl_opts = {
            'format': format_choice,
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                
                # Handle case where file might have different extension
                if not os.path.exists(filename):
                    # Try to find the downloaded file
                    for file in os.listdir(output_dir):
                        if file.endswith(('.mp4', '.webm', '.mkv', '.m4a', '.flv')):
                            filename = os.path.join(output_dir, file)
                            break
                
                if os.path.exists(filename):
                    print(f"✅ Downloaded: {filename}")
                    print(f"   Format used: {format_choice}")
                    return filename
        except Exception as e:
            if format_choice == format_options[-1]:  # Last option failed
                print(f"❌ Error downloading video: {e}")
                print(f"   Tried formats: {', '.join(format_options)}")
                sys.exit(1)
            else:
                print(f"⚠️ Format '{format_choice}' not available, trying next option...")
                continue
    
    # If we get here, all formats failed
    print(f"❌ Error: Could not download video with any available format")
    sys.exit(1)


def get_analysis_class(class_type):
    """Get the appropriate analysis class"""
    class_map = {
        'fixed': FixedFootballAnalysis,
        'aplus': APlusFootballAnalysis,
        'balanced': BalancedFootballAnalysis,
        'default': FixedFootballAnalysis
    }
    
    return class_map.get(class_type.lower(), FixedFootballAnalysis)


def run_analysis(video_file, analyzer, device='cuda'):
    """Run the analysis on the video"""
    print("\n" + "="*60)
    print("🚀 STARTING FOOTBALL ANALYSIS")
    print("="*60)
    
    start_time = time.time()
    
    # Call analyze_video - check if method accepts device parameter
    import inspect
    sig = inspect.signature(analyzer.analyze_video)
    if 'device' in sig.parameters:
        passes, accuracy = analyzer.analyze_video(video_file, device=device)
    else:
        passes, accuracy = analyzer.analyze_video(video_file)
    
    end_time = time.time()
    processing_time = (end_time - start_time) / 60
    
    return passes, accuracy, processing_time


def display_results(passes, accuracy, processing_time):
    """Display detailed analysis results"""
    print("\n" + "="*60)
    print("🏆 DETAILED FOOTBALL ANALYSIS RESULTS")
    print("="*60)
    
    print(f"📊 TOTAL PASSES: {accuracy.get('total_passes', 0)}")
    print(f"🎯 OVERALL ACCURACY: {accuracy.get('overall', 0):.1%}")
    print(f"⏱️ PROCESSING TIME: {processing_time:.1f} minutes")
    print()
    
    df_passes = pd.DataFrame(passes) if passes else pd.DataFrame()
    
    if len(df_passes) > 0:
        # Short passes breakdown
        short_passes = df_passes[df_passes['type'] == 'short']
        short_success = short_passes[short_passes['success'] == True]
        short_failure = short_passes[short_passes['success'] == False]
        
        # Long passes breakdown
        long_passes = df_passes[df_passes['type'] == 'long']
        long_success = long_passes[long_passes['success'] == True]
        long_failure = long_passes[long_passes['success'] == False]
        
        print("📈 PASS TYPE BREAKDOWN:")
        print("-" * 40)
        
        # Short passes
        print(f"🔵 SHORT PASSES:")
        print(f"   Total: {len(short_passes)}")
        if len(short_passes) > 0:
            print(f"   ✅ Success: {len(short_success)} ({len(short_success)/len(short_passes)*100:.1f}%)")
            print(f"   ❌ Failure: {len(short_failure)} ({len(short_failure)/len(short_passes)*100:.1f}%)")
            print(f"   🎯 Accuracy: {accuracy.get('short', 0):.1%}")
        else:
            print(f"   ✅ Success: 0 (0.0%)")
            print(f"   ❌ Failure: 0 (0.0%)")
            print(f"   🎯 Accuracy: 0.0%")
        print()
        
        # Long passes
        print(f"🔴 LONG PASSES:")
        print(f"   Total: {len(long_passes)}")
        if len(long_passes) > 0:
            print(f"   ✅ Success: {len(long_success)} ({len(long_success)/len(long_passes)*100:.1f}%)")
            print(f"   ❌ Failure: {len(long_failure)} ({len(long_failure)/len(long_passes)*100:.1f}%)")
            print(f"   🎯 Accuracy: {accuracy.get('long', 0):.1%}")
        else:
            print(f"   ✅ Success: 0 (0.0%)")
            print(f"   ❌ Failure: 0 (0.0%)")
            print(f"   🎯 Accuracy: 0.0%")
        print()
        
        # Overall success/failure
        total_success = len(short_success) + len(long_success)
        total_failure = len(short_failure) + len(long_failure)
        total_passes = total_success + total_failure
        
        print("📊 OVERALL SUCCESS/FAILURE:")
        print("-" * 40)
        if total_passes > 0:
            print(f"✅ Total Success: {total_success} ({total_success/total_passes*100:.1f}%)")
            print(f"❌ Total Failure: {total_failure} ({total_failure/total_passes*100:.1f}%)")
        print()
        
        # Grade assessment
        overall_acc = accuracy.get('overall', 0)
        if overall_acc >= 0.95:
            grade = "A+ (EXCELLENT)"
            emoji = "🏆"
        elif overall_acc >= 0.90:
            grade = "A (VERY GOOD)"
            emoji = "🥇"
        elif overall_acc >= 0.85:
            grade = "B+ (GOOD)"
            emoji = "🥈"
        elif overall_acc >= 0.80:
            grade = "B (ACCEPTABLE)"
            emoji = "🥉"
        elif overall_acc > 0:
            grade = "C (NEEDS IMPROVEMENT)"
            emoji = "⚠️"
        else:
            grade = "F (NO DETECTIONS)"
            emoji = "❌"
        
        print(f"{emoji} FINAL GRADE: {grade}")
        print("="*60)
        
        return df_passes, grade
    else:
        print("❌ No passes detected in the video")
        print("💡 Possible reasons:")
        print("   - Video quality too low")
        print("   - No clear player movements")
        print("   - Model needs retraining")
        print("   - Try a different video")
        print("="*60)
        return pd.DataFrame(), "F (NO DETECTIONS)"


def create_visualizations(df_passes, accuracy):
    """Create visualization plots"""
    if len(df_passes) == 0:
        return
    
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Football Analysis Results', fontsize=16)
    
    short_passes = df_passes[df_passes['type'] == 'short']
    long_passes = df_passes[df_passes['type'] == 'long']
    short_success = short_passes[short_passes['success'] == True]
    short_failure = short_passes[short_passes['success'] == False]
    long_success = long_passes[long_passes['success'] == True]
    long_failure = long_passes[long_passes['success'] == False]
    total_success = len(short_success) + len(long_success)
    total_failure = len(short_failure) + len(long_failure)
    
    # 1. Pass type distribution
    pass_types = [len(short_passes), len(long_passes)]
    axes[0, 0].pie(pass_types, labels=['Short', 'Long'], autopct='%1.1f%%', colors=['blue', 'red'])
    axes[0, 0].set_title('Pass Type Distribution')
    
    # 2. Success/Failure by type
    categories = ['Short Success', 'Short Failure', 'Long Success', 'Long Failure']
    values = [len(short_success), len(short_failure), len(long_success), len(long_failure)]
    colors = ['lightgreen', 'lightcoral', 'darkgreen', 'darkred']
    axes[0, 1].bar(categories, values, color=colors)
    axes[0, 1].set_title('Success/Failure by Pass Type')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Accuracy comparison
    types = ['Short', 'Long', 'Overall']
    accuracies = [accuracy.get('short', 0), accuracy.get('long', 0), accuracy.get('overall', 0)]
    axes[0, 2].bar(types, accuracies, color=['blue', 'red', 'green'])
    axes[0, 2].set_title('Accuracy by Pass Type')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_ylim(0, 1)
    
    # 4. Success rate pie chart
    success_data = [total_success, total_failure]
    axes[1, 0].pie(success_data, labels=['Success', 'Failure'], autopct='%1.1f%%', colors=['green', 'red'])
    axes[1, 0].set_title('Overall Success Rate')
    
    # 5. Pass distance distribution
    axes[1, 1].hist(df_passes['distance'], bins=20, alpha=0.7, color='purple')
    axes[1, 1].set_title('Pass Distance Distribution')
    axes[1, 1].set_xlabel('Distance (pixels)')
    axes[1, 1].set_ylabel('Frequency')
    
    # 6. Confidence distribution
    axes[1, 2].hist(df_passes['confidence'], bins=20, alpha=0.7, color='orange')
    axes[1, 2].set_title('Confidence Distribution')
    axes[1, 2].set_xlabel('Confidence Score')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('detailed_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("✅ Saved visualization: detailed_analysis_plots.png")
    plt.close()


def save_results(df_passes, accuracy, grade):
    """Save results to files"""
    print("\n💾 Saving results...")
    
    # Save CSV
    if len(df_passes) > 0:
        df_passes.to_csv('detailed_football_results.csv', index=False)
        print("✅ Saved: detailed_football_results.csv")
        
        # Calculate detailed results
        short_passes = df_passes[df_passes['type'] == 'short']
        long_passes = df_passes[df_passes['type'] == 'long']
        short_success = short_passes[short_passes['success'] == True]
        long_success = long_passes[long_passes['success'] == True]
        
        detailed_results = {
            'overall': {
                'total_passes': accuracy.get('total_passes', 0),
                'overall_accuracy': float(accuracy.get('overall', 0)),
                'grade': grade
            },
            'short_passes': {
                'total': len(short_passes),
                'success': len(short_success),
                'failure': len(short_passes) - len(short_success),
                'success_rate': float(len(short_success)/len(short_passes)*100) if len(short_passes) > 0 else 0,
                'accuracy': float(accuracy.get('short', 0))
            },
            'long_passes': {
                'total': len(long_passes),
                'success': len(long_success),
                'failure': len(long_passes) - len(long_success),
                'success_rate': float(len(long_success)/len(long_passes)*100) if len(long_passes) > 0 else 0,
                'accuracy': float(accuracy.get('long', 0))
            }
        }
    else:
        detailed_results = {
            'overall': {
                'total_passes': 0,
                'overall_accuracy': 0.0,
                'grade': grade
            },
            'short_passes': {
                'total': 0,
                'success': 0,
                'failure': 0,
                'success_rate': 0,
                'accuracy': 0.0
            },
            'long_passes': {
                'total': 0,
                'success': 0,
                'failure': 0,
                'success_rate': 0,
                'accuracy': 0.0
            }
        }
    
    # Save JSON
    with open('detailed_accuracy_metrics.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print("✅ Saved: detailed_accuracy_metrics.json")


def run_validation(video_file, results_file='detailed_football_results.csv'):
    """Run validation if available"""
    if not validation_available or create_image_validation_system is None:
        print("\n⚠️ Validation module not available, skipping validation")
        return None
    
    print("\n" + "="*60)
    print("🔍 RUNNING VALIDATION")
    print("="*60)
    
    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"⚠️ Results file not found: {results_file}")
        print("⚠️ Skipping validation")
        return None
    
    # Create a mock uploaded dict for validation and set it in the validation module
    import validation.colab_cell_7_image_validation as validation_module
    import shutil
    
    # Use absolute path for video file
    abs_video_file = os.path.abspath(video_file)
    
    # The validation function expects 'a_plus_football_results.csv'
    # Copy our results file to that name temporarily
    original_file_ref = 'a_plus_football_results.csv'
    temp_file_created = False
    
    try:
        # Copy results file to expected name temporarily
        if os.path.exists(results_file) and results_file != original_file_ref:
            shutil.copy(results_file, original_file_ref)
            temp_file_created = True
            print(f"📋 Using results file: {results_file} (copied to {original_file_ref} for validation)")
        
        # Set the uploaded dict that the validation function expects
        # The validation function does: video_file = list(uploaded.keys())[0]
        # So the key needs to be the full absolute path for cv2.VideoCapture to work
        validation_module.uploaded = {abs_video_file: abs_video_file}
        
        validation_report = create_image_validation_system()
        
        # Clean up temp file if created
        if temp_file_created and os.path.exists(original_file_ref):
            os.remove(original_file_ref)
        
        print("\n✅ Validation complete!")
        return validation_report
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temp file if created
        if temp_file_created and os.path.exists(original_file_ref):
            try:
                os.remove(original_file_ref)
            except:
                pass
        
        return None


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Football Video Analysis - Main Execution Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze default YouTube video (no arguments needed)
  python main.py
  
  # Analyze local video file
  python main.py --video path/to/video.mp4
  
  # Analyze YouTube video
  python main.py --youtube "https://youtube.com/watch?v=..."
  
  # Use default YouTube video explicitly
  python main.py --use-default
  
  # Use A+ analysis class
  python main.py --video video.mp4 --class aplus
  
  # Run without validation
  python main.py --video video.mp4 --no-validation
        """
    )
    
    parser.add_argument('--video', type=str, help='Path to local video file')
    parser.add_argument('--youtube', type=str, default=None,
                       help=f'YouTube URL to download and analyze (default: {DEFAULT_YOUTUBE_URL})')
    parser.add_argument('--use-default', action='store_true',
                       help=f'Use default YouTube video: {DEFAULT_YOUTUBE_URL}')
    parser.add_argument('--class', dest='class_type', type=str, default='fixed',
                       choices=['fixed', 'aplus', 'balanced'],
                       help='Analysis class to use (default: fixed)')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip validation step')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto', help='Device to use (default: auto-detect)')
    parser.add_argument('--no-show', action='store_true',
                       help='Disable video visualization window')
    
    args = parser.parse_args()
    
    # Determine video source
    if args.video:
        if not os.path.exists(args.video):
            print(f"❌ Error: Video file not found: {args.video}")
            sys.exit(1)
        video_file = args.video
    elif args.youtube or args.use_default:
        # Use provided YouTube URL or default
        youtube_url = args.youtube if args.youtube else DEFAULT_YOUTUBE_URL
        print(f"📹 Using YouTube video: {youtube_url}")
        video_file = download_youtube_video(youtube_url)
    else:
        # Default to YouTube video if no arguments provided
        print(f"📹 No video specified, using default YouTube video: {DEFAULT_YOUTUBE_URL}")
        video_file = download_youtube_video(DEFAULT_YOUTUBE_URL)
    
    # Check GPU
    if args.device == 'auto':
        device = check_gpu()
    else:
        device = args.device
        print(f"🔧 Using device: {device}")
    
    # Initialize analyzer
    print(f"\n📦 Initializing {args.class_type} analysis class...")
    AnalysisClass = get_analysis_class(args.class_type)
    show_video = not args.no_show
    
    # Check if the class accepts show_video parameter
    import inspect
    sig = inspect.signature(AnalysisClass.__init__)
    if 'show_video' in sig.parameters:
        analyzer = AnalysisClass(show_video=show_video)
    else:
        analyzer = AnalysisClass()
    
    if show_video:
        print("📺 Video visualization enabled - A window will open showing the analysis")
        print("   Controls: Press Q to quit visualization, Space to pause/resume")
    
    # Run analysis
    passes, accuracy, processing_time = run_analysis(video_file, analyzer, device)
    
    # Display results
    df_passes, grade = display_results(passes, accuracy, processing_time)
    
    # Create visualizations
    if not args.no_viz and len(df_passes) > 0:
        create_visualizations(df_passes, accuracy)
    
    # Save results
    save_results(df_passes, accuracy, grade)
    
    # Run validation
    if not args.no_validation:
        run_validation(video_file)
    
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print("📁 Output files:")
    print("   - detailed_football_results.csv")
    print("   - detailed_accuracy_metrics.json")
    if not args.no_viz and len(df_passes) > 0:
        print("   - detailed_analysis_plots.png")
    if not args.no_validation:
        print("   - validation_images/ (if validation ran successfully)")
    print()


if __name__ == "__main__":
    main()

