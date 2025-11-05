# CELL 6: Fixed Analysis with YouTube Download
import yt_dlp
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import json

# YOUR YOUTUBE URL - Default video for analysis
youtube_url = "https://www.youtube.com/watch?v=awdBdYZSD1Q"

# Download function
def download_youtube_video(url):
    ydl_opts = {
        'format': 'best[height<=1080]',  # Max 1080p for faster processing
        'outtmpl': '/content/%(title)s.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return filename

# Download video
print("📥 Downloading video from YouTube...")
print("⏱️ This may take 5-10 minutes for long videos...")
video_file = download_youtube_video(youtube_url)
print(f"✅ Downloaded: {video_file}")

# Run analysis
print("🚀 Starting analysis...")
start_time = time.time()

passes, accuracy = analyzer.analyze_video(video_file)

end_time = time.time()
print(f"⏱️ Processing took: {(end_time - start_time)/60:.1f} minutes")

# FIXED: Enhanced results display with error handling
print("\n" + "="*60)
print("🏆 DETAILED FOOTBALL ANALYSIS RESULTS")
print("="*60)

# Overall stats with error handling
print(f"📊 TOTAL PASSES: {accuracy.get('total_passes', 0)}")
print(f"🎯 OVERALL ACCURACY: {accuracy.get('overall', 0):.1%}")
print(f"⏱️ PROCESSING TIME: {(end_time - start_time)/60:.1f} minutes")
print()

# Create detailed breakdown with error handling
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
    else:
        print(f"✅ Total Success: 0 (0.0%)")
        print(f"❌ Total Failure: 0 (0.0%)")
    print()
    
    # Accuracy by pass type
    print("🎯 ACCURACY BY PASS TYPE:")
    print("-" * 40)
    print(f"Short Pass Accuracy: {accuracy.get('short', 0):.1%}")
    print(f"Long Pass Accuracy: {accuracy.get('long', 0):.1%}")
    print(f"Overall Accuracy: {accuracy.get('overall', 0):.1%}")
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

else:
    print("❌ No passes detected in the video")
    print("💡 Possible reasons:")
    print("   - Video quality too low")
    print("   - No clear player movements")
    print("   - Model needs retraining")
    print("   - Try a different video")
    print("="*60)

# Create enhanced visualization with error handling
if len(df_passes) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Football Analysis Results', fontsize=16)

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
    if len(df_passes) > 0:
        axes[1, 1].hist(df_passes['distance'], bins=20, alpha=0.7, color='purple')
        axes[1, 1].set_title('Pass Distance Distribution')
        axes[1, 1].set_xlabel('Distance (pixels)')
        axes[1, 1].set_ylabel('Frequency')

    # 6. Confidence distribution
    if len(df_passes) > 0:
        axes[1, 2].hist(df_passes['confidence'], bins=20, alpha=0.7, color='orange')
        axes[1, 2].set_title('Confidence Distribution')
        axes[1, 2].set_xlabel('Confidence Score')
        axes[1, 2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Save detailed results with error handling
detailed_results = {
    'overall': {
        'total_passes': accuracy.get('total_passes', 0),
        'overall_accuracy': accuracy.get('overall', 0),
        'grade': grade if 'grade' in locals() else 'F (NO DETECTIONS)'
    },
    'short_passes': {
        'total': len(short_passes) if 'short_passes' in locals() else 0,
        'success': len(short_success) if 'short_success' in locals() else 0,
        'failure': len(short_failure) if 'short_failure' in locals() else 0,
        'success_rate': len(short_success)/len(short_passes)*100 if 'short_success' in locals() and 'short_passes' in locals() and len(short_passes) > 0 else 0,
        'accuracy': accuracy.get('short', 0)
    },
    'long_passes': {
        'total': len(long_passes) if 'long_passes' in locals() else 0,
        'success': len(long_success) if 'long_success' in locals() else 0,
        'failure': len(long_failure) if 'long_failure' in locals() else 0,
        'success_rate': len(long_success)/len(long_passes)*100 if 'long_success' in locals() and 'long_passes' in locals() and len(long_passes) > 0 else 0,
        'accuracy': accuracy.get('long', 0)
    }
}

# Save to files
df_passes.to_csv('detailed_football_results.csv', index=False)
with open('detailed_accuracy_metrics.json', 'w') as f:
    json.dump(detailed_results, f, indent=2)

print("\n✅ Detailed results saved!")
print("📁 Files created:")
print("  - detailed_football_results.csv")
print("  - detailed_accuracy_metrics.json")

# Download files
from google.colab import files
files.download('detailed_football_results.csv')
files.download('detailed_accuracy_metrics.json')

print("\n🎉 Complete analysis with detailed breakdown!")
