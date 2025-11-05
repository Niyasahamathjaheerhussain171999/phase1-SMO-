# CELL 7: Video Validation - Check AI Accuracy Against Real Video
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

def validate_video_insights():
    """Validate AI results against real video content"""
    print("🎯 VIDEO INSIGHTS VALIDATION SYSTEM")
    print("=" * 60)
    print("This will help you validate AI results against real video content")
    print()
    
    # Get the video file and results
    video_file = list(uploaded.keys())[0]  # Your uploaded video
    results_file = 'a_plus_football_results.csv'  # AI results
    
    print(f"📹 Video: {video_file}")
    print(f"📊 Results: {results_file}")
    print()
    
    # Load results
    try:
        results_df = pd.read_csv(results_file)
        print(f"✅ Loaded {len(results_df)} pass events")
    except:
        print("❌ No results file found. Run analysis first!")
        return
    
    # Load video
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {total_frames} frames at {fps} FPS")
    print(f"⏱️ Duration: {total_frames/fps/60:.1f} minutes")
    print()
    
    # 1. ANALYZE PASS PATTERNS
    print("📈 PASS PATTERN ANALYSIS")
    print("-" * 40)
    
    # Distance analysis
    distances = results_df['distance']
    print(f"📏 DISTANCE STATISTICS:")
    print(f"   Average: {distances.mean():.1f} pixels")
    print(f"   Min: {distances.min():.1f} pixels")
    print(f"   Max: {distances.max():.1f} pixels")
    print(f"   Median: {distances.median():.1f} pixels")
    
    # Pass type distribution
    pass_types = results_df['type'].value_counts()
    print(f"\n⚽ PASS TYPE DISTRIBUTION:")
    for ptype, count in pass_types.items():
        percentage = (count / len(results_df)) * 100
        print(f"   {ptype.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Success rate analysis
    success_rate = results_df['success'].mean()
    print(f"\n✅ SUCCESS RATE ANALYSIS:")
    print(f"   Overall Success Rate: {success_rate:.1%}")
    
    # Confidence analysis
    confidence = results_df['confidence']
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    print(f"   Average Confidence: {confidence.mean():.2f}")
    print(f"   Min Confidence: {confidence.min():.2f}")
    print(f"   Max Confidence: {confidence.max():.2f}")
    
    # 2. VALIDATE SAMPLE PASSES
    print(f"\n🔍 VALIDATING SAMPLE PASSES")
    print("-" * 40)
    
    # Get 5 sample passes
    sample_passes = results_df.sample(min(5, len(results_df)))
    
    validation_results = []
    
    for idx, pass_event in sample_passes.iterrows():
        frame_num = int(pass_event['frame'])
        passer_id = pass_event['passer_id']
        receiver_id = pass_event['receiver_id']
        distance = pass_event['distance']
        pass_type = pass_event['type']
        success = pass_event['success']
        confidence = pass_event['confidence']
        
        print(f"\n📊 PASS #{idx + 1} VALIDATION:")
        print(f"   Frame: {frame_num}")
        print(f"   Passer: {passer_id} → Receiver: {receiver_id}")
        print(f"   Distance: {distance:.1f} pixels")
        print(f"   Type: {pass_type}")
        print(f"   Success: {success}")
        print(f"   Confidence: {confidence:.2f}")
        
        # Show the actual frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Display frame info
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Pass: {passer_id}→{receiver_id}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Type: {pass_type} | Success: {success}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame for review
            frame_filename = f"validation_frame_{frame_num}_pass_{idx+1}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"   📸 Frame saved: {frame_filename}")
            
            validation_results.append({
                'frame': frame_num,
                'pass_id': idx + 1,
                'ai_prediction': {
                    'passer_id': passer_id,
                    'receiver_id': receiver_id,
                    'distance': distance,
                    'type': pass_type,
                    'success': success,
                    'confidence': confidence
                },
                'frame_saved': frame_filename
            })
    
    # 3. GENERATE VALIDATION PLOTS
    print(f"\n📊 GENERATING VALIDATION PLOTS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Video Validation Analysis', fontsize=16)
    
    # 1. Pass distance distribution
    axes[0, 0].hist(results_df['distance'], bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('Pass Distance Distribution')
    axes[0, 0].set_xlabel('Distance (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Confidence distribution
    axes[0, 1].hist(results_df['confidence'], bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Pass type distribution
    pass_types = results_df['type'].value_counts()
    axes[0, 2].pie(pass_types.values, labels=pass_types.index, autopct='%1.1f%%')
    axes[0, 2].set_title('Pass Type Distribution')
    
    # 4. Success rate by pass type
    success_by_type = results_df.groupby('type')['success'].mean()
    axes[1, 0].bar(success_by_type.index, success_by_type.values, color=['blue', 'red'])
    axes[1, 0].set_title('Success Rate by Pass Type')
    axes[1, 0].set_ylabel('Success Rate')
    
    # 5. Passes over time (frames)
    axes[1, 1].scatter(results_df['frame'], results_df['distance'], 
                      c=results_df['success'], cmap='RdYlGn', alpha=0.6)
    axes[1, 1].set_title('Passes Over Time')
    axes[1, 1].set_xlabel('Frame Number')
    axes[1, 1].set_ylabel('Distance (pixels)')
    
    # 6. Confidence vs Success
    axes[1, 2].scatter(results_df['confidence'], results_df['success'], 
                      alpha=0.6, color='purple')
    axes[1, 2].set_title('Confidence vs Success')
    axes[1, 2].set_xlabel('Confidence Score')
    axes[1, 2].set_ylabel('Success (1=Yes, 0=No)')
    
    plt.tight_layout()
    plt.savefig('validation_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. CREATE VALIDATION REPORT
    print(f"\n📋 CREATING VALIDATION REPORT")
    print("-" * 40)
    
    report = {
        'video_info': {
            'path': video_file,
            'total_frames': total_frames,
            'fps': fps,
            'duration_minutes': total_frames / fps / 60
        },
        'ai_results': {
            'total_passes': len(results_df),
            'short_passes': len(results_df[results_df['type'] == 'short']),
            'long_passes': len(results_df[results_df['type'] == 'long']),
            'success_rate': float(results_df['success'].mean()),
            'avg_confidence': float(results_df['confidence'].mean()),
            'avg_distance': float(results_df['distance'].mean())
        },
        'validation_samples': validation_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save report
    with open('video_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Validation report saved: video_validation_report.json")
    
    # 5. DOWNLOAD FILES
    from google.colab import files
    
    print(f"\n📁 DOWNLOADING VALIDATION FILES")
    print("-" * 40)
    
    # Download validation frames
    for result in validation_results:
        files.download(result['frame_saved'])
    
    # Download plots and report
    files.download('validation_analysis_plots.png')
    files.download('video_validation_report.json')
    
    print("\n🎉 VALIDATION COMPLETE!")
    print("=" * 60)
    print("📁 Files created and downloaded:")
    print("   - validation_frame_*.jpg (sample frames to review)")
    print("   - validation_analysis_plots.png (analysis charts)")
    print("   - video_validation_report.json (detailed report)")
    print()
    print("🔍 Next steps:")
    print("   1. Review the sample frames")
    print("   2. Check if AI detections match what you see")
    print("   3. Validate distance and pass type accuracy")
    print("   4. Use insights to improve AI model if needed")
    
    cap.release()
    return report

# Run validation
validation_report = validate_video_insights()
