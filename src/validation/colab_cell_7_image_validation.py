# CELL 7: IMAGE-BASED VALIDATION - Best for User Experience
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

def create_image_validation_system():
    """Create image-based validation system (user preferred)"""
    print("🎯 IMAGE-BASED VALIDATION SYSTEM")
    print("=" * 60)
    print("✅ BEST APPROACH: Image timestamps for instant validation")
    print("📱 User-friendly: View on any device, easy to share")
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
    duration_minutes = total_frames / fps / 60
    
    print(f"📹 Video: {total_frames} frames at {fps} FPS")
    print(f"⏱️ Duration: {duration_minutes:.1f} minutes")
    print()
    
    # Create validation folder
    os.makedirs('validation_images', exist_ok=True)
    
    # 1. CREATE VALIDATION IMAGES WITH TIMESTAMPS
    print("📸 CREATING VALIDATION IMAGES")
    print("-" * 40)
    
    # Get sample passes (10 for good validation)
    sample_passes = results_df.sample(min(10, len(results_df)))
    
    validation_images = []
    
    for idx, pass_event in sample_passes.iterrows():
        frame_num = int(pass_event['frame'])
        passer_id = pass_event['passer_id']
        receiver_id = pass_event['receiver_id']
        distance = pass_event['distance']
        pass_type = pass_event['type']
        success = pass_event['success']
        confidence = pass_event['confidence']
        
        # Calculate timestamp
        timestamp_seconds = frame_num / fps
        minutes = int(timestamp_seconds // 60)
        seconds = int(timestamp_seconds % 60)
        timestamp_str = f"{minutes:02d}:{seconds:02d}"
        
        print(f"📸 Creating validation image #{idx + 1}")
        print(f"   Frame: {frame_num} | Time: {timestamp_str}")
        print(f"   Pass: {passer_id}→{receiver_id} | Type: {pass_type}")
        print(f"   Distance: {distance:.1f}px | Success: {success}")
        
        # Get frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Create enhanced frame with validation info
            height, width = frame.shape[:2]
            
            # Add timestamp overlay
            cv2.putText(frame, f"Time: {timestamp_str}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add pass info
            cv2.putText(frame, f"Pass: Player {passer_id} → Player {receiver_id}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add details
            cv2.putText(frame, f"Type: {pass_type.upper()} | Distance: {distance:.0f}px", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add success/confidence
            success_color = (0, 255, 0) if success else (0, 0, 255)
            cv2.putText(frame, f"Success: {success} | Confidence: {confidence:.2f}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, success_color, 2)
            
            # Add frame number
            cv2.putText(frame, f"Frame: {frame_num}", (width - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add validation instructions
            cv2.putText(frame, "VALIDATE: Are there 2 players? Correct distance?", (10, height - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Save image
            image_filename = f"validation_images/validation_{idx+1:02d}_time_{timestamp_str.replace(':', '')}_frame_{frame_num}.jpg"
            cv2.imwrite(image_filename, frame)
            
            validation_images.append({
                'image_number': idx + 1,
                'frame_number': frame_num,
                'timestamp': timestamp_str,
                'timestamp_seconds': timestamp_seconds,
                'passer_id': passer_id,
                'receiver_id': receiver_id,
                'distance': distance,
                'pass_type': pass_type,
                'success': success,
                'confidence': confidence,
                'image_file': image_filename
            })
            
            print(f"   ✅ Saved: {image_filename}")
    
    # 2. CREATE VALIDATION SUMMARY
    print(f"\n📋 VALIDATION SUMMARY")
    print("-" * 40)
    
    # Calculate statistics
    total_passes = len(results_df)
    short_passes = len(results_df[results_df['type'] == 'short'])
    long_passes = len(results_df[results_df['type'] == 'long'])
    success_rate = results_df['success'].mean()
    avg_confidence = results_df['confidence'].mean()
    avg_distance = results_df['distance'].mean()
    
    print(f"📊 AI ANALYSIS RESULTS:")
    print(f"   Total Passes: {total_passes}")
    print(f"   Short Passes: {short_passes} ({short_passes/total_passes*100:.1f}%)")
    print(f"   Long Passes: {long_passes} ({long_passes/total_passes*100:.1f}%)")
    print(f"   Success Rate: {success_rate:.1%}")
    print(f"   Avg Confidence: {avg_confidence:.2f}")
    print(f"   Avg Distance: {avg_distance:.1f} pixels")
    
    # 3. CREATE VALIDATION REPORT
    print(f"\n📄 CREATING VALIDATION REPORT")
    print("-" * 40)
    
    validation_report = {
        'validation_info': {
            'method': 'image_timestamps',
            'user_preference': 'images_over_video',
            'total_validation_images': len(validation_images),
            'video_duration_minutes': duration_minutes
        },
        'video_info': {
            'file': video_file,
            'total_frames': total_frames,
            'fps': fps,
            'duration_minutes': duration_minutes
        },
        'ai_results': {
            'total_passes': int(total_passes),
            'short_passes': int(short_passes),
            'long_passes': int(long_passes),
            'success_rate': float(success_rate),
            'avg_confidence': float(avg_confidence),
            'avg_distance': float(avg_distance)
        },
        'validation_images': validation_images,
        'validation_instructions': {
            'step_1': 'Review each validation image',
            'step_2': 'Check if 2 players are visible',
            'step_3': 'Verify distance looks realistic',
            'step_4': 'Confirm pass type (short/long)',
            'step_5': 'Validate success/failure makes sense',
            'step_6': 'Note any incorrect detections'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Save report
    with open('validation_images/validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print("✅ Validation report saved: validation_images/validation_report.json")
    
    # 4. CREATE VALIDATION CHECKLIST
    print(f"\n📝 CREATING VALIDATION CHECKLIST")
    print("-" * 40)
    
    checklist_content = f"""
# 🎯 FOOTBALL AI VALIDATION CHECKLIST

## 📊 AI Analysis Results:
- **Total Passes**: {total_passes}
- **Short Passes**: {short_passes} ({short_passes/total_passes*100:.1f}%)
- **Long Passes**: {long_passes} ({long_passes/total_passes*100:.1f}%)
- **Success Rate**: {success_rate:.1%}
- **Average Confidence**: {avg_confidence:.2f}
- **Average Distance**: {avg_distance:.1f} pixels

## 📸 Validation Images to Review:
"""
    
    for img in validation_images:
        checklist_content += f"""
### Image #{img['image_number']:02d} - Time: {img['timestamp']}
- **Frame**: {img['frame_number']}
- **Pass**: Player {img['passer_id']} → Player {img['receiver_id']}
- **Type**: {img['pass_type'].upper()}
- **Distance**: {img['distance']:.1f} pixels
- **Success**: {img['success']}
- **Confidence**: {img['confidence']:.2f}
- **File**: {img['image_file']}

**Validation Questions:**
- [ ] Are there actually 2 players visible?
- [ ] Does the distance look realistic?
- [ ] Is this correctly classified as {img['pass_type']}?
- [ ] Does the success/failure make sense?
- [ ] Overall: Is this detection accurate?

---
"""
    
    checklist_content += f"""
## 🎯 Overall Validation:
- [ ] Most detections look accurate
- [ ] Distance ranges are realistic
- [ ] Pass types are correctly classified
- [ ] Success/failure rates make sense
- [ ] Confidence scores are reliable

## 📝 Notes:
_Add any observations or issues here:_



## ✅ Final Assessment:
- [ ] AI results are accurate and trustworthy
- [ ] Ready for frontend integration
- [ ] Can be shown to team/coaches
- [ ] No major issues found

**Validation completed on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('validation_images/validation_checklist.md', 'w') as f:
        f.write(checklist_content)
    
    print("✅ Validation checklist saved: validation_images/validation_checklist.md")
    
    # 5. DOWNLOAD ALL VALIDATION FILES
    print(f"\n📁 DOWNLOADING VALIDATION FILES")
    print("-" * 40)
    
    from google.colab import files
    
    # Download all validation images
    for img in validation_images:
        files.download(img['image_file'])
    
    # Download report and checklist
    files.download('validation_images/validation_report.json')
    files.download('validation_images/validation_checklist.md')
    
    print("\n🎉 IMAGE-BASED VALIDATION COMPLETE!")
    print("=" * 60)
    print("📁 Files created and downloaded:")
    print(f"   - {len(validation_images)} validation images with timestamps")
    print("   - validation_report.json (detailed data)")
    print("   - validation_checklist.md (easy review guide)")
    print()
    print("🔍 Next steps:")
    print("   1. Review each validation image")
    print("   2. Check if AI detections match reality")
    print("   3. Use checklist to track validation")
    print("   4. Share images with team for feedback")
    print("   5. Adjust AI settings if needed")
    
    cap.release()
    return validation_report

# Run image-based validation
validation_report = create_image_validation_system()
