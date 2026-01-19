import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns

from models.gan_text_detector import get_model as get_text_model
from models.image_spam_detector import get_model as get_image_model
from models.model_trainer import get_trainer
from utils.ocr_processor import get_ocr_processor
from utils.video_processor import get_video_processor
from utils.sample_data import get_sample_messages
from utils.analytics import get_analytics_tracker
from utils.batch_processor import get_batch_processor
from utils.pdf_reporter import get_pdf_reporter

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .spam-box {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .ham-box {
        background-color: #e6ffe6;
        border-left: 5px solid #44ff44;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize models and utilities
@st.cache_resource
def load_models():
    text_detector = get_text_model()
    image_detector = get_image_model()
    ocr_processor = get_ocr_processor()
    video_processor = get_video_processor()
    trainer = get_trainer(text_detector, image_detector)
    batch_processor = get_batch_processor(text_detector, image_detector, ocr_processor)
    pdf_reporter = get_pdf_reporter()
    return text_detector, image_detector, ocr_processor, video_processor, trainer, batch_processor, pdf_reporter

text_detector, image_detector, ocr_processor, video_processor, trainer, batch_processor, pdf_reporter = load_models()

# Initialize analytics tracker in session state
if 'analytics' not in st.session_state:
    st.session_state.analytics = get_analytics_tracker()

# Header
st.markdown('<div class="main-header">🛡️ SMS Spam Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">GAN-based Multi-Modal Spam Detection using Deep Learning</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Project Information")
    st.info("""
    **Final Year Project**
    
    This system uses Generative Adversarial Networks (GANs) and Convolutional Neural Networks (CNNs) to detect spam in:
    - 📱 Text messages (SMS)
    - 🖼️ Images 
    - 🎥 Videos
    
    **Features:**
    - Multi-modal detection
    - OCR text extraction
    - Real-time analysis
    - Confidence scoring
    """)
    
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Spam Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.header("📊 Model Architecture")
    st.write("**Text Detection:**")
    st.write("- GAN-based discriminator")
    st.write("- BiLSTM layers")
    st.write("- Embedding dimension: 128")
    
    st.write("**Image Detection:**")
    st.write("- CNN with 3 conv blocks")
    st.write("- Batch normalization")
    st.write("- Dropout regularization")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📱 Text Analysis", 
    "🖼️ Image Analysis", 
    "🎥 Video Analysis", 
    "📚 Sample Dataset",
    "🎓 Model Training",
    "📦 Batch Processing",
    "📊 Analytics Dashboard",
    "📈 Performance Metrics",
    "ℹ️ About"
])

# TAB 1: Text Analysis
with tab1:
    st.header("SMS Text Message Spam Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter SMS Message")
        text_input = st.text_area(
            "Type or paste your SMS message here:",
            height=150,
            placeholder="Enter the SMS message you want to analyze..."
        )
        
        analyze_text_btn = st.button("🔍 Analyze Text Message", type="primary", use_container_width=True)
        
        if analyze_text_btn and text_input:
            with st.spinner("Analyzing message with GAN model..."):
                spam_prob = text_detector.predict(text_input)
                
                st.subheader("📊 Analysis Results")
                
                # Classification result
                if spam_prob >= confidence_threshold:
                    st.markdown(f"""
                    <div class="spam-box">
                        <h3>⚠️ SPAM DETECTED</h3>
                        <p>This message has been classified as SPAM</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ham-box">
                        <h3>✅ LEGITIMATE MESSAGE</h3>
                        <p>This message appears to be legitimate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Spam Probability", f"{spam_prob:.1%}")
                with col_b:
                    st.metric("Legitimate Probability", f"{(1-spam_prob):.1%}")
                with col_c:
                    st.metric("Confidence", f"{max(spam_prob, 1-spam_prob):.1%}")
                
                # Probability gauge
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh([0], [spam_prob], color='red', alpha=0.7, label='Spam')
                ax.barh([0], [1-spam_prob], left=[spam_prob], color='green', alpha=0.7, label='Legitimate')
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel('Probability')
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2)
                ax.set_yticks([])
                plt.tight_layout()
                st.pyplot(fig)
                
                # Message statistics
                st.subheader("📝 Message Statistics")
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    st.metric("Word Count", len(text_input.split()))
                with col_y:
                    st.metric("Character Count", len(text_input))
                with col_z:
                    caps_ratio = sum(1 for c in text_input if c.isupper()) / max(len(text_input), 1)
                    st.metric("Caps Ratio", f"{caps_ratio:.1%}")
    
    with col2:
        st.subheader("💡 Quick Tips")
        st.info("""
        **Spam Indicators:**
        - Excessive caps
        - Multiple exclamation marks
        - Words like "FREE", "WIN", "PRIZE"
        - Suspicious URLs
        - Urgent language
        - Money/financial offers
        
        **Test the system:**
        - Try the sample messages
        - Enter your own text
        - Compare results
        """)

# TAB 2: Image Analysis
with tab2:
    st.header("Image-based Spam Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_image = st.file_uploader(
            "Choose an image file (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png'],
            key='image_upload'
        )
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Analyzing image..."):
                # Image spam detection
                img_spam_prob, features = image_detector.predict(image)
                
                # OCR text extraction
                ocr_result = ocr_processor.extract_text(image)
                
                with col_img2:
                    st.subheader("📊 Detection Results")
                    
                    if img_spam_prob >= confidence_threshold:
                        st.error("⚠️ SPAM IMAGE DETECTED")
                    else:
                        st.success("✅ LEGITIMATE IMAGE")
                    
                    st.metric("Image Spam Score", f"{img_spam_prob:.1%}")
                    
                    # Progress bar
                    st.progress(img_spam_prob)
                
                # OCR Results
                if ocr_result['text']:
                    st.subheader("📝 Extracted Text (OCR)")
                    st.text_area("Text found in image:", ocr_result['text'], height=100)
                    
                    # Analyze extracted text
                    if len(ocr_result['text']) > 10:
                        text_spam_prob = text_detector.predict(ocr_result['text'])
                        
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            st.metric("Text Spam Score", f"{text_spam_prob:.1%}")
                        with col_t2:
                            st.metric("OCR Confidence", f"{ocr_result['confidence']:.0f}%")
                        
                        # Combined analysis
                        combined_score = (img_spam_prob + text_spam_prob) / 2
                        st.subheader("🔬 Combined Analysis")
                        st.metric("Overall Spam Score", f"{combined_score:.1%}")
                        
                        if combined_score >= confidence_threshold:
                            st.warning("⚠️ High spam probability detected in both image and text!")
                else:
                    st.info("ℹ️ No readable text detected in the image")
                
                # Image features
                st.subheader("🔍 Image Features Analysis")
                feature_cols = st.columns(3)
                feature_list = [
                    ("Edge Density", features.get('edge_density', 0), "{:.3f}"),
                    ("Brightness", features.get('brightness', 0), "{:.1f}"),
                    ("Saturation", features.get('saturation', 0), "{:.1f}"),
                    ("Contrast", features.get('contrast', 0), "{:.1f}"),
                    ("Color Variance", features.get('color_variance', 0), "{:.1f}")
                ]
                
                for i, (name, value, fmt) in enumerate(feature_list):
                    with feature_cols[i % 3]:
                        st.metric(name, fmt.format(value))
    
    with col2:
        st.subheader("💡 Image Analysis Info")
        st.info("""
        **What we analyze:**
        - Visual spam patterns
        - Edge density (text)
        - Color saturation
        - Brightness levels
        - OCR text extraction
        
        **Image types:**
        - Screenshots of spam
        - Promotional images
        - Advertisement banners
        - Text-heavy images
        """)

# TAB 3: Video Analysis
with tab3:
    st.header("Video-based Spam Detection")
    
    st.subheader("Upload Video")
    uploaded_video = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key='video_upload'
    )
    
    if uploaded_video is not None:
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        st.video(uploaded_video)
        
        analyze_video_btn = st.button("🔍 Analyze Video", type="primary", use_container_width=True)
        
        if analyze_video_btn:
            with st.spinner("Extracting and analyzing video frames..."):
                # Get video info
                video_info = video_processor.get_video_info(video_path)
                
                if video_info:
                    st.subheader("📹 Video Information")
                    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                    with col_v1:
                        st.metric("Duration", f"{video_info['duration']:.1f}s")
                    with col_v2:
                        st.metric("Total Frames", video_info['total_frames'])
                    with col_v3:
                        st.metric("FPS", f"{video_info['fps']:.1f}")
                    with col_v4:
                        st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
                
                # Extract frames
                frames, frame_numbers = video_processor.extract_frames(video_path)
                
                if frames:
                    st.subheader(f"🖼️ Extracted {len(frames)} Frames for Analysis")
                    
                    # Show sample frames
                    cols = st.columns(min(5, len(frames)))
                    for i, (frame, frame_num) in enumerate(zip(frames[:5], frame_numbers[:5])):
                        with cols[i]:
                            st.image(frame, caption=f"Frame {frame_num}", use_container_width=True)
                    
                    # Analyze frames
                    with st.spinner("Analyzing frames for spam content..."):
                        results = video_processor.analyze_video_frames(
                            frames, image_detector, text_detector, ocr_processor
                        )
                    
                    st.subheader("📊 Analysis Results")
                    
                    # Overall scores
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("Avg Image Spam Score", f"{results['avg_image_spam_score']:.1%}")
                    with col_r2:
                        st.metric("Avg Text Spam Score", f"{results['avg_text_spam_score']:.1%}")
                    with col_r3:
                        spam_ratio = results['spam_frames'] / results['frame_count']
                        st.metric("Spam Frames", f"{results['spam_frames']}/{results['frame_count']}")
                    
                    # Overall classification
                    overall_score = (results['avg_image_spam_score'] + results['avg_text_spam_score']) / 2
                    
                    if overall_score >= confidence_threshold:
                        st.error("⚠️ SPAM VIDEO DETECTED")
                    else:
                        st.success("✅ LEGITIMATE VIDEO")
                    
                    st.metric("Overall Spam Probability", f"{overall_score:.1%}")
                    st.progress(overall_score)
                    
                    # Text extraction results
                    if results['text_detected']:
                        st.subheader("📝 Extracted Text from Frames")
                        for text_info in results['extracted_texts'][:5]:
                            with st.expander(f"Frame {text_info['frame']} (Confidence: {text_info['confidence']:.0f}%)"):
                                st.write(text_info['text'])
                    
                    # Frame-by-frame scores
                    if results['frame_scores']:
                        st.subheader("📈 Frame-by-Frame Analysis")
                        
                        frame_nums = [s['frame'] for s in results['frame_scores']]
                        img_scores = [s['image_score'] for s in results['frame_scores']]
                        txt_scores = [s['text_score'] for s in results['frame_scores']]
                        
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(frame_nums, img_scores, marker='o', label='Image Spam Score', linewidth=2)
                        ax.plot(frame_nums, txt_scores, marker='s', label='Text Spam Score', linewidth=2)
                        ax.axhline(y=confidence_threshold, color='r', linestyle='--', label='Threshold')
                        ax.set_xlabel('Frame Number')
                        ax.set_ylabel('Spam Probability')
                        ax.set_title('Spam Detection Across Video Frames')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.error("Failed to extract frames from video")
        
        # Clean up temp file
        try:
            os.unlink(video_path)
        except:
            pass

# TAB 4: Sample Dataset
with tab4:
    st.header("📚 Sample Dataset Showcase")
    
    samples = get_sample_messages()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚫 Spam Messages")
        for i, msg in enumerate(samples['spam'], 1):
            with st.expander(f"Spam Sample {i}"):
                st.write(msg)
                if st.button(f"Analyze", key=f"spam_{i}"):
                    prob = text_detector.predict(msg)
                    st.metric("Spam Probability", f"{prob:.1%}")
                    if prob >= 0.5:
                        st.error("✓ Correctly classified as SPAM")
                    else:
                        st.warning("✗ Misclassified as legitimate")
    
    with col2:
        st.subheader("✅ Legitimate Messages")
        for i, msg in enumerate(samples['legitimate'], 1):
            with st.expander(f"Legitimate Sample {i}"):
                st.write(msg)
                if st.button(f"Analyze", key=f"ham_{i}"):
                    prob = text_detector.predict(msg)
                    st.metric("Spam Probability", f"{prob:.1%}")
                    if prob < 0.5:
                        st.success("✓ Correctly classified as LEGITIMATE")
                    else:
                        st.warning("✗ Misclassified as spam")
    
    st.subheader("📊 Dataset Statistics")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Total Samples", len(samples['spam']) + len(samples['legitimate']))
    with col_s2:
        st.metric("Spam Messages", len(samples['spam']))
    with col_s3:
        st.metric("Legitimate Messages", len(samples['legitimate']))

# TAB 5: Model Training
with tab5:
    st.header("🎓 GAN Model Training Interface")
    
    st.info("""
    Train and improve the spam detection models with your own labeled data. 
    This allows the GAN discriminator and CNN to learn from new examples and improve accuracy.
    """)
    
    training_mode = st.radio("Select Training Mode:", ["Text Model", "Image Model"])
    
    if training_mode == "Text Model":
        st.subheader("📝 Train Text Spam Detector")
        
        st.write("Provide labeled text messages to train the GAN discriminator:")
        
        num_samples = st.number_input("Number of training samples:", min_value=1, max_value=50, value=5)
        
        texts = []
        labels = []
        
        for i in range(num_samples):
            col1, col2 = st.columns([3, 1])
            with col1:
                text = st.text_input(f"Message {i+1}:", key=f"train_text_{i}")
                texts.append(text)
            with col2:
                label = st.selectbox(f"Label {i+1}:", ["Legitimate", "Spam"], key=f"train_label_{i}")
                labels.append(1 if label == "Spam" else 0)
        
        epochs = st.slider("Training Epochs:", 1, 20, 5)
        
        if st.button("🚀 Train Text Model", type="primary"):
            valid_texts = [t for t in texts if t.strip()]
            if len(valid_texts) < 2:
                st.error("Please provide at least 2 valid text messages for training.")
            else:
                with st.spinner("Training model..."):
                    history = trainer.train_text_model(valid_texts, labels[:len(valid_texts)], epochs=epochs)
                    
                st.success(f"✅ Model trained successfully on {len(valid_texts)} samples!")
                
                # Show training metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Accuracy", f"{history.history['accuracy'][-1]:.1%}")
                with col2:
                    if 'val_accuracy' in history.history:
                        st.metric("Validation Accuracy", f"{history.history['val_accuracy'][-1]:.1%}")
                
                # Plot training history
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(history.history['accuracy'], label='Training')
                if 'val_accuracy' in history.history:
                    ax1.plot(history.history['val_accuracy'], label='Validation')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Model Accuracy')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(history.history['loss'], label='Training')
                if 'val_loss' in history.history:
                    ax2.plot(history.history['val_loss'], label='Validation')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title('Model Loss')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    else:  # Image Model
        st.subheader("🖼️ Train Image Spam Detector")
        
        st.write("Upload labeled images to train the CNN:")
        
        uploaded_images = st.file_uploader(
            "Choose image files", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key='train_images'
        )
        
        if uploaded_images:
            st.write(f"Uploaded {len(uploaded_images)} images")
            
            images = []
            labels = []
            
            cols = st.columns(min(4, len(uploaded_images)))
            for i, uploaded_file in enumerate(uploaded_images):
                img = Image.open(uploaded_file)
                images.append(img)
                
                with cols[i % 4]:
                    st.image(img, caption=f"Image {i+1}", use_container_width=True)
                    label = st.selectbox(f"Label:", ["Legitimate", "Spam"], key=f"img_label_{i}")
                    labels.append(1 if label == "Spam" else 0)
            
            epochs = st.slider("Training Epochs:", 1, 20, 5, key="img_epochs")
            
            if st.button("🚀 Train Image Model", type="primary"):
                with st.spinner("Training CNN model..."):
                    history = trainer.train_image_model(images, labels, epochs=epochs)
                
                st.success(f"✅ CNN trained successfully on {len(images)} images!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Accuracy", f"{history.history['accuracy'][-1]:.1%}")
                with col2:
                    if 'val_accuracy' in history.history:
                        st.metric("Validation Accuracy", f"{history.history['val_accuracy'][-1]:.1%}")

# TAB 6: Batch Processing
with tab6:
    st.header("📦 Batch Processing")
    
    st.info("Process multiple messages or images at once for efficient spam detection.")
    
    batch_mode = st.radio("Select Batch Mode:", ["Text Batch", "Image Batch"])
    
    if batch_mode == "Text Batch":
        st.subheader("📝 Batch Text Analysis")
        
        input_method = st.radio("Input Method:", ["Manual Entry", "Upload File"])
        
        if input_method == "Manual Entry":
            batch_text = st.text_area(
                "Enter messages (one per line):",
                height=200,
                placeholder="Message 1\nMessage 2\nMessage 3..."
            )
            
            if st.button("🔍 Analyze Batch", type="primary"):
                if batch_text.strip():
                    messages = [line.strip() for line in batch_text.split('\n') if line.strip()]
                    
                    with st.spinner(f"Processing {len(messages)} messages..."):
                        results = batch_processor.process_text_batch(messages)
                    
                    st.success(f"✅ Processed {len(results)} messages!")
                    
                    # Summary metrics
                    spam_count = sum(1 for r in results if r['is_spam'])
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Messages", len(results))
                    with col2:
                        st.metric("Spam Detected", spam_count)
                    with col3:
                        st.metric("Spam Rate", f"{spam_count/len(results):.1%}")
                    
                    # Results table
                    st.subheader("Results")
                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Record analytics
                    for r in results:
                        st.session_state.analytics.record_detection(
                            'text', r['is_spam'], r['spam_probability']
                        )
                    
                    # Export options
                    st.subheader("📥 Export Results")
                    if st.button("Generate PDF Report"):
                        summary_stats = {
                            'total_analyzed': len(results),
                            'total_spam': spam_count,
                            'total_ham': len(results) - spam_count,
                            'spam_rate': spam_count / len(results),
                            'avg_confidence': sum(r['spam_probability'] for r in results) / len(results)
                        }
                        pdf_buffer = pdf_reporter.generate_detection_report(results, 'text', summary_stats)
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"spam_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
        else:
            st.write("Upload a text file with one message per line")
            uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
            
            if uploaded_file and st.button("🔍 Analyze File"):
                content = uploaded_file.read().decode('utf-8')
                messages = [line.strip() for line in content.split('\n') if line.strip()]
                
                with st.spinner(f"Processing {len(messages)} messages..."):
                    results = batch_processor.process_text_batch(messages)
                
                st.success(f"✅ Processed {len(results)} messages!")
                
                import pandas as pd
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
    
    else:  # Image Batch
        st.subheader("🖼️ Batch Image Analysis")
        
        uploaded_images = st.file_uploader(
            "Upload multiple images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key='batch_images'
        )
        
        if uploaded_images and st.button("🔍 Analyze Images"):
            images = [Image.open(img) for img in uploaded_images]
            
            with st.spinner(f"Processing {len(images)} images..."):
                results = batch_processor.process_image_batch(images)
            
            st.success(f"✅ Processed {len(results)} images!")
            
            spam_count = sum(1 for r in results if r['is_spam'])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", len(results))
            with col2:
                st.metric("Spam Detected", spam_count)
            with col3:
                st.metric("Spam Rate", f"{spam_count/len(results):.1%}")
            
            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Record analytics
            for r in results:
                st.session_state.analytics.record_detection(
                    'image', r['is_spam'], r['spam_probability']
                )

# TAB 7: Analytics Dashboard
with tab7:
    st.header("📊 Analytics Dashboard")
    
    stats = st.session_state.analytics.get_summary_stats()
    
    if stats['total_analyzed'] == 0:
        st.info("No detection data available yet. Analyze some messages to see analytics!")
    else:
        # Summary metrics
        st.subheader("📈 Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyzed", stats['total_analyzed'])
        with col2:
            st.metric("Spam Detected", stats['total_spam'])
        with col3:
            st.metric("Legitimate", stats['total_ham'])
        with col4:
            st.metric("Spam Rate", f"{stats['spam_rate']:.1%}")
        
        st.metric("Average Confidence", f"{stats['avg_confidence']:.1%}")
        
        # Type breakdown
        st.subheader("📋 Analysis by Type")
        type_breakdown = st.session_state.analytics.get_type_breakdown()
        
        if type_breakdown:
            breakdown_data = []
            for input_type, counts in type_breakdown.items():
                breakdown_data.append({
                    'Type': input_type.title(),
                    'Total': counts['total'],
                    'Spam': counts['spam'],
                    'Legitimate': counts['ham'],
                    'Spam Rate': f"{counts['spam']/counts['total']:.1%}" if counts['total'] > 0 else "0%"
                })
            
            import pandas as pd
            df = pd.DataFrame(breakdown_data)
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            types = [d['Type'] for d in breakdown_data]
            totals = [d['Total'] for d in breakdown_data]
            spam_counts = [d['Spam'] for d in breakdown_data]
            
            ax1.bar(types, totals, alpha=0.7, label='Total')
            ax1.bar(types, spam_counts, alpha=0.7, label='Spam')
            ax1.set_ylabel('Count')
            ax1.set_title('Detections by Type')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.pie(totals, labels=types, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Distribution by Type')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Confidence distribution
        st.subheader("📊 Confidence Score Distribution")
        conf_dist = st.session_state.analytics.get_confidence_distribution()
        
        if conf_dist['counts']:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(conf_dist['bins'], conf_dist['counts'], width=0.08, alpha=0.7, color='steelblue')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Detection Confidence Scores')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Recent detections
        st.subheader("🕒 Recent Detections")
        recent = st.session_state.analytics.get_recent_detections(10)
        
        for detection in recent:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{detection['type'].title()}** - {detection['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            with col2:
                if detection['is_spam']:
                    st.error("SPAM")
                else:
                    st.success("LEGITIMATE")
            with col3:
                st.write(f"Confidence: {detection['confidence']:.1%}")

# TAB 8: Performance Metrics
with tab8:
    st.header("📈 Model Performance Metrics")
    
    st.info("""
    This section shows the performance metrics for the GAN and CNN models. 
    These metrics help evaluate how well the models are performing.
    """)
    
    # Model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 GAN Text Detector")
        text_info = text_detector.get_model_info()
        st.write(f"**Type:** {text_info['type']}")
        st.write(f"**Architecture:** {text_info['architecture']}")
        st.write(f"**Vocabulary Size:** {text_info['vocab_size']:,}")
        st.write(f"**Embedding Dimension:** {text_info['embedding_dim']}")
        st.write(f"**Max Sequence Length:** {text_info['max_length']}")
        st.write(f"**Trainable Parameters:** {text_info['trainable_params']:,}")
    
    with col2:
        st.subheader("🖼️ CNN Image Detector")
        img_info = image_detector.get_model_info()
        st.write(f"**Type:** {img_info['type']}")
        st.write(f"**Architecture:** {img_info['architecture']}")
        st.write(f"**Input Shape:** {img_info['input_shape']}")
        st.write(f"**Number of Layers:** {img_info['layers']}")
        st.write(f"**Trainable Parameters:** {img_info['trainable_params']:,}")
    
    # Training statistics
    training_stats = trainer.get_training_stats()
    
    if training_stats and training_stats['total_sessions'] > 0:
        st.subheader("🎓 Training Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Training Sessions", training_stats['total_sessions'])
        with col2:
            st.metric("Text Training Sessions", training_stats['text_sessions'])
        with col3:
            st.metric("Image Training Sessions", training_stats['image_sessions'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Text Samples", training_stats['total_text_samples'])
            if training_stats['text_sessions'] > 0:
                st.metric("Avg Text Model Accuracy", f"{training_stats['avg_text_accuracy']:.1%}")
        with col2:
            st.metric("Total Image Samples", training_stats['total_image_samples'])
            if training_stats['image_sessions'] > 0:
                st.metric("Avg Image Model Accuracy", f"{training_stats['avg_image_accuracy']:.1%}")
        
        # Export training report
        if st.button("📥 Export Analytics Report"):
            analytics_data = st.session_state.analytics.get_summary_stats()
            pdf_buffer = pdf_reporter.generate_analytics_report(analytics_data, training_stats)
            st.download_button(
                label="📄 Download Analytics PDF",
                data=pdf_buffer,
                file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    else:
        st.info("No training sessions yet. Use the Model Training tab to train the models with your own data!")
    
    # Theoretical metrics explanation
    st.subheader("📚 Performance Metrics Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Accuracy**")
        st.write("Percentage of correct predictions (both spam and legitimate)")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        
        st.write("**Precision**")
        st.write("Of all predicted spam, how many are actually spam")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
    
    with col2:
        st.write("**Recall (Sensitivity)**")
        st.write("Of all actual spam, how many did we detect")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        
        st.write("**F1-Score**")
        st.write("Harmonic mean of precision and recall")
        st.latex(r"\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
    
    st.write("""
    **Legend:**
    - TP (True Positive): Correctly identified spam
    - TN (True Negative): Correctly identified legitimate
    - FP (False Positive): Legitimate message marked as spam
    - FN (False Negative): Spam message marked as legitimate
    """)

# TAB 9: About
with tab9:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### SMS Spam Detection using GANs
    
    **Final Year Project - Academic Year 2024-2025**
    
    #### 🎯 Project Objectives
    This project demonstrates the application of Generative Adversarial Networks (GANs) and Deep Learning 
    for multi-modal spam detection in SMS messages, images, and videos.
    
    #### 🏗️ System Architecture
    
    **1. Text Spam Detection**
    - **Model**: GAN-based discriminator with BiLSTM layers
    - **Features**: Word embeddings, sequential patterns, contextual analysis
    - **Architecture**: 
        - Embedding layer (vocab: 5000, dim: 128)
        - Bidirectional LSTM layers (128 → 64 units)
        - Dense layers with dropout regularization
        - Sigmoid activation for binary classification
    
    **2. Image Spam Detection**
    - **Model**: Convolutional Neural Network (CNN)
    - **Features**: Edge density, color saturation, contrast, brightness
    - **Architecture**:
        - 3 convolutional blocks (32 → 64 → 128 filters)
        - Batch normalization and max pooling
        - Dense layers with dropout (512 → 256 → 1)
    
    **3. OCR Text Extraction**
    - **Engine**: Tesseract OCR
    - **Preprocessing**: Grayscale conversion, thresholding, denoising
    - **Output**: Extracted text with confidence scores
    
    **4. Video Analysis**
    - **Frame Extraction**: Uniform sampling across video duration
    - **Multi-modal Analysis**: Image features + OCR text + spam classification
    - **Aggregation**: Frame-level scores combined for overall verdict
    
    #### 🔬 Technical Stack
    - **Framework**: Streamlit (Web Interface)
    - **Deep Learning**: TensorFlow/Keras
    - **Image Processing**: OpenCV, Pillow
    - **OCR**: Pytesseract
    - **Data Science**: NumPy, Pandas, Scikit-learn
    - **Visualization**: Matplotlib, Seaborn
    
    #### 📈 Model Performance Considerations
    - **Precision**: Minimizing false positives (legitimate messages marked as spam)
    - **Recall**: Detecting maximum spam messages
    - **F1-Score**: Balance between precision and recall
    - **Confidence Threshold**: Adjustable for different use cases
    
    #### 🎓 Educational Value
    This project demonstrates:
    - GAN architecture and applications
    - Multi-modal machine learning
    - Deep learning for NLP and Computer Vision
    - Real-world spam detection challenges
    - End-to-end ML system development
    
    #### 🚀 Future Enhancements
    - Real-time SMS integration
    - Active learning with user feedback
    - Multi-language support
    - Advanced GAN training interface
    - Mobile application deployment
    - Cloud-based API service
    
    #### 📝 References
    - Generative Adversarial Networks (Goodfellow et al., 2014)
    - Deep Learning for NLP (Stanford CS224N)
    - Convolutional Neural Networks for Visual Recognition
    - SMS Spam Collection Dataset
    
    ---
    
    **Note**: This is an academic demonstration project. The models use simplified architectures 
    suitable for educational purposes and project demonstrations.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p><strong>SMS Spam Detection System using GANs</strong></p>
        <p>Final Year Project | Multi-Modal Deep Learning Approach</p>
        <p>Built with Streamlit, TensorFlow, and OpenCV</p>
    </div>
""", unsafe_allow_html=True)
