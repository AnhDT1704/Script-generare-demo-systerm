import streamlit as st
import os
import tempfile
import json
from datetime import datetime
import pandas as pd
from pathlib import Path
import shutil
from dotenv import load_dotenv

# Load config from environment or streamlit secrets
def get_config():
    try:
        # Ưu tiên sử dụng streamlit secrets
        return {
            "GEMINI_API_KEY": st.secrets["GEMINI_API_KEY"],
            "CLAUDE_API_KEY": st.secrets["CLAUDE_API_KEY"],
            "USE_OPENROUTER": st.secrets.get("USE_OPENROUTER", False)
        }
    except Exception as e:
        # Fallback to .env if exists
        if os.path.exists(".env"):
            load_dotenv()
            return {
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
                "CLAUDE_API_KEY": os.getenv("CLAUDE_API_KEY"),
                "USE_OPENROUTER": os.getenv("USE_OPENROUTER", "false").lower() == "true"
            }
        else:
            st.error("❌ Không tìm thấy API keys. Vui lòng thêm keys vào .streamlit/secrets.toml")
            st.stop()

# Import từ file gốc
try:
    from .grok_VU import EnhancedVideoScriptGenerator, VideoStorageManager, OptimizedVideoAnalyzer
except ImportError:
    try:
        from grok_VU import EnhancedVideoScriptGenerator, VideoStorageManager, OptimizedVideoAnalyzer
    except ImportError:
        st.error("Không thể import grok_VU.py. Hãy đảm bảo file grok_VU.py ở cùng thư mục với app này.")
        st.stop()

# Cấu hình trang
st.set_page_config(
    page_title="Video Analysis & Script Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Fix cho các ô trắng che chữ
st.markdown("""
<style>
    /* Reset container styling */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Header styling */
    .main-header {
        color: #2E86AB;
        text-align: center;
        padding: 10px 0;
        border-bottom: 2px solid #2E86AB;
        margin-bottom: 20px;
        background: transparent !important;
    }
    
    /* Video card styling - loại bỏ background có thể gây che chữ */
    .video-card {
        background: rgba(248, 249, 250, 0.8) !important;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 8px 0;
        backdrop-filter: none;
    }
    
    /* Stat metrics - đảm bảo không che chữ */
    .stat-metric {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        color: white !important;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        margin: 8px 0;
        position: relative;
        z-index: 1;
    }
    
    /* Fix cho Streamlit containers */
    .stContainer {
        background: transparent !important;
    }
    
    /* Fix cho expander */
    .streamlit-expanderHeader {
        background: transparent !important;
    }
    
    /* Fix cho columns */
    .row-widget {
        background: transparent !important;
    }
    
    /* Đảm bảo text không bị che */
    .stMarkdown, .stText {
        background: transparent !important;
        z-index: 10;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.generator = None
    st.session_state.uploaded_videos = []
    st.session_state.current_script = ""

def initialize_system():
    """Khởi tạo hệ thống với API keys"""
    if not st.session_state.initialized:
        try:
            config = get_config()
            st.session_state.generator = EnhancedVideoScriptGenerator(
                api_key=config["GEMINI_API_KEY"],
                claude_api_key=config["CLAUDE_API_KEY"],
                use_openrouter=config["USE_OPENROUTER"],
                storage_root="./streamlit_video_storage"
            )
            st.session_state.initialized = True
            return True
        except Exception as e:
            st.error(f"❌ Lỗi khởi tạo hệ thống: {str(e)}")
            return False
    return st.session_state.initialized

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Video Analysis & Script Generator</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar - Cấu hình
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Checkbox cho OpenRouter
        use_openrouter = st.checkbox("Sử dụng OpenRouter", 
                                    value=st.session_state.get('use_openrouter', True))
        
        # Lưu vào session state
        st.session_state.use_openrouter = use_openrouter
        
        # Kiểm tra kết nối
        if st.button("🔗 Kiểm tra kết nối"):
            if initialize_system():
                st.success("✅ Kết nối thành công!")
            else:
                st.error("❌ Lỗi kết nối. Vui lòng kiểm tra file .env")
    
    # Main interface
    if not initialize_system():
        st.info("❌ Vui lòng kiểm tra file .env và đảm bảo các API keys hợp lệ")
        return
    
    # Tabs chính
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Videos", "📊 Quản lý Videos", "✨ Tạo Script", "📋 Xem Scripts"])
    
    with tab1:
        upload_interface()
    
    with tab2:
        video_management_interface()
    
    with tab3:
        script_generation_interface()
    
    with tab4:
        script_history_interface()

def upload_interface():
    """Giao diện upload videos"""
    st.header("📤 Upload & Phân tích Videos")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Chọn video files", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        accept_multiple_files=True,
        help="Hỗ trợ: MP4, AVI, MOV, MKV. Tối đa 20MB/file để phân tích trực tiếp."
    )
    
    if uploaded_files:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"📹 {len(uploaded_files)} video(s) được chọn:")
            for file in uploaded_files:
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                st.markdown(f"""
                <div class="video-card">
                    <strong>{file.name}</strong><br>
                    📏 Kích thước: {file_size_mb:.1f}MB<br>
                    🎯 Trạng thái: {'✅ Phù hợp' if file_size_mb <= 50 else '⚠️ Có thể quá lớn'}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🚀 Upload & Phân tích", type="primary"):
                upload_and_analyze_videos(uploaded_files)

def upload_and_analyze_videos(uploaded_files):
    """Xử lý upload và phân tích videos"""
    if not st.session_state.generator:
        st.error("Hệ thống chưa được khởi tạo")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Tạo thư mục tạm
        temp_dir = tempfile.mkdtemp()
        video_paths = []
        
        # Lưu files tạm thời
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Đang lưu file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            video_paths.append(temp_path)
            
            progress_bar.progress((i + 1) / (len(uploaded_files) * 2))
        
        # Xử lý videos
        status_text.text("Đang lưu trữ và phân tích videos...")
        video_ids = st.session_state.generator.process_video_uploads(video_paths)
        
        # Phân tích nội dung
        analysis_results = st.session_state.generator.analyze_stored_videos(video_ids)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Hoàn thành!")
        
        # Hiển thị kết quả
        st.success(f"✅ Đã upload và phân tích {len(video_ids)} videos thành công!")
        
        for video_id in video_ids:
            video_info = st.session_state.generator.storage.get_video_info(video_id)
            if video_info:
                with st.expander(f"📹 {video_info.original_name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Thời lượng:** {video_info.duration:.1f}s")
                        st.write(f"**Kích thước:** {video_info.file_size_mb:.1f}MB")
                        st.write(f"**Độ phân giải:** {video_info.width}x{video_info.height}")
                    with col2:
                        st.write(f"**Trạng thái:** {video_info.status}")
                        st.write(f"**ID:** {video_info.id}")
                    
                    if video_info.analysis_result:
                        st.write("**📝 Phân tích nội dung:**")
                        st.info(video_info.analysis_result)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        st.error(f"❌ Lỗi xử lý: {e}")
        progress_bar.empty()
        status_text.empty()

def video_management_interface():
    """Giao diện quản lý videos"""
    st.header("📊 Quản lý Videos")
    
    if not st.session_state.generator:
        st.warning("Hệ thống chưa được khởi tạo")
        return
    
    # Thống kê tổng quan
    stats = st.session_state.generator.storage.get_storage_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-metric">
            <h3>{stats['total_videos']}</h3>
            <p>Tổng videos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-metric">
            <h3>{stats['total_size_mb']:.1f}MB</h3>
            <p>Dung lượng</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-metric">
            <h3>{stats['analyzed_videos']}</h3>
            <p>Đã phân tích</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Danh sách videos
    videos = st.session_state.generator.storage.get_all_videos()
    
    if videos:
        st.subheader("📹 Danh sách Videos")
        
        # Tạo DataFrame
        video_data = []
        for video in videos:
            video_data.append({
                "Tên file": video.original_name,
                "ID": video.id[:8] + "...",
                "Thời lượng (s)": f"{video.duration:.1f}",
                "Kích thước (MB)": f"{video.file_size_mb:.1f}",
                "Trạng thái": video.status,
                "Upload time": video.upload_time[:19],
                "Đã phân tích": "✅" if video.analysis_result else "❌"
            })
        
        df = pd.DataFrame(video_data)
        st.dataframe(df, use_container_width=True)
        
        # Chi tiết video được chọn
        selected_video_name = st.selectbox(
            "Chọn video để xem chi tiết:",
            options=[v.original_name for v in videos],
            index=0
        )
        
        if selected_video_name:
            selected_video = next(v for v in videos if v.original_name == selected_video_name)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📋 Chi tiết Video")
                st.write(f"**Tên gốc:** {selected_video.original_name}")
                st.write(f"**ID:** {selected_video.id}")
                st.write(f"**Thời lượng:** {selected_video.duration:.1f} giây")
                st.write(f"**Độ phân giải:** {selected_video.width}x{selected_video.height}")
                st.write(f"**Kích thước:** {selected_video.file_size_mb:.1f}MB")
                st.write(f"**Trạng thái:** {selected_video.status}")
                st.write(f"**Upload:** {selected_video.upload_time}")
                
                if selected_video.analysis_result:
                    st.subheader("🔍 Kết quả phân tích")
                    st.info(selected_video.analysis_result)
                else:
                    st.warning("Chưa có kết quả phân tích")
            
            with col2:
                st.subheader("🛠️ Thao tác")
                
                # Phân tích lại
                if st.button("🔄 Phân tích lại", key=f"analyze_{selected_video.id}"):
                    analyze_single_video(selected_video.id)
                
                # Nén video
                if st.button("🗜️ Nén video", key=f"compress_{selected_video.id}"):
                    compress_single_video(selected_video.id)
                
                # Xóa video
                if st.button("🗑️ Xóa video", key=f"delete_{selected_video.id}", type="secondary"):
                    delete_single_video(selected_video.id)
        
        # Dọn dẹp hàng loạt
        st.subheader("🧹 Dọn dẹp hàng loạt")
        col1, col2 = st.columns(2)
        
        with col1:
            days_old = st.number_input("Xóa videos cũ hơn (ngày):", min_value=1, value=30)
            if st.button("🧹 Dọn dẹp videos cũ"):
                cleanup_old_videos(days_old)
        
        with col2:
            if st.button("📊 Phân tích tất cả videos chưa xử lý"):
                analyze_all_unprocessed()
    
    else:
        st.info("📭 Chưa có video nào. Hãy upload videos ở tab đầu tiên.")

def script_generation_interface():
    """Giao diện tạo script"""
    st.header("✨ Tạo Script Video")
    
    if not st.session_state.generator:
        st.warning("Hệ thống chưa được khởi tạo")
        return
    
    # Lấy danh sách videos
    videos = st.session_state.generator.storage.get_all_videos()
    
    if not videos:
        st.info("📭 Chưa có video nào. Hãy upload videos trước.")
        return
    
    # Chọn videos để tạo script
    st.subheader("📋 Chọn videos cho script")
    
    selected_video_names = st.multiselect(
        "Chọn videos:",
        options=[v.original_name for v in videos],
        default=[v.original_name for v in videos[:3]]  # Mặc định chọn 3 video đầu
    )
    
    if selected_video_names:
        selected_videos = [v for v in videos if v.original_name in selected_video_names]
        
        # Sắp xếp thứ tự
        st.subheader("🔢 Sắp xếp thứ tự cảnh")
        
        use_custom_order = st.checkbox("Tự sắp xếp thứ tự (không check = AI tự động sắp xếp)")
        
        user_orders = {}
        if use_custom_order:
            for video_name in selected_video_names:
                order = st.number_input(
                    f"Thứ tự cho {video_name}:",
                    min_value=1,
                    max_value=len(selected_video_names),
                    value=selected_video_names.index(video_name) + 1,
                    key=f"order_{video_name}"
                )
                user_orders[video_name] = order
        
        # Cấu hình script
        st.subheader("📝 Cấu hình Script")
        
        col1, col2 = st.columns(2)
        
        with col1:
            main_prompt = st.text_area(
                "Chủ đề script (prompt chính):",
                value="Tạo video giới thiệu sự kiện chuyên nghiệp",
                height=100,
                help="Mô tả chủ đề, mục đích của video script"
            )
        
        with col2:
            tone = st.text_input(
                "Tone/Giọng điệu:",
                value="chuyên nghiệp",
                placeholder="Nhập mô tả tone (ví dụ: Nhiệt huyết, truyền cảm hứng, nghiêm túc, chuyên nghiệp, hướng tới đối tượng trẻ em)",
                help="Mô tả chi tiết tone voiceover mà bạn mong muốn. Hệ thống sẽ truyền trực tiếp vào model AI."
            )
            
            preview_mode = st.checkbox("Chế độ preview (nhanh hơn)")
        
        # Tạo script
        if st.button("🎬 Tạo Script", type="primary"):
            generate_script(selected_videos, user_orders, main_prompt, tone, preview_mode)
    
    # Hiển thị script hiện tại
    if st.session_state.current_script:
        st.subheader("📜 Script hiện tại")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            script_container = st.container()
            with script_container:
                st.text_area(
                    "Script content:",
                    value=st.session_state.current_script,
                    height=600,
                    key="script_display"
                )
        
        with col2:
            # Chỉnh sửa script
            st.subheader("✏️ Chỉnh sửa")
            
            edit_prompt = st.text_area(
                "Yêu cầu chỉnh sửa:",
                placeholder="VD: Làm voiceover cảnh 1 dài hơn, thêm cảm xúc cho cảnh cuối...",
                height=100
            )
            
            if st.button("✏️ Chỉnh sửa") and edit_prompt:
                edit_script(edit_prompt, main_prompt, tone)
            
            # Download script
            if st.download_button(
                "📥 Download Script",
                data=st.session_state.current_script,
                file_name=f"video_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            ):
                st.success("📥 Script đã được download!")

def generate_script(selected_videos, user_orders, main_prompt, tone, preview_mode):
    """Tạo script từ videos đã chọn"""
    try:
        with st.spinner("🎬 Đang tạo script..."):
            video_ids = [v.id for v in selected_videos]
            
            script = st.session_state.generator.generate_script_from_storage(
                video_ids, user_orders, main_prompt, tone
            )
            
            st.session_state.current_script = script
            
            # Lưu script history
            save_script_to_history(script, main_prompt, tone, [v.original_name for v in selected_videos])
        
        st.success("✅ Script đã được tạo thành công!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Lỗi tạo script: {e}")

def edit_script(edit_prompt, main_prompt, tone):
    """Chỉnh sửa script hiện tại"""
    try:
        with st.spinner("✏️ Đang chỉnh sửa script..."):
            new_script = st.session_state.generator._edit_script(
                st.session_state.current_script, edit_prompt, main_prompt, tone
            )
            st.session_state.current_script = new_script
        
        st.success("✅ Script đã được chỉnh sửa!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Lỗi chỉnh sửa: {e}")

def analyze_single_video(video_id):
    """Phân tích lại một video"""
    try:
        with st.spinner("🔍 Đang phân tích video..."):
            video_path = st.session_state.generator.storage.get_video_for_analysis(video_id)
            if video_path:
                analysis = st.session_state.generator.analyzer.analyze_video_robust(video_path)
                st.session_state.generator.storage.update_analysis_result(video_id, analysis)
        
        st.success("✅ Phân tích hoàn thành!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Lỗi phân tích: {e}")

def compress_single_video(video_id):
    """Nén một video"""
    try:
        with st.spinner("🗜️ Đang nén video..."):
            success = st.session_state.generator.storage.compress_and_store(video_id)
            
        if success:
            st.success("✅ Nén video thành công!")
        else:
            st.warning("⚠️ Không thể nén video")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Lỗi nén: {e}")

def delete_single_video(video_id):
    """Xóa một video"""
    try:
        success = st.session_state.generator.storage.delete_video(video_id)
        
        if success:
            st.success("✅ Đã xóa video!")
        else:
            st.error("❌ Không thể xóa video")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Lỗi xóa: {e}")

def cleanup_old_videos(days_old):
    """Dọn dẹp videos cũ"""
    try:
        with st.spinner(f"🧹 Đang dọn dẹp videos cũ hơn {days_old} ngày..."):
            st.session_state.generator.storage.cleanup_old_videos(days_old)
        
        st.success("✅ Dọn dẹp hoàn thành!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Lỗi dọn dẹp: {e}")

def analyze_all_unprocessed():
    """Phân tích tất cả videos chưa xử lý"""
    try:
        videos = st.session_state.generator.storage.get_all_videos()
        unprocessed = [v for v in videos if not v.analysis_result]
        
        if not unprocessed:
            st.info("📋 Tất cả videos đã được phân tích")
            return
        
        with st.spinner(f"🔍 Đang phân tích {len(unprocessed)} videos..."):
            video_ids = [v.id for v in unprocessed]
            st.session_state.generator.analyze_stored_videos(video_ids)
        
        st.success(f"✅ Đã phân tích {len(unprocessed)} videos!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Lỗi phân tích: {e}")

def script_history_interface():
    """Giao diện xem lịch sử scripts"""
    st.header("📋 Lịch sử Scripts")
    
    # Tạo thư mục lưu scripts nếu chưa có
    scripts_dir = Path("./script_history")
    scripts_dir.mkdir(exist_ok=True)
    
    # Lấy danh sách scripts
    script_files = list(scripts_dir.glob("*.json"))
    
    if script_files:
        # Sắp xếp theo thời gian (mới nhất trước)
        script_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Hiển thị danh sách
        for script_file in script_files:
            try:
                with open(script_file, 'r', encoding='utf-8') as f:
                    script_data = json.load(f)
                
                with st.expander(f"📄 {script_data.get('title', script_file.name)} - {script_data.get('created_time', '')}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Chủ đề:** {script_data.get('main_prompt', 'N/A')}")
                        st.write(f"**Tone:** {script_data.get('tone', 'N/A')}")
                        st.write(f"**Videos:** {', '.join(script_data.get('video_names', []))}")
                        
                        st.text_area(
                            "Script content:",
                            value=script_data.get('script_content', ''),
                            height=200,
                            key=f"script_{script_file.name}"
                        )
                    
                    with col2:
                        if st.button("📥 Download", key=f"download_{script_file.name}"):
                            st.download_button(
                                "📥 Tải xuống",
                                data=script_data.get('script_content', ''),
                                file_name=f"{script_file.stem}.txt",
                                mime="text/plain"
                            )
                        
                        if st.button("🗑️ Xóa", key=f"del_script_{script_file.name}"):
                            script_file.unlink()
                            st.success("✅ Đã xóa script!")
                            st.rerun()
            
            except Exception as e:
                st.error(f"❌ Lỗi đọc script {script_file.name}: {e}")
    
    else:
        st.info("📭 Chưa có script nào được lưu.")

def save_script_to_history(script_content, main_prompt, tone, video_names):
    """Lưu script vào lịch sử"""
    try:
        scripts_dir = Path("./script_history")
        scripts_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_data = {
            "title": f"Script_{timestamp}",
            "created_time": datetime.now().isoformat(),
            "main_prompt": main_prompt,
            "tone": tone,
            "video_names": video_names,
            "script_content": script_content
        }
        
        script_file = scripts_dir / f"script_{timestamp}.json"
        with open(script_file, 'w', encoding='utf-8') as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        st.warning(f"⚠️ Không thể lưu script history: {e}")

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        🎬 <strong>Video Analysis & Script Generator</strong><br>
        Powered by Gemini AI + Claude AI | Made with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()